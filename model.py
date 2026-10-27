from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Dict, Tuple

import numpy as np
from scipy.optimize import milp, linprog, LinearConstraint, Bounds
from scipy.sparse import csr_matrix

@dataclass
class HourlyConfig:
    scale_names: tuple = ("medium", "large", "ultra")
    capacity_mw: tuple = (50, 100, 200)
    utilization: float = 0.8
    green_price_discount: float = 0.8
    alpha_route_cny_per_mwh_km: float = 0.01
    latency_tiers_ms: tuple = (10, 30, 100)
    demand_tier_shares: tuple = (0.10, 0.50, 0.20, 0.20)
    mu_share_cap: float = 0.30
    reserve_margin: float = 1.12
    clean_share_band: float = 0.02
    time_limit_s: float = 120
    mip_gap: float = 0.01
    storage_cost_cny_per_mwh: float = 0.0
    storage_eta_charge: float = 0.92
    storage_eta_discharge: float = 0.92
    storage_power_ratio: float = 0.0508
    storage_duration_h: float = 4.0
    tou_peak_multiplier: float = 1.8
    tou_valley_multiplier: float = 0.4
    flexible_share: float = 0.0
    flexible_hourly_cap_multiplier: float = 3.0
    province_share_cap_multipliers: tuple | None = None
    province_share_cap_profile_name: str = "uniform"
    water_pressure_basis: str = "none"

class HourlySpatiotemporalModel:
    def __init__(self, data, scenario: str = "S1", omega: float = 0.0, mu_share=None, config: HourlyConfig | None = None):
        self.data = data
        self.scenario = scenario
        self.omega = omega
        self.cfg = config if config else HourlyConfig()
        if mu_share is not None:
            self.cfg.mu_share_cap = mu_share

        self.n = data.n_provinces
        self.nm = data.n_months
        self.nh = 24
        self.ns = len(self.cfg.scale_names)
        self.cap = np.asarray(self.cfg.capacity_mw, dtype=float)
        self.spatial_on = scenario == "S1"
        self.storage_enabled = self.cfg.storage_power_ratio > 0 and self.cfg.storage_duration_h > 0
        self.storage_design_pue = np.max(np.asarray(self.data.pue_rep, dtype=float), axis=(1, 2))
        self.water_fixed_cost = 0.0
        if self.cfg.province_share_cap_multipliers is None:
            self.province_share_cap_multipliers = np.ones(self.n, dtype=float)
        else:
            self.province_share_cap_multipliers = np.asarray(self.cfg.province_share_cap_multipliers, dtype=float)
            if self.province_share_cap_multipliers.shape != (self.n,):
                raise ValueError("province_share_cap_multipliers must have one entry per province.")

        self._var_start = {}
        self._var_count = 0
        self.c = []
        self.lb = []
        self.ub = []
        self.integrality = []

        self._build_vars()
        self._build_constraints()

    def _reg(self, name: str, n_var: int, lb=0.0, ub=np.inf, integer=False):
        start = self._var_count
        self._var_start[name] = start
        self._var_count += n_var
        self.c.extend([0.0] * n_var)
        self.lb.extend([lb] * n_var)
        self.ub.extend([ub] * n_var)
        self.integrality.extend([1 if integer else 0] * n_var)
        return start, n_var

    def _obj(self, name: str, idx: int, val: float):
        self.c[self._var_start[name] + idx] = val

    def _build_vars(self):
        n, nm, nh, ns = self.n, self.nm, self.nh, self.ns

        self._reg("x", n * ns, lb=0, ub=80, integer=False)

        self.z_idx: Dict[Tuple[int, int, int, int], int] = {}
        nz = 0
        if self.spatial_on:
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    for m in range(nm):
                        for h in range(nh):
                            self.z_idx[(i, j, m, h)] = nz
                            nz += 1
        self._reg("z", max(nz, 1), lb=0.0)

        self._reg("g", n * nm * nh, lb=0.0)
        self._reg("rw", n * nm * nh, lb=0.0)
        self._reg("rs", n * nm * nh, lb=0.0)
        self._reg("rh", n * nm * nh, lb=0.0)
        self._reg("charge", n * nm * nh, lb=0.0)
        self._reg("discharge", n * nm * nh, lb=0.0)
        self._reg("soc", n * nm * nh, lb=0.0)
        self._reg("flex", n * nm * nh, lb=0.0)

        for i in range(n):
            for m in range(nm):
                day_weight = float(self.data.month_days[m])
                for h in range(nh):
                    idx = (i * nm + m) * nh + h
                    water_coeff = float(self.data.water_price_cny_per_m3[i] * self.data.wue_rep[i, m, h])
                    base_it = float(self.data.demand_it_rep[i, m, h])
                    grid_coeff = float(self.data.grid_price[i, m]) * self._grid_tou_multiplier(m, h) * day_weight
                    self._obj("g", idx, grid_coeff)
                    green_cost = float(self.data.grid_price[i, m]) * self.cfg.green_price_discount * day_weight
                    self._obj("rw", idx, green_cost)
                    self._obj("rs", idx, green_cost)
                    self._obj("rh", idx, green_cost)
                    self._obj("discharge", idx, float(self.cfg.storage_cost_cny_per_mwh) * day_weight)
                    self._obj("flex", idx, water_coeff * day_weight)
                    self.water_fixed_cost += water_coeff * base_it * day_weight
                if self.spatial_on:
                    for j in range(n):
                        if i == j:
                            continue
                        for h in range(nh):
                            zi = self.z_idx[(i, j, m, h)]
                            water_delta = (
                                float(self.data.water_price_cny_per_m3[i] * self.data.wue_rep[i, m, h])
                                - float(self.data.water_price_cny_per_m3[j] * self.data.wue_rep[j, m, h])
                            )

                            route_coeff = float(self.data.dist_km[i, j]) * float(self.cfg.alpha_route_cny_per_mwh_km)
                            coeff = (route_coeff + water_delta) * day_weight
                            self._obj("z", zi, coeff)

    def _build_constraints(self):
        rows, lbs, ubs = [], [], []

        def add(row_dict, lb, ub):
            rows.append(row_dict)
            lbs.append(lb)
            ubs.append(ub)

        self._c_demand_balance(add)
        self._c_power_balance(add)
        self._c_renewable_cap(add)
        self._c_grid_cannot_charge_storage(add)
        self._c_storage_state(add)
        self._c_flexible_demand(add)
        self._c_no_transit_exports(add)
        self._c_total_import_cap(add)
        self._c_latency_tiered_imports(add)
        self._c_omega(add)
        self._c_capacity_total(add)
        self._c_share_cap(add)

        data, indices, indptr = [], [], [0]
        for row in rows:
            for idx, val in row.items():
                if 0 <= idx < self._var_count:
                    data.append(val)
                    indices.append(idx)
            indptr.append(len(data))
        A = csr_matrix((data, indices, indptr), shape=(len(rows), self._var_count))
        self.constraints = LinearConstraint(A, np.asarray(lbs, dtype=float), np.asarray(ubs, dtype=float))

    def _x_cap_expr(self, i: int) -> Dict[int, float]:
        x0 = self._var_start["x"]
        return {
            x0 + i * self.ns + s: float(self.cap[s] * self.cfg.utilization)
            for s in range(self.ns)
        }

    def _grid_tou_multiplier(self, month_idx: int, hour_idx: int) -> float:
        peak = float(self.cfg.tou_peak_multiplier)
        valley = float(self.cfg.tou_valley_multiplier)
        month = month_idx + 1
        hour = int(hour_idx)
        if month in {7, 8, 9}:
            if 16 <= hour <= 23:
                return peak
            if hour in {2, 3, 4, 5, 6, 7, 8, 11, 12}:
                return valley
            return 1.0
        if month in {12, 1}:
            if 15 <= hour <= 22:
                return peak
            if hour in {0, 1, 2, 3, 4, 5, 12, 13, 23}:
                return valley
            return 1.0
        if hour in {6, 7, 16, 17, 18, 19, 20, 21}:
            return peak
        if hour in {0, 1, 2, 3, 4, 5, 11, 12, 13, 23}:
            return valley
        return 1.0

    def _c_demand_balance(self, add):
        x0 = self._var_start["x"]
        z0 = self._var_start["z"]
        f0 = self._var_start["flex"]
        for i in range(self.n):
            for m in range(self.nm):
                for h in range(self.nh):
                    row = {}
                    for s in range(self.ns):
                        row[x0 + i * self.ns + s] = float(self.cap[s] * self.cfg.utilization)
                    if self.spatial_on:
                        for j in range(self.n):
                            if (j, i, m, h) in self.z_idx:
                                row[z0 + self.z_idx[(j, i, m, h)]] = 1.0
                            if (i, j, m, h) in self.z_idx:
                                row[z0 + self.z_idx[(i, j, m, h)]] = -1.0
                    idx = (i * self.nm + m) * self.nh + h
                    row[f0 + idx] = -1.0
                    base_it = float(self.data.demand_it_rep[i, m, h])
                    fixed_it = (1.0 - self.cfg.flexible_share) * base_it
                    add(row, fixed_it, np.inf)

    def _c_power_balance(self, add):
        g0 = self._var_start["g"]
        rw0 = self._var_start["rw"]
        rs0 = self._var_start["rs"]
        rh0 = self._var_start["rh"]
        z0 = self._var_start["z"]
        d0 = self._var_start["discharge"]
        f0 = self._var_start["flex"]
        c0 = self._var_start["charge"]
        f0 = self._var_start["flex"]
        for i in range(self.n):
            for m in range(self.nm):
                for h in range(self.nh):
                    row = {}
                    idx = (i * self.nm + m) * self.nh + h
                    row[g0 + idx] = 1.0
                    row[rw0 + idx] = 1.0
                    row[rs0 + idx] = 1.0
                    row[rh0 + idx] = 1.0
                    row[d0 + idx] = 1.0
                    row[c0 + idx] = -1.0
                    pue = float(self.data.pue_rep[i, m, h])
                    base_it = float(self.data.demand_it_rep[i, m, h])
                    fixed_it = (1.0 - self.cfg.flexible_share) * base_it
                    row[f0 + idx] = -pue
                    row_const = -pue * fixed_it
                    if self.spatial_on:
                        for j in range(self.n):
                            if (i, j, m, h) in self.z_idx:
                                row[z0 + self.z_idx[(i, j, m, h)]] = row.get(z0 + self.z_idx[(i, j, m, h)], 0.0) - pue
                            if (j, i, m, h) in self.z_idx:
                                row[z0 + self.z_idx[(j, i, m, h)]] = row.get(z0 + self.z_idx[(j, i, m, h)], 0.0) + pue
                    add(row, -row_const, -row_const)

    def _c_renewable_cap(self, add):
        rw0 = self._var_start["rw"]
        rs0 = self._var_start["rs"]
        rh0 = self._var_start["rh"]
        for i in range(self.n):
            for m in range(self.nm):
                for h in range(self.nh):
                    idx = (i * self.nm + m) * self.nh + h
                    wind_cap = float(self.data.wind_rep[i, m, h])
                    solar_cap = float(self.data.solar_rep[i, m, h])
                    hydro_cap = float(self.data.hydro_rep[i, m, h])
                    raw_total = max(wind_cap + solar_cap + hydro_cap, 1e-12)
                    accessible_total = max(float(self.data.accessible_green_rep[i, m, h]), 0.0)
                    access_share = min(accessible_total / raw_total, 1.0)
                    add({rw0 + idx: 1.0}, 0.0, max(wind_cap * access_share, 0.0))
                    add({rs0 + idx: 1.0}, 0.0, max(solar_cap * access_share, 0.0))
                    add({rh0 + idx: 1.0}, 0.0, max(hydro_cap * access_share, 0.0))

    def _c_grid_cannot_charge_storage(self, add):
        c0 = self._var_start["charge"]
        rw0 = self._var_start["rw"]
        rs0 = self._var_start["rs"]
        rh0 = self._var_start["rh"]
        for i in range(self.n):
            for m in range(self.nm):
                for h in range(self.nh):
                    idx = (i * self.nm + m) * self.nh + h

                    add(
                        {
                            c0 + idx: 1.0,
                            rw0 + idx: -1.0,
                            rs0 + idx: -1.0,
                            rh0 + idx: -1.0,
                        },
                        -np.inf,
                        0.0,
                    )

    def _c_storage_state(self, add):
        c0 = self._var_start["charge"]
        d0 = self._var_start["discharge"]
        s0 = self._var_start["soc"]
        if not self.storage_enabled:
            for i in range(self.n):
                for m in range(self.nm):
                    for h in range(self.nh):
                        idx = (i * self.nm + m) * self.nh + h
                        add({c0 + idx: 1.0}, 0.0, 0.0)
                        add({d0 + idx: 1.0}, 0.0, 0.0)
                        add({s0 + idx: 1.0}, 0.0, 0.0)
            return
        eta_c = float(self.cfg.storage_eta_charge)
        eta_d = float(self.cfg.storage_eta_discharge)
        for i in range(self.n):
            power_coeff = float(self.cfg.storage_power_ratio * self.storage_design_pue[i])
            energy_coeff = float(self.cfg.storage_duration_h * power_coeff)
            capacity_expr = {
                self._var_start["x"] + i * self.ns + scale: float(self.cap[scale])
                for scale in range(self.ns)
            }
            for m in range(self.nm):
                for h in range(self.nh):
                    idx = (i * self.nm + m) * self.nh + h
                    prev_h = (h - 1) % self.nh
                    prev_idx = (i * self.nm + m) * self.nh + prev_h
                    row = {
                        s0 + idx: 1.0,
                        s0 + prev_idx: -1.0,
                        c0 + idx: -eta_c,
                        d0 + idx: 1.0 / eta_d,
                    }
                    add(row, 0.0, 0.0)
                    charge_cap = {c0 + idx: 1.0}
                    discharge_cap = {d0 + idx: 1.0}
                    soc_cap = {s0 + idx: 1.0}
                    for x_idx, cap_mw in capacity_expr.items():
                        charge_cap[x_idx] = -power_coeff * cap_mw
                        discharge_cap[x_idx] = -power_coeff * cap_mw
                        soc_cap[x_idx] = -energy_coeff * cap_mw
                    add(charge_cap, -np.inf, 0.0)
                    add(discharge_cap, -np.inf, 0.0)
                    add(soc_cap, -np.inf, 0.0)

    def _c_flexible_demand(self, add):
        f0 = self._var_start["flex"]
        for i in range(self.n):
            for m in range(self.nm):
                total_flex = 0.0
                avg_flex = 0.0
                for h in range(self.nh):
                    base_it = float(self.data.demand_it_rep[i, m, h])
                    total_flex += self.cfg.flexible_share * base_it
                avg_flex = total_flex / self.nh
                row = {}
                for h in range(self.nh):
                    idx = (i * self.nm + m) * self.nh + h
                    row[f0 + idx] = 1.0
                    add({f0 + idx: 1.0}, 0.0, self.cfg.flexible_hourly_cap_multiplier * avg_flex)
                add(row, total_flex, total_flex)

    def _c_no_transit_exports(self, add):
        if not self.spatial_on:
            return
        z0 = self._var_start["z"]
        for i in range(self.n):
            x_cap = self._x_cap_expr(i)
            for m in range(self.nm):
                for h in range(self.nh):
                    row = {}
                    has_flow = False
                    for j in range(self.n):
                        if (i, j, m, h) in self.z_idx:
                            row[z0 + self.z_idx[(i, j, m, h)]] = 1.0
                            has_flow = True
                    if has_flow:
                        row.update({k: row.get(k, 0.0) - v for k, v in x_cap.items()})
                        add(row, -np.inf, 0.0)

    def _c_total_import_cap(self, add):
        if not self.spatial_on:
            return
        z0 = self._var_start["z"]
        for j in range(self.n):
            for m in range(self.nm):
                for h in range(self.nh):
                    row = {}
                    idx = (j * self.nm + m) * self.nh + h
                    row[self._var_start["flex"] + idx] = -1.0
                    for i in range(self.n):
                        if (i, j, m, h) in self.z_idx:
                            row[z0 + self.z_idx[(i, j, m, h)]] = row.get(z0 + self.z_idx[(i, j, m, h)], 0.0) + 1.0
                    if row:
                        fixed_demand = (1.0 - self.cfg.flexible_share) * float(self.data.demand_it_rep[j, m, h])
                        add(row, -np.inf, fixed_demand)

    def _c_latency_tiered_imports(self, add):
        if not self.spatial_on:
            return
        z0 = self._var_start["z"]
        hot, warm, warm_cold, cold = self.cfg.demand_tier_shares
        thresholds = [
            (self.cfg.latency_tiers_ms[0], warm + warm_cold + cold),
            (self.cfg.latency_tiers_ms[1], warm_cold + cold),
            (self.cfg.latency_tiers_ms[2], cold),
        ]
        for j in range(self.n):
            for m in range(self.nm):
                for h in range(self.nh):
                    base_demand = float(self.data.demand_it_rep[j, m, h])
                    fixed_demand = (1.0 - self.cfg.flexible_share) * base_demand
                    for threshold_ms, share_cap in thresholds:
                        row = {}
                        idx = (j * self.nm + m) * self.nh + h
                        row[self._var_start["flex"] + idx] = -share_cap
                        for i in range(self.n):
                            if (i, j, m, h) in self.z_idx and self.data.latency_ms[i, j] > threshold_ms:
                                row[z0 + self.z_idx[(i, j, m, h)]] = 1.0
                        if row:
                            add(row, -np.inf, share_cap * fixed_demand)

    def _c_omega(self, add):
        g0 = self._var_start["g"]
        rw0 = self._var_start["rw"]
        rs0 = self._var_start["rs"]
        rh0 = self._var_start["rh"]
        c0 = self._var_start["charge"]
        d0 = self._var_start["discharge"]
        lower = max(0.0, float(self.omega) - float(self.cfg.clean_share_band))
        upper = min(1.0, float(self.omega) + float(self.cfg.clean_share_band))
        if lower > 1e-9:
            row = {}
            for i in range(self.n):
                for m in range(self.nm):
                    day_weight = float(self.data.month_days[m])
                    for h in range(self.nh):
                        idx = (i * self.nm + m) * self.nh + h
                        row[g0 + idx] = -lower * day_weight
                        row[rw0 + idx] = (1.0 - lower) * day_weight
                        row[rs0 + idx] = (1.0 - lower) * day_weight
                        row[rh0 + idx] = (1.0 - lower) * day_weight
                        row[c0 + idx] = -(1.0 - lower) * day_weight
                        row[d0 + idx] = (1.0 - lower) * day_weight
            add(row, 0.0, np.inf)
        if upper < 1.0 - 1e-9:
            row = {}
            for i in range(self.n):
                for m in range(self.nm):
                    day_weight = float(self.data.month_days[m])
                    for h in range(self.nh):
                        idx = (i * self.nm + m) * self.nh + h
                        row[g0 + idx] = -upper * day_weight
                        row[rw0 + idx] = (1.0 - upper) * day_weight
                        row[rs0 + idx] = (1.0 - upper) * day_weight
                        row[rh0 + idx] = (1.0 - upper) * day_weight
                        row[c0 + idx] = -(1.0 - upper) * day_weight
                        row[d0 + idx] = (1.0 - upper) * day_weight
            add(row, -np.inf, 0.0)

    def _c_capacity_total(self, add):
        peak = float(self.data.total_it_peak_mw)
        row = {}
        x0 = self._var_start["x"]
        for i in range(self.n):
            for s in range(self.ns):
                row[x0 + i * self.ns + s] = float(self.cap[s] * self.cfg.utilization)
        add(row, peak, peak * self.cfg.reserve_margin)

    def _c_share_cap(self, add):
        x0 = self._var_start["x"]
        for i in range(self.n):
            cap_limit = float(
                self.cfg.mu_share_cap
                * self.data.total_it_peak_mw
                * self.province_share_cap_multipliers[i]
            )
            row = {}
            for s in range(self.ns):
                row[x0 + i * self.ns + s] = float(self.cap[s] * self.cfg.utilization)
            add(row, 0.0, cap_limit)

    def solve(self):
        c = np.asarray(self.c, dtype=float)
        bounds = Bounds(np.asarray(self.lb, dtype=float), np.asarray(self.ub, dtype=float))
        integrality = np.asarray(self.integrality, dtype=int)
        t0 = time.time()
        try:
            if np.any(integrality != 0):
                result = milp(
                    c=c,
                    integrality=integrality,
                    bounds=bounds,
                    constraints=self.constraints,
                    options={
                        "time_limit": float(self.cfg.time_limit_s),
                        "disp": False,
                        "mip_rel_gap": float(self.cfg.mip_gap),
                    },
                )
            else:
                A = self.constraints.A.tocsr()
                lb = np.asarray(self.constraints.lb, dtype=float)
                ub = np.asarray(self.constraints.ub, dtype=float)
                eq_mask = np.isfinite(lb) & np.isfinite(ub) & (np.abs(lb - ub) < 1e-9)
                ub_mask = np.isfinite(ub) & ~eq_mask
                lb_mask = np.isfinite(lb) & ~eq_mask
                A_eq = A[eq_mask]
                b_eq = ub[eq_mask]
                if np.any(ub_mask) or np.any(lb_mask):
                    pieces = []
                    rhs = []
                    if np.any(ub_mask):
                        pieces.append(A[ub_mask])
                        rhs.append(ub[ub_mask])
                    if np.any(lb_mask):
                        pieces.append(-A[lb_mask])
                        rhs.append(-lb[lb_mask])
                    from scipy.sparse import vstack
                    A_ub = vstack(pieces)
                    b_ub = np.concatenate(rhs)
                else:
                    A_ub = None
                    b_ub = None
                lp_bounds = list(zip(np.asarray(self.lb, dtype=float), np.asarray(self.ub, dtype=float)))
                lp_options = {
                    "time_limit": float(self.cfg.time_limit_s),
                    "disp": False,
                }
                result = linprog(
                    c=c,
                    A_ub=A_ub,
                    b_ub=b_ub,
                    A_eq=A_eq,
                    b_eq=b_eq,
                    bounds=lp_bounds,
                    method="highs",
                    options=lp_options,
                )

                if not result.success and int(getattr(result, "status", -1)) == 4:
                    result = linprog(
                        c=c,
                        A_ub=A_ub,
                        b_ub=b_ub,
                        A_eq=A_eq,
                        b_eq=b_eq,
                        bounds=lp_bounds,
                        method="highs-ipm",
                        options=lp_options,
                    )
        except Exception as exc:
            return False, {"error": str(exc), "solve_status": "error", "solver_status_code": None, "solve_time_s": time.time() - t0}
        elapsed = time.time() - t0
        if not result.success:
            message = result.message or "solver failure"
            message_lower = message.lower()
            if "time limit" in message_lower:
                solve_status = "time_limit"
            elif "infeasible" in message_lower:
                solve_status = "infeasible"
            else:
                solve_status = "solver_failure"
            return False, {
                "error": message,
                "solve_status": solve_status,
                "solver_status_code": int(result.status),
                "solve_time_s": elapsed,
            }
        metrics = self._extract(result.x)
        metrics["objective"] = float(result.fun + self.water_fixed_cost)
        metrics["solve_time_s"] = elapsed
        metrics["success"] = True
        metrics["solve_status"] = "optimal"
        metrics["solver_status_code"] = int(result.status)
        return True, metrics

    def _extract(self, x):
        data = self.data
        x0 = self._var_start["x"]
        g0 = self._var_start["g"]
        rw0 = self._var_start["rw"]
        rs0 = self._var_start["rs"]
        rh0 = self._var_start["rh"]
        z0 = self._var_start["z"]
        d0 = self._var_start["discharge"]
        c0 = self._var_start["charge"]
        s0 = self._var_start["soc"]
        f0 = self._var_start["flex"]

        builds = []
        province_capacity_mw = np.zeros(self.n, dtype=float)
        for i in range(self.n):
            for s in range(self.ns):
                val = float(x[x0 + i * self.ns + s])
                if val > 1e-6:
                    mw = val * float(self.cap[s])
                    province_capacity_mw[i] += mw
                    builds.append({
                        "province": data.provinces[i],
                        "scale": self.cfg.scale_names[s],
                        "count": val,
                        "mw": mw,
                    })

        grid = np.zeros((self.n, self.nm, self.nh), dtype=float)
        wind_used = np.zeros((self.n, self.nm, self.nh), dtype=float)
        solar_used = np.zeros((self.n, self.nm, self.nh), dtype=float)
        hydro_used = np.zeros((self.n, self.nm, self.nh), dtype=float)
        discharge = np.zeros((self.n, self.nm, self.nh), dtype=float)
        charge = np.zeros((self.n, self.nm, self.nh), dtype=float)
        soc = np.zeros((self.n, self.nm, self.nh), dtype=float)
        flex = np.zeros((self.n, self.nm, self.nh), dtype=float)
        for i in range(self.n):
            for m in range(self.nm):
                for h in range(self.nh):
                    idx = (i * self.nm + m) * self.nh + h
                    grid[i, m, h] = x[g0 + idx]
                    wind_used[i, m, h] = x[rw0 + idx]
                    solar_used[i, m, h] = x[rs0 + idx]
                    hydro_used[i, m, h] = x[rh0 + idx]
                    discharge[i, m, h] = x[d0 + idx]
                    charge[i, m, h] = x[c0 + idx]
                    soc[i, m, h] = x[s0 + idx]
                    flex[i, m, h] = x[f0 + idx]

        renew = wind_used + solar_used + hydro_used
        weights = self.data.month_days[np.newaxis, :, np.newaxis]
        total_grid = float((grid * weights).sum())
        total_wind_used = float((wind_used * weights).sum())
        total_solar_used = float((solar_used * weights).sum())
        total_hydro_used = float((hydro_used * weights).sum())
        total_renew = float((renew * weights).sum())
        total_charge = float((charge * weights).sum())
        total_discharge = float((discharge * weights).sum())
        total_clean_delivered = total_renew - total_charge + total_discharge
        total_energy = total_grid + total_clean_delivered
        carbon_tco2 = float((grid * self.data.grid_ef[:, :, np.newaxis] * weights).sum())

        spatial_flow = np.zeros((self.n, self.n, self.nm, self.nh), dtype=float)
        if self.spatial_on:
            for (i, j, m, h), zi in self.z_idx.items():
                spatial_flow[i, j, m, h] = x[z0 + zi]

        total_transfer_it_mwh = float((spatial_flow * weights[np.newaxis, :, :]).sum())
        gross_inflow = spatial_flow.sum(axis=0)
        actual_it_demand = (1.0 - self.cfg.flexible_share) * data.demand_it_rep + flex
        with np.errstate(divide="ignore", invalid="ignore"):
            import_share = np.divide(
                gross_inflow,
                actual_it_demand,
                out=np.zeros_like(gross_inflow),
                where=actual_it_demand > 0,
            )
        latency_over = {}
        for threshold in self.cfg.latency_tiers_ms:
            inbound_over = np.zeros((self.n, self.nm, self.nh), dtype=float)
            if self.spatial_on:
                for (i, j, m, h), zi in self.z_idx.items():
                    if data.latency_ms[i, j] > threshold:
                        inbound_over[j, m, h] += x[z0 + zi]
            with np.errstate(divide="ignore", invalid="ignore"):
                share_over = np.divide(
                    inbound_over,
                    actual_it_demand,
                    out=np.zeros_like(inbound_over),
                    where=actual_it_demand > 0,
                )
            latency_over[str(threshold)] = float(np.nanmax(share_over)) if share_over.size else 0.0

        accessible_total = float((data.accessible_green_rep * weights).sum())
        national_grid_rep = grid.sum(axis=0)
        national_wind_used_rep = wind_used.sum(axis=0)
        national_solar_used_rep = solar_used.sum(axis=0)
        national_hydro_used_rep = hydro_used.sum(axis=0)
        national_renew_rep = renew.sum(axis=0)
        national_storage_rep = discharge.sum(axis=0)
        national_storage_charge_rep = charge.sum(axis=0)
        national_storage_soc_rep = soc.sum(axis=0)
        national_it_rep = actual_it_demand.sum(axis=0)
        national_transfer_rep = spatial_flow.sum(axis=(0, 1)) if self.spatial_on else np.zeros((self.nm, self.nh), dtype=float)
        month_weights = self.data.month_days[np.newaxis, :, np.newaxis]
        monthly_import_mwh = gross_inflow * month_weights
        monthly_export_mwh = spatial_flow.sum(axis=1) * month_weights
        monthly_net_import_mwh = monthly_import_mwh.sum(axis=2) - monthly_export_mwh.sum(axis=2)
        pair_weights = self.data.month_days[np.newaxis, np.newaxis, :, np.newaxis]
        annual_pair_flow_mwh = (spatial_flow * pair_weights).sum(axis=(2, 3))
        monthly_pair_flow_mwh = (spatial_flow * pair_weights).sum(axis=3)
        hourly_pair_flow_records = []
        if self.spatial_on:
            nz = np.argwhere(spatial_flow > 1e-6)
            for i, j, m, h in nz:
                hourly_pair_flow_records.append(
                    {
                        "source": data.provinces[int(i)],
                        "target": data.provinces[int(j)],
                        "month": int(m) + 1,
                        "hour": int(h),
                        "mwh_it_per_representative_hour": float(spatial_flow[int(i), int(j), int(m), int(h)]),
                        "monthly_mwh_it": float(spatial_flow[int(i), int(j), int(m), int(h)] * self.data.month_days[int(m)]),
                    }
                )
        annual_it_demand_mwh = (actual_it_demand * weights).sum(axis=(1, 2))
        executed_it = actual_it_demand + spatial_flow.sum(axis=1) - gross_inflow
        facility_load = executed_it * self.data.pue_rep
        annual_facility_demand_mwh = (facility_load * weights).sum(axis=(1, 2))
        water_m3 = executed_it * self.data.wue_rep * weights
        annual_water_m3 = water_m3.sum(axis=(1, 2))
        cost_water_by_province = annual_water_m3 * self.data.water_price_cny_per_m3
        total_water_m3 = float(annual_water_m3.sum())
        cost_water = float(cost_water_by_province.sum())
        storage_power_capacity_mw = self.cfg.storage_power_ratio * self.storage_design_pue * province_capacity_mw
        storage_energy_capacity_mwh = self.cfg.storage_duration_h * storage_power_capacity_mw
        annual_storage_discharge = (discharge * weights).sum(axis=(1, 2))
        equivalent_full_cycles = np.divide(
            annual_storage_discharge,
            storage_energy_capacity_mwh,
            out=np.zeros_like(annual_storage_discharge),
            where=storage_energy_capacity_mwh > 1e-9,
        )
        simultaneous_max_mw = float(np.max(np.minimum(charge, discharge)))
        soc_step_residual = np.zeros_like(soc)
        for h in range(self.nh):
            soc_step_residual[:, :, h] = (
                soc[:, :, h]
                - soc[:, :, (h - 1) % self.nh]
                - self.cfg.storage_eta_charge * charge[:, :, h]
                + discharge[:, :, h] / self.cfg.storage_eta_discharge
            )
        power_balance_residual = grid + renew + discharge - charge - facility_load

        return {
            "scenario": self.scenario,
            "omega": self.omega,
            "builds": builds,
            "province_mw": {data.provinces[i]: float(province_capacity_mw[i]) for i in range(self.n)},
            "grid_mwh": {data.provinces[i]: (grid[i] * self.data.month_days[:, None]).sum(axis=1).tolist() for i in range(self.n)},
            "renew_mwh": {data.provinces[i]: (renew[i] * self.data.month_days[:, None]).sum(axis=1).tolist() for i in range(self.n)},
            "wind_used_mwh": {data.provinces[i]: (wind_used[i] * self.data.month_days[:, None]).sum(axis=1).tolist() for i in range(self.n)},
            "solar_used_mwh": {data.provinces[i]: (solar_used[i] * self.data.month_days[:, None]).sum(axis=1).tolist() for i in range(self.n)},
            "hydro_used_mwh": {data.provinces[i]: (hydro_used[i] * self.data.month_days[:, None]).sum(axis=1).tolist() for i in range(self.n)},
            "storage_discharge_mwh": {data.provinces[i]: (discharge[i] * self.data.month_days[:, None]).sum(axis=1).tolist() for i in range(self.n)},
            "storage_charge_mwh": {data.provinces[i]: (charge[i] * self.data.month_days[:, None]).sum(axis=1).tolist() for i in range(self.n)},
            "storage_charge_rep_mw": {data.provinces[i]: charge[i].tolist() for i in range(self.n)},
            "storage_discharge_rep_mw": {data.provinces[i]: discharge[i].tolist() for i in range(self.n)},
            "storage_soc_rep_mwh": {data.provinces[i]: soc[i].tolist() for i in range(self.n)},
            "storage_power_capacity_mw": {data.provinces[i]: float(storage_power_capacity_mw[i]) for i in range(self.n)},
            "storage_energy_capacity_mwh": {data.provinces[i]: float(storage_energy_capacity_mwh[i]) for i in range(self.n)},
            "storage_equivalent_full_cycles": {data.provinces[i]: float(equivalent_full_cycles[i]) for i in range(self.n)},
            "flexible_it_mwh": {data.provinces[i]: (flex[i] * self.data.month_days[:, None]).sum(axis=1).tolist() for i in range(self.n)},
            "annual_it_demand_mwh": {data.provinces[i]: float(annual_it_demand_mwh[i]) for i in range(self.n)},
            "annual_facility_demand_mwh": {data.provinces[i]: float(annual_facility_demand_mwh[i]) for i in range(self.n)},
            "annual_water_m3": {data.provinces[i]: float(annual_water_m3[i]) for i in range(self.n)},
            "water_price_cny_per_m3": {data.provinces[i]: float(self.data.water_price_cny_per_m3[i]) for i in range(self.n)},
            "cost_water_by_province": {data.provinces[i]: float(cost_water_by_province[i]) for i in range(self.n)},
            "monthly_net_import_mwh": {data.provinces[i]: monthly_net_import_mwh[i].tolist() for i in range(self.n)},
            "annual_pair_flow_mwh": {
                data.provinces[i]: {
                    data.provinces[j]: float(annual_pair_flow_mwh[i, j])
                    for j in range(self.n)
                    if i != j and annual_pair_flow_mwh[i, j] > 1e-6
                }
                for i in range(self.n)
            },
            "monthly_pair_flow_mwh": {
                f"{data.provinces[i]}->{data.provinces[j]}": monthly_pair_flow_mwh[i, j].tolist()
                for i in range(self.n)
                for j in range(self.n)
                if i != j and monthly_pair_flow_mwh[i, j].sum() > 1e-6
            },
            "hourly_pair_flow_records": hourly_pair_flow_records,
            "national_grid_rep_mwh": national_grid_rep.tolist(),
            "national_wind_used_rep_mwh": national_wind_used_rep.tolist(),
            "national_solar_used_rep_mwh": national_solar_used_rep.tolist(),
            "national_hydro_used_rep_mwh": national_hydro_used_rep.tolist(),
            "national_renew_rep_mwh": national_renew_rep.tolist(),
            "national_storage_rep_mwh": national_storage_rep.tolist(),
            "national_storage_charge_rep_mwh": national_storage_charge_rep.tolist(),
            "national_storage_soc_rep_mwh": national_storage_soc_rep.tolist(),
            "national_it_rep_mwh": national_it_rep.tolist(),
            "national_transfer_rep_it_mwh": national_transfer_rep.tolist(),
            "total_grid_mwh": total_grid,
            "total_wind_used_mwh": total_wind_used,
            "total_solar_used_mwh": total_solar_used,
            "total_hydro_used_mwh": total_hydro_used,
            "total_renew_mwh": total_renew,
            "total_storage_charge_mwh": total_charge,
            "total_storage_discharge_mwh": total_discharge,
            "total_clean_delivered_mwh": total_clean_delivered,
            "clean_share": total_clean_delivered / max(total_energy, 1.0),
            "carbon_tco2": carbon_tco2,
            "cost_electricity": float(
                np.dot(np.asarray(self.c[self._var_start["g"]: self._var_start["g"] + self.n * self.nm * self.nh]), grid.ravel())
                + np.dot(np.asarray(self.c[self._var_start["rw"]: self._var_start["rw"] + self.n * self.nm * self.nh]), wind_used.ravel())
                + np.dot(np.asarray(self.c[self._var_start["rs"]: self._var_start["rs"] + self.n * self.nm * self.nh]), solar_used.ravel())
                + np.dot(np.asarray(self.c[self._var_start["rh"]: self._var_start["rh"] + self.n * self.nm * self.nh]), hydro_used.ravel())
            ),
            "cost_transfer": float(np.dot(np.asarray(self.c[self._var_start["z"]: self._var_start["z"] + max(len(self.z_idx), 1)]), x[z0: z0 + max(len(self.z_idx), 1)])),
            "cost_storage": float(np.dot(np.asarray(self.c[self._var_start["discharge"]: self._var_start["discharge"] + self.n * self.nm * self.nh]), x[self._var_start["discharge"]: self._var_start["discharge"] + self.n * self.nm * self.nh])),
            "cost_water": cost_water,
            "total_water_m3": total_water_m3,
            "total_transfer_it_mwh": total_transfer_it_mwh,
            "accessible_green_utilization": total_renew / max(accessible_total, 1.0),
            "max_imported_demand_share": float(np.nanmax(import_share)) if import_share.size else 0.0,
            "latency_import_share_max": latency_over,
            "demand_tier_shares": list(self.cfg.demand_tier_shares),
            "latency_tiers_ms": list(self.cfg.latency_tiers_ms),
            "clean_share_band": self.cfg.clean_share_band,
            "mu_share_cap": self.cfg.mu_share_cap,
            "storage_cost_cny_per_mwh": self.cfg.storage_cost_cny_per_mwh,
            "storage_power_ratio": self.cfg.storage_power_ratio,
            "storage_duration_h": self.cfg.storage_duration_h,
            "tou_peak_multiplier": self.cfg.tou_peak_multiplier,
            "tou_valley_multiplier": self.cfg.tou_valley_multiplier,
            "storage_eta_charge": self.cfg.storage_eta_charge,
            "storage_eta_discharge": self.cfg.storage_eta_discharge,
            "storage_simultaneous_max_mw": simultaneous_max_mw,
            "storage_soc_residual_max_mwh": float(np.max(np.abs(soc_step_residual))),
            "power_balance_residual_max_mw": float(np.max(np.abs(power_balance_residual))),
            "alpha_route_cny_per_mwh_km": self.cfg.alpha_route_cny_per_mwh_km,
            "green_price_discount": self.cfg.green_price_discount,
            "storage_enabled": self.storage_enabled,
            "flexible_share": self.cfg.flexible_share,
            "province_share_cap_profile_name": self.cfg.province_share_cap_profile_name,
            "water_pressure_basis": self.cfg.water_pressure_basis,
            "province_share_cap_multipliers": {
                data.provinces[i]: float(self.province_share_cap_multipliers[i])
                for i in range(self.n)
            },
            "province_share_cap_mw": {
                data.provinces[i]: float(
                    self.cfg.mu_share_cap
                    * self.data.total_it_peak_mw
                    * self.province_share_cap_multipliers[i]
                )
                for i in range(self.n)
            },
        }

def solve_one(data, scenario="S1", omega=0.0, mu_share=None, config=None):
    model = HourlySpatiotemporalModel(data=data, scenario=scenario, omega=omega, mu_share=mu_share, config=config)
    return model.solve()
