from pathlib import Path
from types import SimpleNamespace
import argparse
import hashlib
import json
import os
import sys

os.environ.setdefault('OMP_NUM_THREADS', '1')
ROOT = Path(__file__).resolve().parent
PACKAGE = ROOT.parent
sys.path.insert(0, str(PACKAGE))
import numpy as np
from model import HourlyConfig, HourlySpatiotemporalModel

SETTINGS = json.loads((ROOT / 'settings.json').read_text())


def input_data(curve):
    for name, expected in SETTINGS['shared_sha256'].items():
        assert hashlib.sha256((PACKAGE / name).read_bytes()).hexdigest() == expected
    with np.load(PACKAGE / 'data/timeseries.npz', allow_pickle=False) as z:
        source = {k: z[k] for k in z.files}
    original = source['demand_it_hourly']
    means = original.mean(axis=2, keepdims=True)
    p = original / means
    profile = np.ones_like(p) if curve == 'flat' else 1 + 0.5 * (p - 1) if curve == 'half' else p
    demand = means * profile
    clean = SETTINGS['access_share'] * (source['wind_hourly'] + source['solar_hourly'] + source['hydro_hourly'])
    totals = clean.sum(axis=2, keepdims=True)
    priority = np.divide(clean, totals, out=np.broadcast_to(profile[0, 0] / 24, clean.shape).copy(), where=totals > 0)
    demand = 0.8 * demand + 0.2 * demand.sum(axis=2, keepdims=True) * priority
    np.testing.assert_allclose(demand.sum(axis=2), original.sum(axis=2), rtol=1e-13, atol=1e-8)
    values = {k: source[k] for k in ['provinces', 'month_days', 'dist_km', 'latency_ms',
              'grid_price', 'grid_ef', 'water_price_cny_per_m3']}
    hourly = dict(demand_it_rep=demand, pue_rep=source['pue_q50'], wue_rep=source['wue_q50'],
                  wind_rep=source['wind_hourly'], solar_rep=source['solar_hourly'],
                  hydro_rep=source['hydro_hourly'], accessible_green_rep=clean)
    for key, array in hourly.items():
        rep = np.empty((31, 12, 24))
        start = 0
        for month, count in enumerate(source['month_days'].astype(int)):
            rep[:, month] = array[:, start:start + count].mean(axis=1)
            start += count
        values[key] = rep
    values.update(provinces=source['provinces'].tolist(), n_provinces=31, n_months=12,
                  total_it_peak_mw=SETTINGS['capacity_reference_peak_mw'])
    return SimpleNamespace(**values)


def solve(key):
    curve, label = key.split('_')
    target = ROOT / 'work' / (key + '.json')
    assert not target.exists(), target
    data = input_data(curve)
    spec = SETTINGS['cases'][label]
    for limit in spec['time_limit_schedule']:
        cfg = HourlyConfig(**dict(spec['config_kwargs'], time_limit_s=limit))
        model = HourlySpatiotemporalModel(data, scenario=spec['scenario'], omega=spec['omega'], config=cfg)
        original_extract = model._extract
        def checked_extract(x):
            lhs = model.constraints.A @ x
            violation = max(float(np.maximum(model.constraints.lb - lhs, 0).max()),
                            float(np.maximum(lhs - model.constraints.ub, 0).max()))
            bound = max(float(np.maximum(np.asarray(model.lb) - x, 0).max()),
                        float(np.maximum(x - np.asarray(model.ub), 0).max()))
            assert np.isfinite(x).all() and max(violation, bound) <= 1e-6
            return original_extract(x)
        model._extract = checked_extract
        ok, result = model.solve()
        if ok or result.get('solve_status') != 'time_limit':
            break
    assert ok, result
    assert abs(result['clean_share'] - spec['omega']) < 1e-8
    assert max(result['power_balance_residual_max_mw'], result['storage_soc_residual_max_mwh']) <= 1e-6
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(key, result['solve_status'], result['objective'])


if __name__ == '__main__':
    keys = [f'{curve}_{case}' for curve in ['flat', 'half'] for case in SETTINGS['cases']]
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', choices=keys)
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if args.case or args.all:
        for key in keys if args.all else [args.case]:
            solve(key)
    else:
        for curve in ['original', 'flat', 'half']:
            input_data(curve)
        print('Inputs PASS; no solve')
