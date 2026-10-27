from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np

ROOT = Path(__file__).resolve().parent


def prepare():
    with np.load(ROOT / 'data/timeseries.npz', allow_pickle=False) as z:
        source = {k: z[k] for k in z.files}
    specifications = json.loads((ROOT / 'data/cases.json').read_text())
    days = source['month_days']
    outputs = {k: source[k] for k in ['provinces', 'month_days', 'dist_km',
               'latency_ms', 'grid_price', 'grid_ef', 'water_price_cny_per_m3',
               'socio_water_stress_category']}
    for key, spec in specifications['inputs'].items():
        pue = source['pue_q' + str(spec['we_percentile'])]
        wue = source['wue_q' + str(spec['we_percentile'])]
        accessible = spec['access_share'] * (source['wind_hourly'] +
                     source['solar_hourly'] + source['hydro_hourly'])
        demand = source['demand_it_hourly'].copy()
        share = spec['flexible_shift_share']
        if share > 0:
            for i in range(len(demand)):
                for d in range(365):
                    base = demand[i, d].copy()
                    total = float(base.sum())
                    if total <= 0:
                        continue
                    priority = accessible[i, d].copy()
                    if priority.sum() <= 0:
                        priority = source['load_shape'].copy()
                    priority = np.maximum(priority, 0.0)
                    priority = priority / priority.sum()
                    demand[i, d] = (1.0 - share) * base + share * total * priority
        values = dict(pue_rep=pue, wue_rep=wue, demand_it_rep=demand,
                      wind_rep=source['wind_hourly'], solar_rep=source['solar_hourly'],
                      hydro_rep=source['hydro_hourly'], accessible_green_rep=accessible)
        for name, hourly in values.items():
            rep = np.zeros((31, 12, 24), dtype=float)
            start = 0
            for month, count in enumerate(days):
                rep[:, month] = hourly[:, start:start + int(count)].mean(axis=1)
                start += int(count)
            outputs[key + '__' + name] = rep
    path = ROOT / 'work/model_inputs.npz'
    path.parent.mkdir(exist_ok=True)
    np.savez_compressed(path, **outputs)
    return path


def load_input(key):
    path = ROOT / 'work/model_inputs.npz'
    if not path.exists():
        prepare()
    with np.load(path, allow_pickle=False) as z:
        values = {k: z[k] for k in z.files if '__' not in k}
        values.update({k.split('__', 1)[1]: z[k] for k in z.files if k.startswith(key + '__')})
    provinces = values.pop('provinces').tolist()
    values.update(provinces=provinces, n_provinces=len(provinces), n_months=12,
                  total_it_peak_mw=float(values['demand_it_rep'].sum(axis=0).max()))
    return SimpleNamespace(**values)


if __name__ == '__main__':
    print(prepare())
