from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
import pyarrow.feather as feather

ROOT = Path(__file__).resolve().parent
SETTINGS = json.loads((ROOT / 'settings.json').read_text())


def tables():
    path = ROOT / 'results.feather'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == SETTINGS['results_sha256']
    raw = feather.read_feather(path)
    return {key: frame.drop(columns='_table').dropna(axis=1, how='all').reset_index(drop=True)
            for key, frame in raw.groupby('_table', sort=False)}


def summarize(work=False):
    data = tables()
    national, province, corridors = [data[k].copy() for k in ['national', 'province', 'corridors']]
    if work:
        from run import input_data
        national = national[national.curve.eq('original')].copy()
        province = province[province.curve.eq('original')].copy()
        corridors = corridors[corridors.curve.eq('original')].copy()
        for curve in ['flat', 'half']:
            inputs = input_data(curve)
            names = inputs.provinces
            for case in SETTINGS['cases']:
                result = json.loads((ROOT / 'work' / f'{curve}_{case}.json').read_text())
                assert result['solve_status'] == 'optimal'
                pair = np.array([[result['annual_pair_flow_mwh'][a].get(b, 0) for b in names] for a in names])
                executed = np.array([result['annual_it_demand_mwh'][p] for p in names]) + pair.sum(1) - pair.sum(0)
                emissions = [np.dot(result['grid_mwh'][p], inputs.grid_ef[i]) for i, p in enumerate(names)]
                national = pd.concat([national, pd.DataFrame([dict(case=case, curve=curve,
                    cost_cny=result['objective'], emissions_tco2=result['carbon_tco2'], annual_it_mwh=executed.sum())])], ignore_index=True)
                province = pd.concat([province, pd.DataFrame(dict(case=case, curve=curve, province=names,
                    executed_it_mwh=executed, emissions_tco2=emissions))], ignore_index=True)
                ordered = sorted([(a, b, pair[i, j]) for i, a in enumerate(names) for j, b in enumerate(names)
                                  if i != j and pair[i, j] > 1e-6], key=lambda x: (-x[2], x[0], x[1]))[:5]
                extra = [dict(case=case, curve=curve, rank=i, host=a, origin=b, annual_it_twh=v / 1e6)
                         for i, (a, b, v) in enumerate(ordered, 1)]
                if extra:
                    corridors = pd.concat([corridors, pd.DataFrame(extra)], ignore_index=True)
    rows = []
    for case in SETTINGS['cases']:
        for curve in ['original', 'flat', 'half']:
            n = national[national.case.eq(case) & national.curve.eq(curve)].iloc[0]
            p = province[province.case.eq(case) & province.curve.eq(curve)]
            c = corridors[corridors.case.eq(case) & corridors.curve.eq(curve)].sort_values('rank')
            assert len(p) == 31 and np.isfinite(p[['executed_it_mwh', 'emissions_tco2']]).all().all()
            np.testing.assert_allclose(p.executed_it_mwh.sum(), n.annual_it_mwh, rtol=1e-12, atol=1e-6)
            np.testing.assert_allclose(p.emissions_tco2.sum(), n.emissions_tco2, rtol=1e-12, atol=1e-6)
            row = dict(case=case, curve=curve, cost_billion_cny=n.cost_cny / 1e9, emissions_mtco2=n.emissions_tco2 / 1e6)
            for region, members in SETTINGS['regions'].items():
                mask = p.province.isin(members)
                assert mask.sum() == 5
                row[region + '_it_pct'] = 100 * p.loc[mask, 'executed_it_mwh'].sum() / p.executed_it_mwh.sum()
                row[region + '_emissions_pct'] = 100 * p.loc[mask, 'emissions_tco2'].sum() / p.emissions_tco2.sum()
            row['top5_host_to_origin_twh_it'] = '; '.join(
                f'{r.host} -> {r.origin}: {r.annual_it_twh:.3f}' for r in c.itertuples()) or 'None'
            rows.append(row)
    return pd.DataFrame(rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--work', action='store_true')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    frame = summarize(args.work)
    if args.check:
        reference = pd.read_csv(ROOT / 'comparison.csv', keep_default_na=False)
        assert frame[['case', 'curve', 'top5_host_to_origin_twh_it']].equals(reference[['case', 'curve', 'top5_host_to_origin_twh_it']])
        np.testing.assert_allclose(frame.select_dtypes('number'), reference.select_dtypes('number'), rtol=1e-12, atol=1e-10)
        print('Comparison PASS; no solve')
    else:
        target = ROOT / 'work/comparison.csv'
        target.parent.mkdir(exist_ok=True)
        frame.to_csv(target, index=False)
        print(target)
