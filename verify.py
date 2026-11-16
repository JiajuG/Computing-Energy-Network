import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from plot_utils import ROOT, atlas_table, bundle, check_data, great_circle_route_pairs, table


def check_atlas():
    with np.load(ROOT / 'data/timeseries.npz', allow_pickle=False) as series:
        provinces = series['provinces'].tolist()
        days = series['month_days']
        distances = pd.DataFrame(series['dist_km'], index=provinces, columns=provinces)
    assert len(set(provinces)) == 31 and days.shape == (12,) and days.sum() == 365

    pairs = great_circle_route_pairs()
    assert len(pairs) == 930 and set(pairs.source_en) == set(pairs.target_en) == set(provinces)
    flows = pairs.flow_twh_it.to_numpy()
    roles = atlas_table('f2_province_roles', ['province_en', 'local_demand_twh_it',
                        'local_processed_twh_it', 'executed_supply_twh_it',
                        'external_served_twh_it', 'outsourced_demand_twh_it']).set_index('province_en')
    assert roles.index.is_unique and set(roles.index) == set(provinces)
    for name, direction in [('external_served_twh_it', 'source_en'),
                            ('outsourced_demand_twh_it', 'target_en')]:
        np.testing.assert_allclose(pairs.groupby(direction).flow_twh_it.sum().loc[roles.index],
                                   roles[name], atol=1e-8, rtol=0, err_msg=name)
    for total, remote in [('executed_supply_twh_it', 'external_served_twh_it'),
                          ('local_demand_twh_it', 'outsourced_demand_twh_it')]:
        np.testing.assert_allclose(roles[total], roles.local_processed_twh_it + roles[remote],
                                   atol=1e-8, rtol=0, err_msg=total)

    status = atlas_table('f4_target_frontier', ['family', 'omega', 'solve_status'])
    costs = ['cost_electricity', 'cost_routing_pure', 'cost_storage', 'cost_water']
    data = bundle('nature_story_atlas.feather')
    fields = ['family', 'omega', 'objective_cny', 'carbon_tco2', 'total_transfer_it_mwh', *costs]
    ok = data.loc[data.atlas_table.eq('f4_target_frontier') & data.solve_status.eq('optimal'), fields]
    assert not ok.empty and np.isfinite(ok[['objective_cny', 'carbon_tco2',
                                           'total_transfer_it_mwh', *costs]]).all().all()
    residual = (ok.objective_cny - ok[costs].sum(axis=1)).abs().max()
    assert residual < 0.01, f'Cost closure: {residual} CNY'
    ref = pd.read_csv(ROOT / 'data/reference_summary.csv')
    keys = ['family', 'omega']
    ref = ref.loc[ref.family.isin(status.family)].copy()
    status['omega'] = status.omega.round(8)
    ref['omega'] = ref.omega.round(8)
    ok = ok.assign(omega=ok.omega.round(8))
    status = status.set_index(keys).sort_index()
    ref = ref.set_index(keys).sort_index()
    assert status.index.is_unique and status.index.equals(ref.index)
    pd.testing.assert_series_equal(status.solve_status, ref.solve_status)
    ok = ok.set_index(keys)
    for field, tol in [('objective_cny', 0.01), ('carbon_tco2', 1e-6),
                       ('total_transfer_it_mwh', 1e-6)]:
        np.testing.assert_allclose(ok[field], ref.loc[ok.index, field], atol=tol, rtol=0, err_msg=field)

    st80 = ok.loc[('spatial_temporal', 0.8)]
    src = [provinces.index(p) for p in pairs.source_en]
    dst = [provinces.index(p) for p in pairs.target_en]
    fee = ref.loc[('spatial_temporal', 0.8), 'alpha_route_cny_per_mwh_km']
    route_cost = float(flows @ distances.to_numpy()[src, dst] * 1e6 * fee)
    np.testing.assert_allclose(route_cost, st80.cost_routing_pure, atol=0.01, rtol=0)
    np.testing.assert_allclose(flows.sum(), st80.total_transfer_it_mwh / 1e6, atol=1e-8, rtol=0)

    hourly = atlas_table('f3_national_hourly_shift_periods', ['period', 'hour',
                        'demand_static_gwh_h', 'demand_shifted_gwh_h',
                        'clean_power_used_gwh_h', 'routed_compute_gwh_it_h'])
    assert set(hourly.period) == {'Annual mean', 'January', 'July'}
    assert all(sorted(g.hour) == list(range(24)) for _, g in hourly.groupby('period'))
    return dict(provinces=len(provinces), representative_days=int(days.sum()),
                main_cases=len(status), cost_cases=len(ok), max_cost_residual_cny=float(residual),
                st80_pure_routing_cny=route_cost, s9_pairs=len(pairs),
                s9_active=int(pairs.active.sum()), s9_routed_twh_it=float(flows.sum()),
                s9_share_over_2000_km=float(pairs.loc[pairs.distance_km.gt(2000), 'flow_twh_it'].sum() / flows.sum()))


def main():
    check_data()
    checks = check_atlas()
    pairs = table('rf_road_pairs')
    assert len(pairs) == 930 and int(pairs.active.sum()) == 133
    predictions = table('rf_predictions')
    metrics = table('rf_metrics')
    for row in metrics.to_dict('records'):
        subset = predictions.loc[predictions.validation.eq(row['validation'])]
        score_col = next(c for c in ['probability', 'prediction', 'pred', 'y_prob', 'y_score'] if c in subset)
        target_col = next(c for c in ['active', 'y_true', 'y'] if c in subset)
        y, p = subset[target_col], subset[score_col]
        actual = [roc_auc_score(y, p), average_precision_score(y, p), brier_score_loss(y, p)]
        np.testing.assert_allclose(actual, [row['roc_auc'], row['average_precision'], row['brier_score']], atol=1e-12, rtol=0)
    print(json.dumps({'data': 'PASS', 'atlas': checks, 'rf': 'PASS', 'training': False, 'solving': False},
                     indent=2))


if __name__ == '__main__':
    main()
