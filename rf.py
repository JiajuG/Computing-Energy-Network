import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from plot_utils import ROOT, table

FEATURES = ['distance_km', 'same_region', 'source_access_ratio', 'source_access_clean_twh',
            'source_hydro_twh', 'source_wind_solar_twh', 'source_effective_pue',
            'source_carbon_intensity', 'source_host_cap_mw', 'target_demand_twh',
            'target_shortage_twh', 'target_access_ratio']


def fit():
    frame = table('rf_road_pairs')
    x = frame[FEATURES]
    y = frame.active.astype(int).to_numpy()
    rows = []
    predictions = []
    for mode in ['pair', 'source', 'target']:
        if mode == 'pair':
            folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(x, y)
        else:
            folds = GroupKFold(n_splits=5).split(x, y, frame[mode])
        pred = np.full(len(y), np.nan)
        for fold, (train, test) in enumerate(folds):
            model = RandomForestClassifier(n_estimators=300, min_samples_leaf=4,
                max_features='sqrt', class_weight='balanced', random_state=100+fold, n_jobs=1)
            model.fit(x.iloc[train], y[train])
            pred[test] = model.predict_proba(x.iloc[test])[:, 1]
        rows.append(dict(validation=mode, roc_auc=roc_auc_score(y, pred),
            average_precision=average_precision_score(y, pred), brier_score=brier_score_loss(y, pred)))
        predictions.append(pd.DataFrame(dict(validation=mode, row=np.arange(len(y)), prediction=pred)))
    out = ROOT / 'work'
    out.mkdir(exist_ok=True)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out/'rf_metrics.csv', index=False)
    pd.concat(predictions, ignore_index=True).to_csv(out/'rf_predictions.csv', index=False)
    return metrics


if __name__ == '__main__':
    print(fit().to_string(index=False))
