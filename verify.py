import json
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from plot_utils import check_data, table


def main():
    check_data()
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
    print(json.dumps({'data':'PASS', 'rf':'PASS', 'training':False}))


if __name__ == '__main__':
    main()
