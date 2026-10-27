from functools import lru_cache
from pathlib import Path
import hashlib
import json
import pandas as pd
import pyarrow.feather as feather

ROOT = Path(__file__).resolve().parent


def check_data():
    manifest = json.loads((ROOT / 'VERSION.json').read_text())
    for name, expected in manifest['data_sha256'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name


@lru_cache(maxsize=None)
def bundle(name):
    return pd.read_feather(ROOT / 'data' / name)


def load_integrated_table(source_csv, **kwargs):
    packed = bundle('current_results_legacy_plot_schema.feather')
    rows = packed.loc[packed.source_csv.eq(source_csv)]
    if rows.empty:
        raise KeyError(source_csv)
    prefix = rows.source_prefix.iloc[0]
    dtypes = json.loads(rows.source_dtypes_json.iloc[0])
    columns = [c for c in rows if c.startswith(prefix)]
    result = rows[columns].copy()
    result.columns = [c[len(prefix):] for c in columns]
    for col, dtype in dtypes.items():
        if dtype.startswith(('int', 'float')):
            result[col] = pd.to_numeric(result[col], errors='raise').astype(dtype)
        elif dtype == 'bool':
            result[col] = result[col].astype(bool)
        elif dtype == 'object':
            result[col] = result[col].where(result[col].notna(), None).astype(object)
    return result.reset_index(drop=True)


def table(name):
    packed = feather.read_table(ROOT / 'data/rf_data.feather')
    metadata = json.loads(packed.schema.metadata[b'two_checks_manifest'])
    spec = metadata['tables'][name]
    data = packed.to_pandas()
    result = data.loc[data._table.eq(name)].sort_values('_row')[spec['columns']].reset_index(drop=True).copy()
    assert len(result) == spec['rows']
    for col, dtype in spec['dtypes'].items():
        result[col] = result[col].astype(dtype)
    return result
