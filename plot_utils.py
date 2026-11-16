from functools import lru_cache
from pathlib import Path
import hashlib
import json
import numpy as np
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


def atlas_table(name, columns):
    data = bundle('nature_story_atlas.feather')
    rows = data.loc[data.atlas_table.eq(name)]
    if rows.empty:
        raise ValueError(f'Missing or empty atlas table: {name}')
    missing = set(columns) - set(rows.columns)
    if missing:
        raise ValueError(f'{name}: missing columns {sorted(missing)}')
    rows = rows[columns].copy()
    if rows.isna().any().any() or not np.isfinite(rows.select_dtypes(include='number')).all().all():
        raise ValueError(f'{name}: missing or non-finite required values')
    return rows


def great_circle_route_pairs():
    coords = atlas_table('f1_province_spatial', ['province_en', 'lat', 'lon']).set_index('province_en')
    if len(coords) != 31 or not coords.index.is_unique:
        raise ValueError('Expected 31 unique province coordinates')
    lat, lon = np.deg2rad(coords[['lat', 'lon']].to_numpy()).T
    hav = (np.sin((lat[:, None] - lat) / 2)**2
           + np.cos(lat[:, None]) * np.cos(lat) * np.sin((lon[:, None] - lon) / 2)**2)
    source, target = np.where(~np.eye(len(coords), dtype=bool))
    distance = 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(hav[source, target], 0, 1)))
    keys = ['source_en', 'target_en']
    pairs = pd.DataFrame(dict(source_en=coords.index[source], target_en=coords.index[target],
                              distance_km=distance))
    routes = atlas_table('f2_annual_routes', keys + ['flow_twh_it'])
    flows = routes.set_index(keys).flow_twh_it
    pair_index = pd.MultiIndex.from_frame(pairs[keys])
    if (not flows.index.is_unique or not flows.index.isin(pair_index).all()
            or (flows < 0).any()):
        raise ValueError('Invalid annual route keys or flow values')
    if not np.isfinite(distance).all() or (distance <= 0).any():
        raise ValueError('Invalid great-circle distances')
    pairs['flow_twh_it'] = flows.reindex(pair_index, fill_value=0).to_numpy()
    pairs['active'] = pairs.flow_twh_it > 1e-6
    if not pairs.active.any():
        raise ValueError('No active annual routes')
    return pairs


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
