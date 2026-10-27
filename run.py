import argparse
import json
from pathlib import Path
import numpy as np
from model import HourlyConfig, solve_one
from preprocess import ROOT, load_input


def native(value):
    if isinstance(value, dict):
        return {str(k): native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', default='local_static__S0_w0.00')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--list', action='store_true')
    args = parser.parse_args()
    cases = json.loads((ROOT / 'data/cases.json').read_text())['cases']
    if args.list:
        print('\n'.join(cases))
        return
    selected = cases if args.all else {args.case: cases[args.case]}
    output = ROOT / 'work/model_results'
    output.mkdir(parents=True, exist_ok=True)
    for key, spec in selected.items():
        target = output / (key + '.json')
        if target.exists():
            raise FileExistsError(target)
        data = load_input(spec['input'])
        kwargs = dict(spec['config_kwargs'])
        if spec.get('waterstress_mode'):
            high = [0.70, 0.85, 0.95] if spec['waterstress_mode'] == 'moderate' else [0.50, 0.70, 0.85]
            factors = dict(zip(['Extremely High (>80%)', 'High (40-80%)', 'Medium - High (20-40%)'], high))
            kwargs['province_share_cap_multipliers'] = tuple(factors.get(x, 1.0) for x in data.socio_water_stress_category)
        for limit in spec['time_limit_schedule']:
            config = HourlyConfig(**dict(kwargs, time_limit_s=limit))
            ok, result = solve_one(data, scenario=spec['scenario'], omega=spec['omega'], config=config)
            if ok or result.get('solve_status') != 'time_limit':
                break
        target.write_text(json.dumps(native(result), indent=2), encoding='utf-8')
        print(key, result.get('solve_status'), result.get('objective'))


if __name__ == '__main__':
    main()
