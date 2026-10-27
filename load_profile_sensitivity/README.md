# Load profiles

LT45: local temporal 45%. ST80: spatiotemporal 80%.
Daily provincial IT totals and original capacity bounds remain fixed.
Uses the parent package's model, inputs and dependencies.
`results.feather` contains accepted outputs; `comparison.csv` is the reference table.
Corridors run from host to demand origin. Inner Mongolia belongs to North China.

```bash
python run.py --check
python compare.py --check
```

Optional recomputation:

```bash
python run.py --all
python compare.py --work
```

New outputs stay in `work`; existing results are not overwritten.
