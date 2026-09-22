# configs/generated/

Generated scenario variants (composition sweeps, family sweeps). Nothing here is
tracked except this note: the directory is produced on demand by

```
python scripts/generate_composition_sweep.py --scenario configs/benchmarks/main_abilene.yaml --family-sweep --output configs/generated/family_composition
```

The script creates its parents, so a fresh checkout can run the documented command
without this directory existing first; it is kept so the documented output path is
a real path rather than a dangling reference.
