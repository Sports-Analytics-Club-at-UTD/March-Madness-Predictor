# Vilnius experiment results

All scores are walk-forward Brier; lower is better. HIST = 2015–2024 excluding 2020. Historical ensemble weights are fitted on preceding HIST seasons; target weights freeze at 2024. Acceptance compares standalone/ensemble to final_model_only and market rows to final.

|Config|hist|hM|hW|v21–24|2025|M25|W25|2026|M26|W26|Accept|
|---|---|---|---|---|---|---|---|---|---|---|---|
|final_model_only|0.16430|0.18810|0.14056|0.17519|0.11715|0.13424|0.10006|0.12457|0.14844|0.10069|baseline|
|final|0.16430|0.18810|0.14056|0.17519|0.10992|0.12614|0.09370|0.12267|0.14458|0.10076|baseline|
|V|0.16432|0.18886|0.13983|0.17330|0.11144|0.12901|0.09388|0.12962|0.16062|0.09862|yes|
|V_ensemble|0.16412|0.18870|0.13960|0.17331|0.11469|0.13265|0.09672|0.12656|0.15183|0.10129|yes|
|V_market|0.16412|0.18870|0.13960|0.17331|0.10950|0.12590|0.09309|0.12282|0.14498|0.10066|yes|
|modeh|0.16476|0.18775|0.14180|0.17259|0.11312|0.13075|0.09549|0.13083|0.16155|0.10012|no|
|modeh_ensemble|0.16411|0.18869|0.13959|0.17356|0.11608|0.13270|0.09946|0.12866|0.15381|0.10350|no|
|modeh_market|0.16411|0.18869|0.13959|0.17356|0.11010|0.12597|0.09422|0.12364|0.14524|0.10204|no|
|cauchy|0.16429|0.18884|0.13979|0.17318|0.11152|0.12925|0.09380|0.12986|0.16084|0.09889|yes|
|cauchy_ensemble|0.16409|0.18870|0.13955|0.17329|0.11469|0.13273|0.09664|0.12665|0.15195|0.10135|yes|
|cauchy_market|0.16409|0.18870|0.13955|0.17329|0.10948|0.12592|0.09305|0.12284|0.14499|0.10068|yes|
|seeds|0.16427|0.18882|0.13977|0.17322|0.11133|0.12907|0.09358|0.12968|0.16075|0.09861|yes|
|seeds_ensemble|0.16411|0.18870|0.13957|0.17330|0.11460|0.13266|0.09655|0.12659|0.15194|0.10124|yes|
|seeds_market|0.16411|0.18870|0.13957|0.17330|0.10946|0.12591|0.09302|0.12281|0.14500|0.10062|yes|

## Recommendation

Use the base recipe with history-fitted L/X/V weights and unchanged final market settings. The three-seed market result is slightly better (about 0.00002 mean Brier), but costs three times as many Vilnius fits. This is too small to prefer over the simpler base model.

```json
{
  "model": "vilnius_ensemble",
  "variant": "V",
  "market_w": {
    "M": 0.85,
    "W": 0.5
  },
  "mkt_sharpen": 1.2
}
```

Learned weights (L, X, V), fitted only on 2015–2024:

```json
{
  "M": [
    0.0,
    0.5659461126995003,
    0.4340538873004997
  ],
  "W": [
    0.655176342262174,
    3.3458787348959997e-17,
    0.3448236577378259
  ]
}
```

The modeh variants improve v21–24 but worsen the target mean and are rejected. No tested configuration fails solely because historical degradation exceeds 0.0005. None reaches top-20 in both targets; the recommended blend slightly worsens 2026 relative to final.

## Market sensitivity

No market parameter was retuned. Moving both existing gender weights by +/-0.05 gives:

```json
{
  "-0.05": {
    "weights": {
      "M": 0.7999999999999999,
      "W": 0.45
    },
    "scores": {
      "2025": {
        "M": 0.12582274138252678,
        "W": 0.09326886618506895
      },
      "2026": {
        "M": 0.14522165311235624,
        "W": 0.10057535298738832
      }
    },
    "mean": 0.11622215341683506
  },
  "0": {
    "weights": {
      "M": 0.85,
      "W": 0.5
    },
    "scores": {
      "2025": {
        "M": 0.12590075241328563,
        "W": 0.09309419815656829
      },
      "2026": {
        "M": 0.14497503040496793,
        "W": 0.10065991300817446
      }
    },
    "mean": 0.11615747349574906
  },
  "0.05": {
    "weights": {
      "M": 0.9,
      "W": 0.55
    },
    "scores": {
      "2025": {
        "M": 0.1260181938225722,
        "W": 0.09295406953655119
      },
      "2026": {
        "M": 0.14474790386516956,
        "W": 0.10077803954864238
      }
    },
    "mean": 0.11612455169323384
  }
}
```

## Validation

Six focused unittest checks passed. The separate, unchanged final.py grid reproduced 2025=0.1099 and 2026=0.1227 (model-only 0.1171 / 0.1246). Input data was not modified.
