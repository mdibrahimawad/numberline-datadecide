# Auto-generated tables (do not edit; rerun `python -m tt.report`)

## Synthetic validation

| test | criterion | pass |
|---|---|---|
| 1_no_signal | |L1-0.5|<0.06, |T1 gap|<0.15, |z(T3b)|<3, |z(T3c)|<3 | yes |
| 2_shared_shift | L1>0.9, L3>0.9, |T1 gap|<0.15, z(T3b)>3, z(T3c)>3 | yes |
| 3_topic_specific_shift | L1_within>0.9, |L3-0.5|<0.1, |z(T3c)|<3, z(T3b)>3 | yes |
| 4_shape_only | |L1-0.5|<0.1, T1>0.95, T1 gap>0.2 | yes |
| 5_abrupt_change | H1 velocity peaks at the change transition, peak/median > 3 | yes |
| 6_scale_invariance | x10 cloud: normalised summaries, T1, T3 identical (rel. diff < 1e-6) | yes |

## Confound check P1

```
{
 "length_unit": "words",
 "topics": {
  "cities": {
   "n_true": 275,
   "n_false": 275,
   "length_auc": 0.5152462809917355,
   "rebalanced": false,
   "passes": true,
   "length_by_class": {
    "0": {
     "mean": 7.385454545454546,
     "hist": {
      "7": "207",
      "8": "46",
      "9": "14",
      "10": "5",
      "12": "2",
      "14": "1"
     }
    },
    "1": {
     "mean": 7.327272727272727,
     "hist": {
      "7": "220",
      "8": "33",
      "9": "14",
      "10": "5",
      "11": "1",
      "12": "2"
     }
    }
   },
   "countries": 51,
   "countries_one_class_only": 0
  },
  "sp_en_trans": {
   "n_true": 177,
   "n_false": 177,
   "length_auc": 0.5171885473522934,
   "rebalanced": false,
   "passes": true,
   "length_by_class": {
    "0": {
     "mean": 6.209039548022599,
     "hist": {
      "6": "140",
      "7": "37"
     }
    },
    "1": {
     "mean": 6.163841807909605,
     "hist": {
      "6": "148",
      "7": "29"
     }
    }
   }
  },
  "larger_than": {
   "n_true": 300,
   "n_false": 300,
   "length_auc": 0.5,
   "rebalanced": false,
   "passes": true,
   "length_by_class": {
    "0": {
     "mean": 5.0,
     "hist": {
      "5": "300"
     }
    },
    "1": {
     "mean": 5.0,
     "hist": {
      "5": "300"
     }
    }
   }
  }
 },
 "passes": true
}
```

## Confound check P2

```
{
 "length_unit": "words",
 "relations": [
  "P30",
  "P27",
  "P413",
  "P1412",
  "P103",
  "P176",
  "P495",
  "P37",
  "P17",
  "P136"
 ],
 "n": 1200,
 "length_auc": 0.5573611111111112,
 "passes": true
}
```






