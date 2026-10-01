# DR04 Evaluator Split Build

Metadata-only deterministic construction; no image bytes were read.

## Split counts

- EVAL_TRAIN: 94200
- EVAL_VAL: 5000
- REAL_GENERATOR_OOD_CEILING: 400
- REAL_ID_CEILING: 400

- Manifest SHA-256: `09fa2cc89caf3f9b2e72d97b3a4cda16a2c7d728ef895c9111fc2c1caa558ee9`
- Config SHA-256: `6585ba88d949e4f929a62057a4f3695ed0f381e48e1f303177922965857e8197`

## Hash canonicalization

`seed=<seed>|namespace=<namespace>|image_id=<canonical_id>|<sorted key=value fields>`

Namespaces: dr04-evaluator/ood-ceiling, dr04-evaluator/control-quota-remainder, dr04-evaluator/control-candidate-assignment, dr04-evaluator/eval-val-exclusive-ood, dr04-evaluator/eval-val-natural-remainder
