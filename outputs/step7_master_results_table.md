| model | full_label_illicit_f1 | label_eff_f1_at_0.1frac | pre_shutdown_illicit_f1 | post_shutdown_illicit_f1 |
|---|---|---|---|---|
| Baseline | 0.6008 | 0.4510 | 0.6807 | 0.0260 |
| SSL: fine-tuned | 0.5924 | 0.4212 | 0.6971 | 0.0496 |
| SSL + A2 (Domain Adv) | 0.5836 | 0.4320 | 0.6819 | 0.0420 |
| SSL + B1 (Distillation) | 0.6095 | 0.4763 | 0.6978 | 0.0233 |
| SSL + A2 + B1 | 0.6049 | 0.4754 | 0.6897 | 0.0237 |