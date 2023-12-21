# Roadmap

This roadmap documents action items such as features or bugs to be developed/fixed.

_Updated: 14 Dec 2023, 23:20 GMT_

## MoE Toolkit

| Status | Item                                                                                                               |
| :----: | :----------------------------------------------------------------------------------------------------------------- |
|   ✔    | Create README and ROADMAP                                                                                          |
|   ✔    | Write vanilla autoregressive transformer                                                                           |
|   ✔    | Implement Switch Transformer                                                                                       |
|   ✔    | Implement expert dropout                                                                                           |
|   ✔    | Implement automatic mixed precision (AMP) training                                                                 |
|   ✔    | Implement functionality for pre vs. post layer norm                                                                |
|   ❌   | Implement switch_first=True whether to begin with a Switch layer                                                   |
|   ❌   | Implement every_n_switch=2 for Switch layer frequency                                                              |
|   ❌   | Implement smaller Switch weight initialization                                                                     |
|   ❌   | Implement the original sparse MoE (SMoE) (Shazeer et al. 2017)                                                     |
|   ❌   | Hyperparameter sweeps via wandb                                                                                    |
|   ❌   | Implement other MoE paradigms such as expert-choice routing, PR-MoE (one fixed MLP + more experts in later layers) |
|   ❌   | Train longer on TinyShakespeare, monitor performance and training time                                             |
|   ❌   | Consider implementing multi-query attention (MQA) and grouped query attention (GQA)                                |
|   ❌   | Implement MoE on attention (QKV) layers or heads (e.g. MoA (Zhang et al.))                                         |
