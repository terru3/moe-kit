# Roadmap

This roadmap documents action items such as features or bugs to be developed/fixed.

_Updated: 14 Dec 2023, 23:20 GMT_

## MoE Toolkit

| Status | Item                                                                    |
| :----: | :---------------------------------------------------------------------- |
|   ✔    | Create README and ROADMAP                                               |
|   ✔    | Write vanilla autoregressive transformer                                |
|   ✔    | Implement Switch Transformer                                            |
|   ✔    | Implement expert dropout                                                |
|   ✔    | Implement automatic mixed precision (AMP) training                      |
|   ✔    | Implement functionality for pre vs. post layer norm                     |
|   ❌   | Implement functionality for Switch layer alternating 1/3/5 vs. 2/4/6    |
|   ❌   | Implement smaller Switch weight initialization                          |
|   ❌   | Hyperparameter sweeps via wandb                                         |
|   ❌   | Implement Switch on attention (QKV) layers                              |
|   ❌   | Implement other MoE paradigms such as expert-choice routing and ST-MoE  |
|   ❌   | Train longer on TinyShakespeare, monitor performance and training time  |
|   ❌   | Implement multi-query attention (MQA) and grouped query attention (GQA) |
