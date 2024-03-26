# Roadmap

This roadmap documents action items such as features or bugs to be developed/fixed.

_Updated: 26 Mar 2024, 19:13 GMT_

## MoE Toolkit

| Status | Item                                                                                                                                                    |
| :----: | :------------------------------------------------------------------------------------------------------------------------------------------------------ |
|   ✔    | Create README and ROADMAP                                                                                                                               |
|   ✔    | Write vanilla autoregressive transformer                                                                                                                |
|   ✔    | Implement Switch Transformer                                                                                                                            |
|   ✔    | Implement expert dropout                                                                                                                                |
|   ✔    | Implement automatic mixed precision (AMP) training                                                                                                      |
|   ✔    | Implement functionality for pre vs. post layer norm                                                                                                     |
|   ✔    | Implement switch_first=True whether to begin with a Switch layer                                                                                        |
|   ✔    | Implement every_n_switch=2 for Switch layer frequency                                                                                                   |
|   ✔    | Implement rotary embeddings (RoPE) (+ scaling trick to extend context length)                                                                           |
|   ✔    | Implement multi-query attention (MQA) / grouped query attention (GQA)                                                                                   |
|   ✔    | Implement activation function customizability                                                                                                           |
|   ✔    | Write official benchmarking (time/memory) experiments for training and inference of switch vs. vanilla transformers and their various hyperparameters   |
|   ✔    | Implement RMSNorm                                                                                                                                       |
|   ✔    | Implement smaller Switch weight initialization                                                                                                          |
|   ❌   | Fix slowdown when using softmax-off-by-one                                                                                                              |
|   ❌   | Train at length after HPO, monitor via wandb                                                                                                            |
|   ❌   | Implement other MoE paradigms such as the original sparse MoE (SMoE) (Shazeer et al. 2017), PR-MoE (one fixed MLP + more experts in later layers), etc. |
|   ❌   | Implement MoE on attention (QKV) layers or heads (e.g. MoA (Zhang et al.) or SwitchHead (Csordás et al.))                                               |
|   ❌   | Implement non-autoregressive transformers and compare with methods such as expert-choice routing and Soft MoE                                           |
|   ❌   | Implement KV-cache for decoding                                                                                                                         |
