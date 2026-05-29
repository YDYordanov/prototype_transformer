# Prototype Attention

This repository contains the code for **Prototype Transformer: Towards Language Model
Architectures Interpretable by Design**, accepted at ICML 2026.

If you use this code, please cite:

```bibtex
@inproceedings{yordanov2026prototype,
  title = {Prototype Transformer: Towards Language Model Architectures Interpretable by Design},
  author = {Yordanov, Yordan and Forasassi, Matteo and Menzat, Bayar and Wang, Ruizhi and Qi, Chang and Kaltenberger, Markus and M'Charrak, Amine and Salvatori, Tommaso and Lukasiewicz, Thomas},
  booktitle = {International Conference on Machine Learning (ICML)},
  year = {2026},
  url = {https://arxiv.org/abs/2602.11852}
}
```

This repository contains code from 7 collaborators, which explains the structure and organization of the code. We have made efforts to organize it, but there may still be some issues. If you have any questions about the code, please feel free to reach out to us.


## Installation

Basic requirements:

E.g. inside a new conda env install:

Python (<= 3.11 for Deltanet speedups)

and then:

```bash
pip install torch torchtune torchao transformers datasets==3.6.0 sentencepiece protobuf matplotlib seaborn
```

For speed optimisations for the DeltaNet baseline (for Python <= 3.11):

```bash
pip install flash-linear-attention causal_conv1d
```

For speed optimisations for Mamba-1 (requires a PyTorch version compatible with `mamba_ssm`):

```bash
pip install mamba_ssm
```

For speed optimisations for LLaMA 3.1:

```bash
pip install flash-attn triton
```

For EasyTuna hyperparameter search:

Install EasyTuna from [YDYordanov/EasyTuna](https://github.com/YDYordanov/EasyTuna)
and the Optuna BoTorch integration:

```bash
pip install git+https://github.com/YDYordanov/EasyTuna.git
pip install optuna "optuna-integration[botorch]"
```

## Usage

Data preparation instructions for the FineWeb-Edu dataset are in
[`fineweb_data_prep/`](fineweb_data_prep/).

To run the main hyperparameter search across all models:

```bash
python hyperparameter_search.py
```

### Experiment Scripts

The scalability and ablation studies, together with the table construction
scripts, are in the `experiments/` folder:

- `experiments/ablation_studies`
- `experiments/scalability`

These experiments use EasyTuna for experiment management. EasyTuna snapshots
the training code for each study, can resume studies with
`resume_if_exists=True`, and writes logs/results under:

```text
<log_dir>/<exper_id>/<study_id>/
```

Use `monitor_experiments.py` from the EasyTuna repo to monitor or terminate
running searches:

```bash
python monitor_experiments.py path/to/experiments
```

The scalability experiments require test-evaluation to be run separately after training, which can be done for each sub-experiment (e.g. scalability_models_ctx) as so:

```bash
python test_eval_experiments.py --experiment_dir=logs/scalability_models_ctx
```

### Further Experiments

The downstream performance, text-generation, interpretability, intervention, and robustness experiments will be found in the subfolders of the `experiments/` folder, where each will have self-contained code and instructions.

Note: some experiment scripts import top-level modules (e.g. from `prototype_attention`), so for those make sure to run them from the top-level directory e.g. as follows:

```bash
PYTHONPATH="$PWD" python experiments/interpretability/find_proto_activations.py
```
