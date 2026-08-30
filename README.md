# HISS

Source code for **HISS**, a subset selection framework for efficient robust post-training. HISS combines proxy-based risk estimation with history-aware similarity scoring to prioritize high-risk and non-redundant samples at lower computational cost.

## Acknowledgements

This implementation is based on the following open-source codebases:

- [ET-BERT](https://github.com/linwhitehat/ET-BERT)
- [TrafficFormer](https://github.com/IDP-code/TrafficFormer)

We sincerely thank the authors for making their implementations publicly available.

## Repository Scope

This repository is not a standalone reimplementation of ET-BERT or TrafficFormer. Instead, it provides the modified training script used for HISS-based robust post-training. The unchanged backbone-related files are inherited from the original ET-BERT/UER-py and TrafficFormer codebases.

To reproduce the results, users should:

1. Clone the ET-BERT repository.
2. Install its required dependencies.
3. Place `run_ser_classifier.py` in the ET-BERT fine-tuning environment.
4. Download the required pretrained checkpoint and vocabulary files.
5. Run the commands provided below.

## Main-Experiment Standard Deviations

The main experiments with ET-BERT are averaged over five random seeds. The corresponding standard deviations for ISCX-VPN-Service, ISCX-VPN-App, USTC-TFC, and CSTNET-TLS 1.3 are provided in [HISS_Main_Experiment_Standard_Deviations.pdf](./HISS_Main_Experiment_Standard_Deviations.pdf).

## Requirements

The code requires the following major dependencies:

- Python 3.8+
- PyTorch
- NumPy
- SciPy
- pandas
- psutil
- UER-py / ET-BERT-compatible modules
- apex, optional, only required when using `--fp16`

Example installation:

```bash
pip install numpy scipy pandas psutil
```

Please install PyTorch according to your CUDA version.

## Datasets

All datasets used in this paper are obtained from the `datasets` directory of the ET-BERT repository:

- [ET-BERT datasets directory](https://github.com/linwhitehat/ET-BERT/tree/main/datasets)

Please download the corresponding dataset archives from the ET-BERT repository, decompress them, and organize the processed TSV files as follows.

### Expected Directory Structure

```text
datasets/
├── ISCX-VPN_Service_dataset/
│   └── packet/
│       └── result/
│           ├── train_dataset.tsv
│           ├── valid_dataset.tsv
│           └── test_dataset.tsv
├── ISCX-VPN_app_dataset/
│   └── packet/
│       └── result/
│           ├── train_dataset.tsv
│           ├── valid_dataset.tsv
│           └── test_dataset.tsv
├── USTC-TFC_dataset/
│   └── packet/
│       └── result/
│           ├── train_dataset.tsv
│           ├── valid_dataset.tsv
│           └── test_dataset.tsv
└── cstnet-tls1.3/
    └── packet/
        └── result/
            ├── train_dataset.tsv
            ├── valid_dataset.tsv
            └── test_dataset.tsv
```

## Pre-trained Models

This repository does not redistribute the original ET-BERT or TrafficFormer pre-trained checkpoints. Please download them from the official release links and place them in the expected paths.

| Backbone | Download link | Expected path |
| --- | --- | --- |
| ET-BERT | [Google Drive](https://drive.google.com/file/d/1r1yE34dU2W8zSqx1FkB8gCWri4DQWVtE/view) | `models/pre-trained_model.bin` |
| TrafficFormer | [Google Drive](https://drive.google.com/file/d/1pR6ZaWE7MWFDQWiF4LDzSyjSq0Gj3kV7/view) | `models/nomoe_bertflow_pre-trained_model.bin-120000` |

The vocabulary and configuration files should also be prepared according to the original ET-BERT and TrafficFormer repositories.
For convenience, the default paths used by `run_ser_classifier.py` are:

```text
models/encryptd_vocab.txt
models/bert/base_config.json
```

## Reproducibility Commands

The following commands reproduce ET-BERT experiments on the ISCX-VPN-Service dataset.

Before running experiments on a different dataset, please modify the `DATASET` variable in `run_ser_classifier.py`, which is used to name output folders and log files. For example, change `DATASET = "ser"` to `DATASET = "app"` for ISCX-VPN-App. The dataset paths in `--train_path`, `--dev_path`, and `--test_path` should also be updated accordingly.

Common setting:

- Backbone: ET-BERT
- Dataset: ISCX-VPN-Service
- Batch size: 32
- Sequence length: 128
- Backbone learning rate: `2e-5`
- Candidate pool size for subset-selection methods: 64
- Selected batch size for subset-selection methods: 32
- Candidate pool size is obtained by using `--batch_size 32 --CVaR_alpha 0.5`
- Main experimental results are averaged over five random seeds; the corresponding standard deviations are provided in [`HISS_Main_Experiment_Standard_Deviations.pdf`](./HISS_Main_Experiment_Standard_Deviations.pdf)

### HISS

```bash
python run_ser_classifier.py \
  --pretrained_model_path models/pre-trained_model.bin \
  --vocab_path models/encryptd_vocab.txt \
  --train_path datasets/ISCX-VPN_Service_dataset/packet/result/train_dataset.tsv \
  --dev_path datasets/ISCX-VPN_Service_dataset/packet/result/valid_dataset.tsv \
  --test_path datasets/ISCX-VPN_Service_dataset/packet/result/test_dataset.tsv \
  --epochs_num 20 \
  --batch_size 32 \
  --embedding word_pos_seg \
  --encoder transformer \
  --mask fully_visible \
  --seq_length 128 \
  --learning_rate 2e-5 \
  --method HISS \
  --rm_lr 1e-4 \
  --CVaR_alpha 0.5 \
  --hiss_lambda_base 0.1 \
  --hiss_memory_capacity 320 \
  --hiss_rbf softmax \
  --hiss_tau 0.3 \
  --seed 9
```

### HISS without History Penalty

This variant removes the history-aware redundancy penalty and keeps only proxy-based risk selection.

```bash
python run_ser_classifier.py \
  --pretrained_model_path models/pre-trained_model.bin \
  --vocab_path models/encryptd_vocab.txt \
  --train_path datasets/ISCX-VPN_Service_dataset/packet/result/train_dataset.tsv \
  --dev_path datasets/ISCX-VPN_Service_dataset/packet/result/valid_dataset.tsv \
  --test_path datasets/ISCX-VPN_Service_dataset/packet/result/test_dataset.tsv \
  --epochs_num 20 \
  --batch_size 32 \
  --embedding word_pos_seg \
  --encoder transformer \
  --mask fully_visible \
  --seq_length 128 \
  --learning_rate 2e-5 \
  --method HISS-no-penalty \
  --rm_lr 1e-4 \
  --CVaR_alpha 0.5 \
  --seed 9
```

### ERM

```bash
python run_ser_classifier.py \
  --pretrained_model_path models/pre-trained_model.bin \
  --vocab_path models/encryptd_vocab.txt \
  --train_path datasets/ISCX-VPN_Service_dataset/packet/result/train_dataset.tsv \
  --dev_path datasets/ISCX-VPN_Service_dataset/packet/result/valid_dataset.tsv \
  --test_path datasets/ISCX-VPN_Service_dataset/packet/result/test_dataset.tsv \
  --epochs_num 10 \
  --batch_size 32 \
  --embedding word_pos_seg \
  --encoder transformer \
  --mask fully_visible \
  --seq_length 128 \
  --learning_rate 2e-5 \
  --method ERM \
  --seed 9
```

### MC-CVaR

```bash
python run_ser_classifier.py \
  --pretrained_model_path models/pre-trained_model.bin \
  --vocab_path models/encryptd_vocab.txt \
  --train_path datasets/ISCX-VPN_Service_dataset/packet/result/train_dataset.tsv \
  --dev_path datasets/ISCX-VPN_Service_dataset/packet/result/valid_dataset.tsv \
  --test_path datasets/ISCX-VPN_Service_dataset/packet/result/test_dataset.tsv \
  --epochs_num 20 \
  --batch_size 32 \
  --embedding word_pos_seg \
  --encoder transformer \
  --mask fully_visible \
  --seq_length 128 \
  --learning_rate 2e-5 \
  --method MC-CVaR \
  --CVaR_alpha 0.5 \
  --seed 9
```

### Random Selection

```bash
python run_ser_classifier.py \
  --pretrained_model_path models/pre-trained_model.bin \
  --vocab_path models/encryptd_vocab.txt \
  --train_path datasets/ISCX-VPN_Service_dataset/packet/result/train_dataset.tsv \
  --dev_path datasets/ISCX-VPN_Service_dataset/packet/result/valid_dataset.tsv \
  --test_path datasets/ISCX-VPN_Service_dataset/packet/result/test_dataset.tsv \
  --epochs_num 20 \
  --batch_size 32 \
  --embedding word_pos_seg \
  --encoder transformer \
  --mask fully_visible \
  --seq_length 128 \
  --learning_rate 2e-5 \
  --method Random \
  --CVaR_alpha 0.5 \
  --seed 9
```

### OHTM

```bash
python run_ser_classifier.py \
  --pretrained_model_path models/pre-trained_model.bin \
  --vocab_path models/encryptd_vocab.txt \
  --train_path datasets/ISCX-VPN_Service_dataset/packet/result/train_dataset.tsv \
  --dev_path datasets/ISCX-VPN_Service_dataset/packet/result/valid_dataset.tsv \
  --test_path datasets/ISCX-VPN_Service_dataset/packet/result/test_dataset.tsv \
  --epochs_num 20 \
  --batch_size 32 \
  --embedding word_pos_seg \
  --encoder transformer \
  --mask fully_visible \
  --seq_length 128 \
  --learning_rate 2e-5 \
  --method OHTM \
  --CVaR_alpha 0.5 \
  --seed 9
```

### GroupDRO

```bash
python run_ser_classifier.py \
  --pretrained_model_path models/pre-trained_model.bin \
  --vocab_path models/encryptd_vocab.txt \
  --train_path datasets/ISCX-VPN_Service_dataset/packet/result/train_dataset.tsv \
  --dev_path datasets/ISCX-VPN_Service_dataset/packet/result/valid_dataset.tsv \
  --test_path datasets/ISCX-VPN_Service_dataset/packet/result/test_dataset.tsv \
  --epochs_num 10 \
  --batch_size 32 \
  --embedding word_pos_seg \
  --encoder transformer \
  --mask fully_visible \
  --seq_length 128 \
  --learning_rate 2e-5 \
  --method GroupDRO \
  --gdro_tau 0.1 \
  --seed 9
```

### TDRO

```bash
python run_ser_classifier.py \
  --pretrained_model_path models/pre-trained_model.bin \
  --vocab_path models/encryptd_vocab.txt \
  --train_path datasets/ISCX-VPN_Service_dataset/packet/result/train_dataset.tsv \
  --dev_path datasets/ISCX-VPN_Service_dataset/packet/result/valid_dataset.tsv \
  --test_path datasets/ISCX-VPN_Service_dataset/packet/result/test_dataset.tsv \
  --epochs_num 10 \
  --batch_size 32 \
  --embedding word_pos_seg \
  --encoder transformer \
  --mask fully_visible \
  --seq_length 128 \
  --learning_rate 2e-5 \
  --method TDRO \
  --tdro_rho 0.05 \
  --tdro_lambda 0.05 \
  --seed 9
```

### Focal Loss

```bash
python run_ser_classifier.py \
  --pretrained_model_path models/pre-trained_model.bin \
  --vocab_path models/encryptd_vocab.txt \
  --train_path datasets/ISCX-VPN_Service_dataset/packet/result/train_dataset.tsv \
  --dev_path datasets/ISCX-VPN_Service_dataset/packet/result/valid_dataset.tsv \
  --test_path datasets/ISCX-VPN_Service_dataset/packet/result/test_dataset.tsv \
  --epochs_num 10 \
  --batch_size 32 \
  --embedding word_pos_seg \
  --encoder transformer \
  --mask fully_visible \
  --seq_length 128 \
  --learning_rate 2e-5 \
  --method Focal \
  --focal_gamma 2.0 \
  --focal_alpha 1.0 \
  --seed 9
```

## Method-specific Parameters

### HISS

The key method-specific parameters for HISS are:

```text
--rm_lr 1e-4
--loss_hidden_size 256
--warmup_erm 2
--rcc_lambda_base 0.1
--rcc_memory_capacity 320
--rcc_rbf softmax
--rcc_tau 0.3
--rcc_min_sigma 1e-6
--min_delta 1e-6
```

For datasets with more severe tail risks than ISCX-VPN-Service, replace:

```bash
--rcc_lambda_base 0.1
```

with:

```bash
--rcc_lambda_base 0.6
```

### HISS-no-penalty

The key method-specific parameters are:

```text
--rm_lr 1e-4
--loss_hidden_size 256
--warmup_erm 2
--min_delta 1e-6
```

This method does not use the history-aware penalty. Therefore, `--rcc_lambda_base`, `--rcc_memory_capacity`, `--rcc_rbf`, and `--rcc_tau` are not required.

### MC-CVaR, Random Selection, and OHTM

These candidate-screening methods use:

```text
--CVaR_alpha 0.5
```

With `--batch_size 32`, this gives a candidate pool size of 64 and selects 32 samples for update.

### GroupDRO

The key method-specific parameter is:

```text
--gdro_tau 0.1
```

### TDRO

The key method-specific parameters are:

```text
--tdro_rho 0.05
--tdro_lambda 0.05
```

### Focal Loss

The key method-specific parameters are:

```text
--focal_gamma 2.0
--focal_alpha 1.0
```

## Important Notes

For HISS, MC-CVaR, Random Selection, and OHTM, we use 20 training epochs because only a subset of candidates is used for each update. For ERM, GroupDRO, TDRO, and Focal Loss, we use 10 training epochs.

## Outputs

The code automatically creates result folders under:

```text
ET-BERT_Results/
```

or

```text
TrafficFormer_Results/
```

depending on the backbone checkpoint.

The following files are saved after evaluation:

```text
acc.txt
confusion.txt
CVaR.txt
acc_list.txt
prec_list.txt
rec_list.txt
f1_list.txt
rem.txt
```

Training logs are saved under each result folder:

```text
logs/
models/
```

Efficiency logs are saved under:

```text
ET-BERT_efficiency_results_ser/
```

or

```text
TrafficFormer_efficiency_results_ser/
```
