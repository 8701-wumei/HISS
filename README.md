# HISS

Source code for **HISS**, a subset selection framework for efficient robust post-training. HISS combines proxy-based risk estimation with history-aware similarity scoring to prioritize high-risk and non-redundant samples at lower computational cost.

**Note:** This code is based on ET-BERT and TrafficFormer. Many thanks to the authors of these codebases.

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

The dataset files should follow the TSV format used by ET-BERT/UER-style fine-tuning.

For the ISCX-VPN-Service dataset, the expected paths are:

```text
datasets/ISCX-VPN_Service_dataset/packet/result/train_dataset.tsv
datasets/ISCX-VPN_Service_dataset/packet/result/valid_dataset.tsv
datasets/ISCX-VPN_Service_dataset/packet/result/test_dataset.tsv
```

Each file should contain at least the following columns:

```text
label    text_a
```

Optional columns such as `text_b` and `logits` are supported by the code when using sentence-pair classification or soft targets.

## Pre-trained Models

For ET-BERT experiments, place the pre-trained model and vocabulary files under:

```text
models/pre-trained_model.bin
models/encryptd_vocab.txt
```

For TrafficFormer experiments, use the corresponding TrafficFormer pre-trained checkpoint and configuration files.

The default TrafficFormer paths in the code are:

```text
models/nomoe_bertflow_pre-trained_model.bin-120000
models/encryptd_vocab.txt
models/bert/base_config.json
```

## Reproducibility Commands

The following commands reproduce ET-BERT experiments on the ISCX-VPN-Service dataset.

Common setting:

- Backbone: ET-BERT
- Dataset: ISCX-VPN-Service
- Batch size: 32
- Sequence length: 128
- Backbone learning rate: `2e-5`
- Candidate pool size for subset-selection methods: 64
- Selected batch size for subset-selection methods: 32
- Candidate pool size is obtained by using `--batch_size 32 --CVaR_alpha 0.5`
- Main results should be repeated over multiple random seeds, for example `--seed 1`, `--seed 2`, ..., `--seed 5`

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
  --CVaR_alpha 0.5 \
  --rm_lr 1e-4 \
  --loss_hidden_size 256 \
  --warmup_erm 2 \
  --rcc_lambda_base 0.1 \
  --rcc_memory_capacity 96 \
  --rcc_rbf softmax \
  --rcc_tau 0.3 \
  --rcc_min_sigma 1e-6 \
  --proxy_steps_high 1 \
  --proxy_steps_low 2 \
  --proxy_spearman_low 0.3 \
  --proxy_spearman_window 20 \
  --proxy_schedule_warmup 10 \
  --min_delta 1e-6 \
  --temperature 1.0 \
  --seed 1
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
  --CVaR_alpha 0.5 \
  --rm_lr 1e-4 \
  --loss_hidden_size 256 \
  --warmup_erm 2 \
  --proxy_steps_high 1 \
  --proxy_steps_low 2 \
  --proxy_spearman_low 0.3 \
  --proxy_spearman_window 20 \
  --proxy_schedule_warmup 10 \
  --min_delta 1e-6 \
  --temperature 1.0 \
  --seed 1
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
  --seed 1
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
  --seed 1
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
  --seed 1
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
  --seed 1
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
  --gdro_tau 1.0 \
  --seed 1
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
  --tdro_rho 1e-3 \
  --tdro_lambda 1.0 \
  --seed 1
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
  --seed 1
```

## Method-specific Parameters

### HISS

The key method-specific parameters for HISS are:

```text
--rm_lr 1e-4
--loss_hidden_size 256
--warmup_erm 2
--rcc_lambda_base 0.1
--rcc_memory_capacity 96
--rcc_rbf softmax
--rcc_tau 0.3
--rcc_min_sigma 1e-6
--min_delta 1e-6
--temperature 1.0
```

For datasets other than ISCX-VPN-Service, use the dataset-specific value of `--rcc_lambda_base`.

For example, if the final configuration uses `lambda = 0.6` on another dataset, replace:

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
--temperature 1.0
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
--gdro_tau 1.0
```

### TDRO

The key method-specific parameters are:

```text
--tdro_rho 1e-3
--tdro_lambda 1.0
```

### Focal Loss

The key method-specific parameters are:

```text
--focal_gamma 2.0
--focal_alpha 1.0
```

## Important Notes

Do not pass `--weighted False` on the command line.

In the current argument parser, `--weighted` is parsed with `type=bool`, so passing the string `"False"` may still be interpreted unexpectedly. Omit `--weighted` to keep the default unweighted pairwise ranking loss.

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

## Citation

If you use this code, please cite the corresponding paper.

```bibtex
@inproceedings{hiss,
  title     = {HISS: History-Informed Subset Selection for Efficient Robust Post-training of Encrypted Traffic Foundation Models},
  author    = {Anonymous},
  booktitle = {Anonymous Submission},
  year      = {2026}
}
```
