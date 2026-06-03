"""
This script provides an exmaple to wrap UER-py for classification.
"""
import random
import argparse
import math
import torch
import torch.nn as nn
from uer.layers import *
from uer.encoders import *
from uer.utils.vocab import Vocab
from uer.utils.constants import *
from uer.utils import *
from uer.utils.optimizers import *
from uer.utils.config import load_hyperparam
from uer.utils.seed import set_seed
from uer.model_saver import save_model
from uer.opts import finetune_opts
from torch import optim
import time
import os
import torch.nn.functional as F
import numpy as np
from datetime import timedelta # Import timedelta for remaining-time estimation.
import psutil
import json
from datetime import datetime
from typing import Dict, List, Optional
from collections import deque
import pandas as pd
from scipy.stats import spearmanr, pearsonr
import warnings

os.environ["CUDA_VISIBLE_DEVICES"] = "5,4,3,1"

TRAFFICFORMER_PRETRAINED_MODEL_PATH = "models/nomoe_bertflow_pre-trained_model.bin-120000"
TRAFFICFORMER_VOCAB_PATH = "models/encryptd_vocab.txt"
TRAFFICFORMER_CONFIG_PATH = "models/bert/base_config.json"
TRAFFICFORMER_PRETRAINED_MODEL_URL = "https://drive.google.com/file/d/1pR6ZaWE7MWFDQWiF4LDzSyjSq0Gj3kV7/view?usp=sharing"

DATASET = "ser"


def capture_rng_state():
    state = {
        "torch_cpu": torch.get_rng_state()
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    else:
        state["torch_cuda"] = None
    return state


def restore_rng_state(state):
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state["torch_cuda"] is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


class TrainingEfficiencyMonitor:
    """
    Training Efficiency Monitor - Used to compare the performance differences between different training methods.
    Only collects key efficiency metrics and saves the results.
    """

    def __init__(self,
                 experiment_name: str,
                 method_name: str,
                 output_dir: str,
                 monitor_interval: float = 1.0):

        self.experiment_name = experiment_name
        self.method_name = method_name
        self.output_dir = output_dir
        self.monitor_interval = monitor_interval

        os.makedirs(output_dir, exist_ok=True)

        self.metrics_history = []
        self.start_time = None
        self.process = psutil.Process()

        print(f"🔍 Start monitoring: {experiment_name} - {method_name}")

    def start(self):

        self.start_time = datetime.now()
        initial_metrics = self._collect_metrics("initial")
        self.metrics_history.append(initial_metrics)

        print(f"⏱️  Start time: {self.start_time.strftime('%H:%M:%S')}")
        return self

    def _collect_metrics(self, phase: str) -> Dict:

        try:
            with self.process.oneshot():
                mem_info = self.process.memory_info()

                return {
                    'timestamp': datetime.now().isoformat(),
                    'phase': phase,
                    'runtime_seconds': (datetime.now() - self.start_time).total_seconds(),

                    'memory_mb': mem_info.rss / 1024 / 1024,
                    'cpu_percent': self.process.cpu_percent(interval=None),
                    'threads': self.process.num_threads(),

                    'io_read_mb': self.process.io_counters().read_bytes / 1024 / 1024 if hasattr(self.process, 'io_counters') else 0,
                }
        except:
            return None

    def record_checkpoint(self, checkpoint_name: str):
        metrics = self._collect_metrics(f"checkpoint_{checkpoint_name}")
        if metrics:
            self.metrics_history.append(metrics)
        return metrics

    def stop(self, final_stats: Dict = None) -> Dict:
        if not self.start_time:
            return {}

        final_metrics = self._collect_metrics("final")
        if final_metrics:
            self.metrics_history.append(final_metrics)

        training_metrics = [
            m for m in self.metrics_history
            if m and m.get('phase', '').startswith('checkpoint') or m.get('phase') == 'training'
        ]

        efficiency_summary = self._calculate_efficiency_summary(training_metrics)

        if final_stats:
            efficiency_summary.update(final_stats)

        self._save_results(efficiency_summary)

        return efficiency_summary

    def _calculate_efficiency_summary(self, training_metrics: List[Dict]) -> Dict:
        if not training_metrics:
            return {}

        memory_values = [m['memory_mb'] for m in training_metrics]
        cpu_values = [m['cpu_percent'] for m in training_metrics]
        thread_values = [m['threads'] for m in training_metrics]

        initial_memory = self.metrics_history[0]['memory_mb'] if self.metrics_history else 0
        final_memory = self.metrics_history[-1]['memory_mb'] if self.metrics_history else 0

        return {
            'experiment': self.experiment_name,
            'method': self.method_name,
            'start_time': self.start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'duration_seconds': (datetime.now() - self.start_time).total_seconds(),

            'memory_initial_mb': initial_memory,
            'memory_final_mb': final_memory,
            'memory_peak_mb': max(memory_values) if memory_values else 0,
            'memory_avg_mb': sum(memory_values) / len(memory_values) if memory_values else 0,
            'memory_growth_mb': final_memory - initial_memory,

            'cpu_peak_percent': max(cpu_values) if cpu_values else 0,
            'cpu_avg_percent': sum(cpu_values) / len(cpu_values) if cpu_values else 0,

            'threads_avg': sum(thread_values) / len(thread_values) if thread_values else 0,
            'threads_max': max(thread_values) if thread_values else 0,

            'efficiency_score': self._calculate_efficiency_score(
                sum(memory_values) / len(memory_values) if memory_values else 0,
                sum(cpu_values) / len(cpu_values) if cpu_values else 0,
                final_memory - initial_memory
            )
        }

    def _calculate_efficiency_score(self, avg_memory: float, avg_cpu: float, memory_growth: float) -> float:
        memory_score = min(avg_memory / 1000, 1.0)  # Assume 1 GB as the upper bound.
        cpu_score = min(avg_cpu / 100, 1.0)  # Assume 100% as the upper bound.
        growth_score = min(abs(memory_growth) / 500, 1.0)  # Assume 500 MB growth as the upper bound.
        efficiency = (memory_score * 0.4) + (cpu_score * 0.3) + (growth_score * 0.3)
        return round(efficiency, 3)

    def _save_results(self, efficiency_summary: Dict):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_filename = f"{self.experiment_name}_{self.method_name}"

        summary_file = os.path.join(self.output_dir, f"{base_filename}_summary_{timestamp}.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(efficiency_summary, f, indent=2, ensure_ascii=False)

        if self.metrics_history:
            df = pd.DataFrame(self.metrics_history)
            history_file = os.path.join(self.output_dir, f"{base_filename}_history_{timestamp}.csv")
            df.to_csv(history_file, index=False, encoding='utf-8')

        self._update_comparison_file(efficiency_summary)

        self._print_summary(efficiency_summary)

        print(f"\n💾 The results have been saved to: {self.output_dir}/")
        print(f"  Efficiency Summary: {os.path.basename(summary_file)}")
        if self.metrics_history:
            print(f"  Detailed data: {os.path.basename(history_file)}")

    def _update_comparison_file(self, efficiency_summary: Dict):
        comparison_file = os.path.join(self.output_dir, f"{self.experiment_name}_comparison.csv")

        comparison_data = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'method': efficiency_summary['method'],
            'duration_seconds': efficiency_summary['duration_seconds'],
            'memory_peak_mb': efficiency_summary['memory_peak_mb'],
            'memory_avg_mb': efficiency_summary['memory_avg_mb'],
            'memory_growth_mb': efficiency_summary['memory_growth_mb'],
            'cpu_avg_percent': efficiency_summary['cpu_avg_percent'],
            'efficiency_score': efficiency_summary['efficiency_score']
        }

        df_new = pd.DataFrame([comparison_data])

        if os.path.exists(comparison_file):
            df_existing = pd.read_csv(comparison_file)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_combined = df_new

        df_combined.to_csv(comparison_file, index=False, encoding='utf-8')

    def _print_summary(self, efficiency_summary: Dict):
        print(f"\n{'=' * 60}")
        print(f"📊 Training efficiency report: {efficiency_summary['experiment']} - {efficiency_summary['method']}")
        print(f"{'=' * 60}")

        print(f"⏱️  Training duration: {efficiency_summary['duration_seconds']:.1f} seconds")

        print(f"\n💾 Memory usage:")
        print(f"  Initial: {efficiency_summary['memory_initial_mb']:.1f} MB")
        print(f"  Final: {efficiency_summary['memory_final_mb']:.1f} MB")
        print(f"  Peak: {efficiency_summary['memory_peak_mb']:.1f} MB")
        print(f"  Average: {efficiency_summary['memory_avg_mb']:.1f} MB")
        print(f"  Growth: {efficiency_summary['memory_growth_mb']:+.1f} MB")

        print(f"\n⚡ CPU usage:")
        print(f"  Peak: {efficiency_summary['cpu_peak_percent']:.1f}%")
        print(f"  Average: {efficiency_summary['cpu_avg_percent']:.1f}%")

        print(f"\n🧵 Thread usage:")
        print(f"  Average: {efficiency_summary['threads_avg']:.1f}")
        print(f"  Maximum: {efficiency_summary['threads_max']}")

        print(f"\n⭐ Overall efficiency score: {efficiency_summary['efficiency_score']} lower is better")
        print(f"{'=' * 60}")


class SurrogateLossPredictor(nn.Module):
    """
    Lightweight surrogate model: predicts the total cross-entropy loss given (src, seg, tgt, soft_tgt, soft_alpha, soft_targets).
    
    """
    def __init__(self, args):
        super(SurrogateLossPredictor, self).__init__()

        self.emb_size = args.emb_size
        self.labels_num = args.labels_num
        # H_loss is the low-dimensional feature space used by the surrogate.
        self.loss_hidden_size = args.loss_hidden_size
        self.soft_targets = args.soft_targets
        self.soft_alpha = args.soft_alpha
        self.dropout = nn.Dropout(args.dropout)

        # 2. Sequence feature extractor: [emb_size -> H_loss]
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.emb_size, self.loss_hidden_size),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(self.loss_hidden_size, self.loss_hidden_size),
            nn.ReLU(),
            nn.Dropout(args.dropout),
        )

        # 3. Hard-label embedding: [labels_num -> H_loss]
        self.target_embedding = nn.Embedding(self.labels_num, self.loss_hidden_size)

        # 4. Soft-target projector: [labels_num -> H_loss]
        self.soft_tgt_projector = nn.Sequential(
            nn.Linear(self.labels_num, self.loss_hidden_size),
            nn.ReLU()
        )

        # 5. Number of hyperparameter features (soft_alpha, soft_targets)
        self.num_hyper_features = 2

        # 6. Loss prediction head.
        # Input dimension = sequence H_loss + hard-label H_loss + soft-label H_loss + two hyperparameter features.
        if self.soft_targets:
            final_feature_dim = (self.loss_hidden_size * 3) + self.num_hyper_features
        else:
            final_feature_dim = self.loss_hidden_size * 2
        self.loss_predictor = nn.Sequential(
            nn.Linear(final_feature_dim, final_feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(final_feature_dim//2, final_feature_dim // 2),
            nn.ReLU(),
            nn.Linear(final_feature_dim // 2, 1) # Output a scalar predicted loss.
        )

    def forward(self, emb, tgt, soft_tgt):
        """
        Args:
            src: [batch_size x seq_length]
            seg: [batch_size x seq_length]
            tgt: [batch_size]
            soft_tgt: [batch_size x labels_num] (Logits/Probabilities)
            soft_alpha: [batch_size] (floating-point weight)
            soft_targets: [batch_size] (Boolean value represented as 0.0 or 1.0 floating point)

        Returns:
            predicted_loss: [batch_size x 1]
        """

        features = self.feature_extractor(emb)
        pooled_features = torch.mean(features, dim=(1,2)) # Global Mean Pooling
        pooled_features = self.dropout(pooled_features)
        tgt_emb = self.target_embedding(tgt.view(-1))
        if tgt is not None:
            if self.soft_targets and soft_tgt is not None:
                soft_tgt_features = self.soft_tgt_projector(soft_tgt)
                if self.soft_alpha.dim() == 1:
                    self.soft_alpha = self.soft_alpha.unsqueeze(-1)
                if self.soft_targets.dim() == 1:
                    self.soft_targets = self.soft_targets.unsqueeze(-1)

                hyper_features = torch.cat((self.soft_alpha, self.soft_targets), dim=-1)
                combined_features = torch.cat((pooled_features, tgt_emb, soft_tgt_features, hyper_features), dim=-1)
                predicted_loss = self.loss_predictor(combined_features)
            else:
                combined_features = torch.cat((pooled_features, tgt_emb), dim=-1)
                predicted_loss = self.loss_predictor(combined_features)
        return predicted_loss


class Classifier(nn.Module):
    def __init__(self, args):
        super(Classifier, self).__init__()
        self.embedding = str2embedding[args.embedding](args, len(args.tokenizer.vocab))
        self.encoder = str2encoder[args.encoder](args)
        self.labels_num = args.labels_num
        self.pooling = args.pooling
        self.soft_targets = args.soft_targets
        self.soft_alpha = args.soft_alpha
        self.method = args.method
        self.focal_gamma = getattr(args, "focal_gamma", 2.0)
        self.focal_alpha = getattr(args, "focal_alpha", 1.0)
        if self.method == "TDRO":
            self.tdro_alpha = nn.Parameter(torch.zeros(()))
        self.output_layer_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.output_layer_2 = nn.Linear(args.hidden_size, self.labels_num)

    def _hard_label_loss(self, logits, tgt):
        log_probs = F.log_softmax(logits, dim=1)
        per_sample_ce = -log_probs.gather(dim=1, index=tgt.unsqueeze(1)).squeeze(1)
        if self.method != "Focal":
            return per_sample_ce

        probs = log_probs.exp()
        pt = probs.gather(dim=1, index=tgt.unsqueeze(1)).squeeze(1).clamp(min=1e-8, max=1.0)
        focal_weight = (1.0 - pt).pow(self.focal_gamma)
        return self.focal_alpha * focal_weight * per_sample_ce

    def forward(self, src, tgt, seg, soft_tgt=None):
        """
        Args:
            src: [batch_size x seq_length]
            tgt: [batch_size]
            seg: [batch_size x seq_length]
        """
        

        
        if src.dim() == 3:
            batch_size_num, num_chunks, seq_length = src.size()

            src_flat = src.contiguous().view(-1, seq_length)   # [B*5, 128]
            seg_flat = seg.contiguous().view(-1, seq_length)   # [B*5, 128]

            emb_flat = self.embedding(src_flat, seg_flat)      # [B*5, 128, H]
            hidden_size = emb_flat.size(-1)
            emb_data = emb_flat.view(batch_size_num, num_chunks, seq_length, hidden_size)
        else:
            batch_size_num = src.size(0)
            seq_length = src.size(1)
            num_chunks = 1

            emb_flat = self.embedding(src, seg)                # [B, L, H]
            hidden_size = emb_flat.size(-1)
            emb_data = emb_flat.view(batch_size_num, num_chunks, seq_length, hidden_size)

        # Process samples through the encoder one by one, following the original logic; output shape: [batch_size x 5 x 768].
        output = torch.Tensor(0).to(src.device)  # Move to the input device (GPU/CPU).
        for each_batch_size in range(emb_data.size(0)):
            emb = emb_data[each_batch_size]  # [5 x seq_length x 768]
            seg_data = seg[each_batch_size]  # [5 x seq_length]
            output_emb = self.encoder(emb, seg_data)  # [5 x seq_length x 768]
            output_data = output_emb[:, :1, :]  # Take the first token of each subsequence, e.g., CLS: [5 x 1 x 768].
            cls_output = output_data.squeeze(1).unsqueeze(0)  # [1 x 5 x 768]
            if output.size(0) == 0:
                output = cls_output
            else:
                output = torch.cat((output, cls_output), 0)  # Final shape: [batch_size x 5 x 768].

        # Pooling operation following the original logic.
        if self.pooling == "mean":
            output = torch.mean(output, dim=1)  # [batch_size x 768]
        elif self.pooling == "max":
            output = torch.max(output, dim=1)[0]  # [batch_size x 768]
        elif self.pooling == "last":
            output = output[:, -1, :]  # [batch_size x 768]
        else:  # "cls"（default）
            output = output[:, 0, :]  # [batch_size x 768]

        # Output layer following the original logic.
        output = torch.tanh(self.output_layer_1(output))  # [batch_size x hidden_dim]
        logits = self.output_layer_2(output)  # [batch_size x output_dim]：Raw prediction values.

        ### 2. Main change: compute both batch-mean loss and per-sample loss.
        loss = None
        per_sample_loss = None  # Store the loss of each sample; shape: [batch_size].

        if tgt is not None:
            if self.soft_targets and soft_tgt is not None:
                # Hybrid loss: soft-label MSE plus hard-label NLL, both retaining the per-sample dimension.
                ##### Soft-label MSE loss per sample.
                mse_loss_per_element = F.mse_loss(logits, soft_tgt, reduction='none')  # [batch_size x output_dim]
                per_sample_mse = mse_loss_per_element.sum(dim=1)  # Sum of MSE for each sample: [batch_size].

                ##### Hard-label NLL loss per sample.
                log_probs = F.log_softmax(logits, dim=1)  # [batch_size x output_dim]
                # Extract the log probability of the target class for each sample; gather avoids automatic averaging.
                per_sample_nll = -log_probs.gather(
                    dim=1,
                    index=tgt.unsqueeze(1)  # [batch_size] -> [batch_size x 1]（Match the gather dimension.）
                ).squeeze(1)  # [batch_size x 1] -> [batch_size]

                ##### Hybrid per-sample loss and batch-mean loss.
                if self.method == "Focal":
                    per_sample_nll = self._hard_label_loss(logits, tgt)
                per_sample_loss = self.soft_alpha * per_sample_mse + (1 - self.soft_alpha) * per_sample_nll
                loss = per_sample_loss.mean()  # Mean loss used for backpropagation.

            else:
                # Hard labels only: per-sample NLL loss.
                log_probs = F.log_softmax(logits, dim=1)  # [batch_size x output_dim]
                per_sample_loss = -log_probs.gather(
                    dim=1,
                    index=tgt.unsqueeze(1)
                ).squeeze(1)  # [batch_size]
                if self.method == "Focal":
                    per_sample_loss = self._hard_label_loss(logits, tgt)
                loss = per_sample_loss.mean()  # Batch-mean loss.

        ### 3. Return mean loss for training, per-sample loss for analysis, and logits for prediction.
        return loss, per_sample_loss, logits


def compute_pearson_corr(pred_loss, true_loss):
        """Pearson correlation coefficient, an improved version of the original method."""
        if len(pred_loss) < 2:
            return 0.0
        
        corr, p_value = pearsonr(pred_loss, true_loss)
        return {
            'pearson_r': float(corr),
            'p_value': float(p_value),
            'significant': p_value < 0.05
        }


def compute_spearman_corr(pred_loss, true_loss):
    """Spearman rank correlation coefficient; more robust to outliers."""
    if len(pred_loss) < 2:
        return 0.0
    
    corr, p_value = spearmanr(pred_loss, true_loss)
    return {
        'spearman_rho': float(corr) if not np.isnan(corr) else 0.0,
        'p_value': float(p_value) if not np.isnan(p_value) else 1.0
    }


def precision_at_k(pred_scores, true_losses, k=32):
    """
    P@K = |top-K(proxy) ∩ top-K(true_loss)| / K
    """
    top_k_pred = set(np.argsort(pred_scores)[-k:])
    top_k_true = set(np.argsort(true_losses)[-k:])
    return len(top_k_pred & top_k_true) / k


def get_spearman_moving_average(history_deque):
    if len(history_deque) == 0:
        return None
    return float(sum(history_deque) / len(history_deque))


def schedule_proxy_steps(avg_spearman, args):
    """
    Dynamic schedule based on moving-average Spearman.
    """
    if avg_spearman is None:
        return args.proxy_steps_high

    if avg_spearman > args.proxy_spearman_low:
        return args.proxy_steps_high
    else:
        return args.proxy_steps_low


# Use pairwise ranking loss instead of proxy regression loss.
def pairwise_ranking_loss(
    pred_scores,
    true_scores,
    margin=1.0,          # Kept for interface compatibility; not directly used in the softplus version.
    max_pairs=512,
    min_delta=1e-4,
    temperature=1.0,
    weighted=False,
    weight_clip=5.0,
    top_ratio=None,
):
    """
    Pairwise ranking loss with optional pair weighting.

    Args:
        pred_scores: Proxy-predicted scores, shape [N].
        true_scores: Target scores for ranking, shape [N].
        margin: Kept for compatibility; not used in softplus loss.
        max_pairs: Maximum number of sampled pairs. If None, use all valid pairs.
        min_delta: Ignore pairs whose true-score difference is too small.
        temperature: Temperature for scaling prediction differences.
        weighted: Whether to weight pairs by true-score difference.
        weight_clip: Upper bound for normalized pair weights.

    Returns:
        Scalar ranking loss.
    """
    device = pred_scores.device
    n = pred_scores.size(0)

    if n < 2:
        return pred_scores.sum() * 0.0

    # Use upper-triangular pairs to avoid duplicate symmetric pairs.
    i_idx, j_idx = torch.triu_indices(n, n, offset=1, device=device)

    diff_true = true_scores[i_idx] - true_scores[j_idx]
    diff_pred = pred_scores[i_idx] - pred_scores[j_idx]

    # Remove pairs whose target scores are nearly tied.
    valid = diff_true.abs() > min_delta
    if top_ratio is not None:
        k = max(1, int(n * top_ratio))
        top_idx = torch.topk(true_scores, k=k, largest=True).indices
        top_mask = torch.zeros(n, dtype=torch.bool, device=device)
        top_mask[top_idx] = True

        pair_involves_top = top_mask[i_idx] | top_mask[j_idx]
        valid = valid & pair_involves_top
    if valid.sum() == 0:
        return pred_scores.sum() * 0.0

    diff_true = diff_true[valid]
    diff_pred = diff_pred[valid]

    # Randomly sample pairs if needed.
    if max_pairs is not None and diff_pred.numel() > max_pairs:
        idx = torch.randperm(diff_pred.numel(), device=device)[:max_pairs]
        diff_true = diff_true[idx]
        diff_pred = diff_pred[idx]

    # +1 means i should rank above j; -1 means j should rank above i.
    target = torch.sign(diff_true)

    # Smooth pairwise ranking loss.
    loss = F.softplus(-target * diff_pred / temperature)

    if weighted:
        # Larger true-score gaps provide more reliable ranking supervision.
        pair_weight = diff_true.abs().detach()
        pair_weight = pair_weight / (pair_weight.mean() + 1e-8)

        # Avoid a few large-gap pairs dominating the proxy update.
        if weight_clip is not None:
            pair_weight = torch.clamp(pair_weight, max=weight_clip)

        loss = loss * pair_weight

    return loss.mean()


def count_labels_num(path):
    labels_set, columns = set(), {}
    with open(path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            if line_id == 0:
                for i, column_name in enumerate(line.strip().split("\t")):
                    columns[column_name] = i
                continue
            line = line.strip().split("\t")
            label = int(line[columns["label"]])
            labels_set.add(label)
    return len(labels_set)


def _torch_load_checkpoint(model_path):
    try:
        return torch.load(model_path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(model_path, map_location="cpu")


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            nested_state = checkpoint.get(key)
            if isinstance(nested_state, dict):
                return nested_state
    return checkpoint


def load_or_initialize_parameters(args, model):
    if args.pretrained_model_path is not None:
        if not os.path.exists(args.pretrained_model_path):
            raise FileNotFoundError(
                "TrafficFormer pretrained checkpoint not found: "
                f"{args.pretrained_model_path}\n"
                "Download the TrafficFormer pretrained model from "
                f"{TRAFFICFORMER_PRETRAINED_MODEL_URL} and put it at this path, "
                "or pass --pretrained_model_path explicitly."
            )

        checkpoint = _torch_load_checkpoint(args.pretrained_model_path)
        pretrained_state = _extract_state_dict(checkpoint)
        if not isinstance(pretrained_state, dict):
            raise ValueError(f"Unsupported checkpoint format: {args.pretrained_model_path}")

        model_state = model.state_dict()
        compatible_state = {}
        skipped_by_shape = []
        unexpected_keys = []

        for key, value in pretrained_state.items():
            normalized_key = key[7:] if key.startswith("module.") else key
            if not torch.is_tensor(value):
                unexpected_keys.append(normalized_key)
                continue
            if normalized_key not in model_state:
                unexpected_keys.append(normalized_key)
                continue
            if tuple(model_state[normalized_key].shape) != tuple(value.shape):
                skipped_by_shape.append(
                    (normalized_key, tuple(value.shape), tuple(model_state[normalized_key].shape))
                )
                continue
            compatible_state[normalized_key] = value

        if not compatible_state:
            raise ValueError(
                "No compatible backbone parameters were found in "
                f"{args.pretrained_model_path}. Please check --config_path, --vocab_path, "
                "--embedding, --encoder and --mask."
            )

        missing_keys, _ = model.load_state_dict(compatible_state, strict=False)
        print(f"Initialized backbone from pretrained checkpoint: {args.pretrained_model_path}")
        print(
            "Loaded compatible keys: "
            f"{len(compatible_state)}; ignored checkpoint-only keys: {len(unexpected_keys)}; "
            f"shape-mismatched keys: {len(skipped_by_shape)}; model-missing keys: {len(missing_keys)}"
        )
        if skipped_by_shape:
            print("First shape-mismatched keys:", skipped_by_shape[:5])
    else:
        # Initialize with normal distribution.
        for n, p in list(model.named_parameters()):
            if n == "tdro_alpha":
                p.data.zero_()
                continue
            if "gamma" not in n and "beta" not in n:
                p.data.normal_(0, 0.02)


def build_optimizer(args, model):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta', 'tdro_alpha']
    optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]
    if args.optimizer in ["adamw"]:
        optimizer = str2optimizer[args.optimizer](optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=False)
    else:
        optimizer = str2optimizer[args.optimizer](optimizer_grouped_parameters, lr=args.learning_rate,
                                                    scale_parameter=False, relative_step=False)
    if args.scheduler in ["constant"]:
        scheduler = str2scheduler[args.scheduler](optimizer)
    elif args.scheduler in ["constant_with_warmup"]:
        scheduler = str2scheduler[args.scheduler](optimizer, args.train_steps*args.warmup)
    else:
        scheduler = str2scheduler[args.scheduler](optimizer, args.train_steps*args.warmup, args.train_steps)
    return optimizer, scheduler


def get_tdro_alpha(model):
    module = model.module if hasattr(model, "module") else model
    if not hasattr(module, "tdro_alpha"):
        raise AttributeError("TDRO requires Classifier.tdro_alpha.")
    return module.tdro_alpha


def tdro_softplus_loss(per_sample_loss, alpha, rho, lam):
    rho = float(rho)
    lam = float(lam)
    if not (0.0 < rho < 1.0):
        raise ValueError("--tdro_rho must be in (0, 1).")
    if lam <= 0.0:
        raise ValueError("--tdro_lambda must be positive.")

    losses = per_sample_loss.view(-1)
    exponent = (losses - alpha) / lam + math.log(rho)
    return alpha + (lam / rho) * F.softplus(exponent).mean()


def batch_loader(batch_size, src, tgt, seg, soft_tgt=None):
    instances_num = src.size()[0]
    for i in range(instances_num // batch_size):
        src_batch = src[i * batch_size : (i + 1) * batch_size, :]
        tgt_batch = tgt[i * batch_size : (i + 1) * batch_size]
        seg_batch = seg[i * batch_size : (i + 1) * batch_size, :]
        if soft_tgt is not None:
            soft_tgt_batch = soft_tgt[i * batch_size : (i + 1) * batch_size, :]
            yield src_batch, tgt_batch, seg_batch, soft_tgt_batch
        else:
            yield src_batch, tgt_batch, seg_batch, None
    if instances_num > instances_num // batch_size * batch_size:
        src_batch = src[instances_num // batch_size * batch_size :, :]
        tgt_batch = tgt[instances_num // batch_size * batch_size :]
        seg_batch = seg[instances_num // batch_size * batch_size :, :]
        if soft_tgt is not None:
            soft_tgt_batch = soft_tgt[instances_num // batch_size * batch_size :, :]
            yield src_batch, tgt_batch, seg_batch, soft_tgt_batch
        else:
            yield src_batch, tgt_batch, seg_batch, None


def read_dataset(args, path):
    dataset, columns = [], {}

    with open(path, mode="r", encoding="utf-8") as f:
        try:
            for line_id, line in enumerate(f):
                if line_id == 0:
                    for i, column_name in enumerate(line.strip().split("\t")):
                        columns[column_name] = i
                    continue
                line = line[:-1].split("\t")
                tgt = int(line[columns["label"]])
                if args.soft_targets and "logits" in columns.keys():
                    soft_tgt = [float(value) for value in line[columns["logits"]].split(" ")]

                src_dataset,seg_dataset = [], [] # not source code
                if "text_b" not in columns:  # Sentence classification.
                    text_a = line[columns["text_a"]]
                    ### source code as up
                    text_a_list = text_a.split(" | ")
                    if text_a_list:
                        for text_a_index in range(len(text_a_list)):
                            src = args.tokenizer.convert_tokens_to_ids([CLS_TOKEN] + args.tokenizer.tokenize(text_a_list[text_a_index]))
                            src_dataset.append(src)
                            seg_dataset.append([1] * len(src))
                    else:
                        print("BBBB ",text_a_list," BBBBBBBBBBBBB",path)
                else:  # Sentence-pair classification.
                    text_a, text_b = line[columns["text_a"]], line[columns["text_b"]]
                    src_a = args.tokenizer.convert_tokens_to_ids([CLS_TOKEN] + args.tokenizer.tokenize(text_a) + [SEP_TOKEN])
                    src_b = args.tokenizer.convert_tokens_to_ids(args.tokenizer.tokenize(text_b) + [SEP_TOKEN])
                    src = src_a + src_b
                    seg = [1] * len(src_a) + [2] * len(src_b)

                if src_dataset:
                    for index in range(len(src_dataset)):
                        if len(src_dataset[index]) > args.seq_length:
                            src_dataset[index] = src_dataset[index][: args.seq_length]
                            seg_dataset[index] = seg_dataset[index][: args.seq_length]
                        while len(src_dataset[index]) < args.seq_length:
                            src_dataset[index].append(0)
                            seg_dataset[index].append(0)
                else:
                    print("BBBB ",text_a_list," BBBBBBBBBBBBB",path)
                src = src_dataset # src [5]
                seg = seg_dataset

                if args.soft_targets and "logits" in columns.keys():
                    dataset.append((src, tgt, seg, soft_tgt))
                else:
                    dataset.append((src, tgt, seg))
        except Exception as e:
            print(path)
            print(e)

    return dataset


def train_model(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch, soft_tgt_batch=None,inference=False):
    if inference:
        src_batch = src_batch.to(args.device)
        tgt_batch = tgt_batch.to(args.device)
        seg_batch = seg_batch.to(args.device)
        if soft_tgt_batch is not None:
            soft_tgt_batch = soft_tgt_batch.to(args.device)

        loss, loss_, _ = model(src_batch, tgt_batch, seg_batch, soft_tgt_batch)
        if torch.cuda.device_count() > 1:
            loss = torch.mean(loss)
            loss_ = loss_.view(-1)
    else:
        model.zero_grad()

        src_batch = src_batch.to(args.device)  # torch.Size([64, 1, 128])
        tgt_batch = tgt_batch.to(args.device)
        seg_batch = seg_batch.to(args.device)
        if soft_tgt_batch is not None:
            soft_tgt_batch = soft_tgt_batch.to(args.device)
        loss, loss_, _ = model(src_batch, tgt_batch, seg_batch, soft_tgt_batch)
        if torch.cuda.device_count() > 1:
            loss = torch.mean(loss)

            loss_ = loss_.view(-1)
        if args.fp16:
            with args.amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()
        scheduler.step()

    return loss,loss_


def pool_selector_embeddings(emb):
    """
    Convert proxy embeddings to [batch_size x hidden_dim] for subset selection.
    """
    if emb.dim() == 2:
        return emb.float()
    if emb.dim() < 2:
        raise ValueError("Expected embeddings with at least 2 dimensions.")
    reduce_dims = tuple(range(1, emb.dim() - 1))
    if len(reduce_dims) == 0:
        return emb.float()
    return emb.float().mean(dim=reduce_dims)


def compute_rbf_bandwidth(candidate_embeddings, min_sigma=1e-6):
    """
    Median pairwise L2 distance on candidate embeddings.
    """
    n = candidate_embeddings.size(0)
    if n < 2:
        return candidate_embeddings.new_tensor(max(float(min_sigma), 1.0))
    pairwise_dist = torch.cdist(candidate_embeddings, candidate_embeddings, p=2)
    triu_i, triu_j = torch.triu_indices(n, n, offset=1, device=candidate_embeddings.device)
    if triu_i.numel() == 0:
        return candidate_embeddings.new_tensor(max(float(min_sigma), 1.0))
    sigma = pairwise_dist[triu_i, triu_j].median()
    sigma = torch.clamp(sigma, min=float(min_sigma))
    return sigma


def compute_rbf_novelty_penalty(
    candidate_embeddings,
    memory_embeddings,
    sigma,
    reduction="topk_mean",
    gamma=0.5,
    tau=0.7,
    topk=5,
    clip_quantile=None,
):
    """
    Compute RBF-based history novelty penalty.

    Args:
        candidate_embeddings: Candidate embeddings, shape [N, D].
        memory_embeddings: Historical memory embeddings, shape [M, D].
        sigma: RBF kernel bandwidth.
        reduction:
            - "sum": raw sum of similarities.
            - "mean": average similarity over memory.
            - "normalized_sum": sum / |memory|^gamma.
            - "softmax": softmax-weighted similarity aggregation.
        gamma: Exponent used only for "normalized_sum".
        tau: Temperature used only for "softmax".

    Returns:
        penalty: Novelty penalty for each candidate, shape [N].
        
    """
    if memory_embeddings is None or memory_embeddings.numel() == 0:
        return torch.zeros(
            candidate_embeddings.size(0),
            device=candidate_embeddings.device,
            dtype=candidate_embeddings.dtype
        )

    dist = torch.cdist(candidate_embeddings, memory_embeddings, p=2)
    sim = torch.exp(-(dist ** 2) / (2.0 * (sigma ** 2)))

    if reduction == "sum":
        penalty = sim.sum(dim=1)

    elif reduction == "mean":
        penalty = sim.mean(dim=1)

    elif reduction == "normalized_sum":
        memory_size = memory_embeddings.size(0)
        norm_factor = float(memory_size) ** gamma
        penalty = sim.sum(dim=1) / norm_factor

    elif reduction == "softmax":
        weights = torch.softmax(sim / tau, dim=1)
        penalty = (weights * sim).sum(dim=1)

    elif reduction == "topk_mean":
        k = min(topk, sim.size(1))
        topk_vals, _ = torch.topk(sim, k=k, dim=1)
        penalty = topk_vals.mean(dim=1)

    else:
        raise ValueError(f"Unsupported reduction: {reduction}")
    
    if clip_quantile is not None:
        clip_val = torch.quantile(penalty.detach(), clip_quantile)
        penalty = torch.clamp(penalty, max=clip_val)
    
    return penalty


#     Args:
#         candidate_embeddings: Tensor of shape [N, D]
#         memory_embeddings: Tensor of shape [M, D]
#         sigma: RBF bandwidth
#         tau: softmax temperature; smaller tau makes the aggregation closer to max


def evaluate(args, dataset, print_confusion_matrix=False):
    src = torch.LongTensor([sample[0] for sample in dataset])
    tgt = torch.LongTensor([sample[1] for sample in dataset])
    seg = torch.LongTensor([sample[2] for sample in dataset])

    batch_size = args.batch_size

    correct = 0
    # Confusion matrix.
    confusion = torch.zeros(args.labels_num, args.labels_num, dtype=torch.long)

    args.model.eval()

    loss_list = None
    pred_list = []  # [ADDED]
    gold_list = []  # [ADDED]

    for i, (src_batch, tgt_batch, seg_batch, _) in enumerate(batch_loader(batch_size, src, tgt, seg)):
        src_batch = src_batch.to(args.device)
        tgt_batch = tgt_batch.to(args.device)
        seg_batch = seg_batch.to(args.device)
        with torch.no_grad():
            _, loss_, logits = args.model(src_batch, tgt_batch, seg_batch)

        if loss_list is None:  # [MODIFIED]
            loss_list = loss_
        else:
            loss_list = torch.cat([loss_list, loss_], dim=0)

        pred = torch.argmax(logits, dim=1)  # [MODIFIED]
        gold = tgt_batch

        pred_list.append(pred.detach().cpu())  # [ADDED]
        gold_list.append(gold.detach().cpu())  # [ADDED]

        for j in range(pred.size()[0]):
            confusion[pred[j], gold[j]] += 1
        correct += torch.sum(pred == gold).item()
        if i % 100 == 0:
            # \r returns the cursor to the beginning of the line; end="" prevents an automatic newline; flush=True forces real-time display.
            print(f"\rc:{i}/{args.train_steps}", end='', flush=True)

    loss_list = loss_list.view(-1)
    loss_list_cpu = loss_list.detach().cpu()  # [ADDED]
    pred_list = torch.cat(pred_list, dim=0)   # [ADDED]
    gold_list = torch.cat(gold_list, dim=0)   # [ADDED]

    CVaR_list = []
    alpha_list = [0., 0.5, 0.7, 0.9, 0.95]
    acc_list = []
    prec_list = []
    rec_list = []
    f1_list = []

    for a in alpha_list:
        k = int(len(loss_list_cpu) * (1 - a))  # [MODIFIED]
        loss_alpha, idx_alpha = torch.topk(loss_list_cpu, k=k)  # [MODIFIED]
        CVaR_list.append(torch.mean(loss_alpha).item())
        correctm = torch.sum(pred_list[idx_alpha] == gold_list[idx_alpha]).item()  # [MODIFIED]
        acc_list.append(correctm / k)  # [MODIFIED]

        subset_confusion = torch.zeros(args.labels_num, args.labels_num, dtype=torch.long)
        for pred_label, gold_label in zip(pred_list[idx_alpha], gold_list[idx_alpha]):
            subset_confusion[pred_label, gold_label] += 1

        label_precisions = []
        label_recalls = []
        label_f1s = []
        for label_id in range(args.labels_num):
            tp = subset_confusion[label_id, label_id].item()
            predicted_num = subset_confusion[label_id, :].sum().item()
            gold_num = subset_confusion[:, label_id].sum().item()
            precision = tp / predicted_num if predicted_num > 0 else 0.0
            recall = tp / gold_num if gold_num > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
            label_precisions.append(precision)
            label_recalls.append(recall)
            label_f1s.append(f1)

        prec_list.append(float(np.mean(label_precisions)))
        rec_list.append(float(np.mean(label_recalls)))
        f1_list.append(float(np.mean(label_f1s)))

    rem = np.zeros(shape=(confusion.size()[0], 4))
    if print_confusion_matrix:
        print("Confusion matrix:")
        print(confusion)
        print("Report precision, recall, and f1:")
        for i in range(confusion.size()[0]):
            p = confusion[i, i].item() / confusion[i, :].sum().item()
            r = confusion[i, i].item() / confusion[:, i].sum().item()
            f1 = 2 * p * r / (p + r)
            rem[i:i+1, :] = [i, p, r, f1]
            print("Label {}: {:.4f}, {:.4f}, {:.4f}".format(i, p, r, f1))

    print("Acc. (Correct/Total): {:.4f} ({}/{}) ".format(correct / len(dataset), correct, len(dataset)))
    return correct / len(dataset), confusion, CVaR_list, acc_list, prec_list, rec_list, f1_list, rem


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    finetune_opts(parser)
    parser.set_defaults(
        pretrained_model_path=TRAFFICFORMER_PRETRAINED_MODEL_PATH,
        vocab_path=TRAFFICFORMER_VOCAB_PATH,
        config_path=TRAFFICFORMER_CONFIG_PATH,
        embedding="word_pos_seg",
        encoder="transformer",
        mask="fully_visible",
    )

    parser.add_argument("--method", choices=["ERM", "HISS-no-penalty", "GroupDRO", "MC-CVaR", "HISS", "Focal", "Random", "OHTM", "TDRO"],
                        default="ERM",
                        help="the method of finetune")
    parser.add_argument("--CVaR_alpha", default=0.5, type=float,
                        help="Robust param")
    parser.add_argument("--loss_hidden_size", default=256, type=int,
                        help="riskmodel dim")

    parser.add_argument("--pooling", choices=["mean", "max", "first", "last"], default="first",
                        help="Pooling type.")

    parser.add_argument("--tokenizer", choices=["bert", "char", "space"], default="bert",
                        help="Specify the tokenizer."
                             "Original Google BERT uses bert tokenizer on Chinese corpus."
                             "Char tokenizer segments sentences into characters."
                             "Space tokenizer segments sentences into words according to space."
                             )

    parser.add_argument("--soft_targets", action='store_true',
                        help="Train model with logits.")
    parser.add_argument("--soft_alpha", type=float, default=0.5,
                        help="Weight of the soft targets loss.")
    parser.add_argument("--only_test", type=bool, default=False,
                        help="only test")
    parser.add_argument("--rm_lr", type=float,
                        help="learning_rate of riskmodel")
    parser.add_argument("--hiss_memory_capacity", type=int, default=96,
                        help="Capacity of the HISS high-risk memory bank.")  # 32 samples times 3 rounds.
    parser.add_argument("--hiss_rbf", choices=["mean", "max", "softmax", "topk_mean", "normalized_sum"], default="topk_mean",
                        help="RBF type.")
    parser.add_argument("--hiss_topk", type=int, default=3,
                        help="Top-k mean of RBF similarity.")  # k=3,5  
    parser.add_argument("--hiss_tau", type=float, default=0.3,
                        help="Softmax of RBF similarity.")    # tau=0.3
    parser.add_argument("--hiss_gamma", type=float, default=0.5,
                        help="Normalized sum of RBF similarity.")    # gamma=0.5,0.7
    parser.add_argument("--hiss_lambda_base", type=float, default=1.0,
                        help="Base lambda value used in HISS score: risk - lambda * novelty.")
    parser.add_argument("--hiss_min_sigma", type=float, default=1e-6,
                        help="Minimum HISS RBF bandwidth for subset selection.")
    # Dynamic scheduling of proxy_steps based on the moving-average Spearman correlation.
    parser.add_argument("--proxy_steps_high", type=int, default=1,
                    help="Proxy update steps when recent Spearman is high.")
    parser.add_argument("--proxy_steps_low", type=int, default=2,
                        help="Proxy update steps when recent Spearman is low.")
    parser.add_argument("--proxy_spearman_low", type=float, default=0.3,
                        help="Lower Spearman threshold for proxy step scheduling.")
    parser.add_argument("--proxy_spearman_window", type=int, default=20,
                        help="Window size for moving-average Spearman.")
    parser.add_argument("--proxy_schedule_warmup", type=int, default=10,
                        help="Minimum number of Spearman observations before enabling dynamic scheduling.")
    # Optimization parameters for pairwise ranking loss.
    parser.add_argument("--warmup_erm", type=int, default=2,
                        help="the epoch of warmup using ERM.")
    parser.add_argument("--min_delta", type=float, default=1e-6,
                        help="Ignore pairs whose true-score difference is too small.")
    # GroupDRO parameters.
    parser.add_argument("--gdro_tau", type=float, default=1.0,
                        help="Temperature parameter for softmax.")
    # TDRO parameters.
    parser.add_argument("--tdro_rho", type=float, default=1e-3,
                        help="Safe-KL/SoftPlus approximation accuracy parameter for TDRO.")
    parser.add_argument("--tdro_lambda", type=float, default=1.0,
                        help="KL-DRO penalty coefficient used by TDRO.")
    # Focal Loss parameters.
    parser.add_argument("--focal_gamma", type=float, default=2.0,
                        help="Gamma parameter for Focal loss.")
    parser.add_argument("--focal_alpha", type=float, default=1.0,
                        help="Alpha weight for Focal loss.")
    
    args = parser.parse_args()

    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)

    set_seed(args.seed)

    # Count the number of labels.
    args.labels_num = count_labels_num(args.train_path)

    # Build tokenizer.
    args.tokenizer = str2tokenizer[args.tokenizer](args)

    model = Classifier(args)

    # Load or initialize parameters.
    load_or_initialize_parameters(args, model)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(args.device)
    for p in model.embedding.parameters():
        p.requires_grad = False
    # Training phase.
    trainset = read_dataset(args, args.train_path)

    instances_num = len(trainset)
    batch_size = args.batch_size

    args.train_steps = int(instances_num * args.epochs_num / batch_size) + 1

    print("Batch size: ", batch_size)
    print("The number of training instances:", instances_num)

    optimizer, scheduler = build_optimizer(args, model)
    robust_selected_batch_size = max(1, int(batch_size * (1 - args.CVaR_alpha)))
    robust_candidate_batch_size = max(robust_selected_batch_size, int(batch_size/(1-args.CVaR_alpha)))
    if args.method == 'HISS-no-penalty' or args.method == 'HISS':
        riskmodel = SurrogateLossPredictor(args)
        riskmodel = riskmodel.to(args.device)
        if args.rm_lr == None:
            args.rm_lr = args.learning_rate
        optim = torch.optim.Adam(riskmodel.parameters(),lr=args.rm_lr)
        risk_lossfn = nn.SmoothL1Loss(beta=1.0)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
        args.amp = amp

    if torch.cuda.device_count() > 1:
        print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
    args.model = model

    total_loss, result, best_result = 0.0, 0.0, 0.0

    total_time = 0.0
    total_pearson_corr, total_spearman_corr, total_prec_k = 0.0, 0.0, 0.0
    metric_count = 0 # Counter for averaging post-warmup correlation and related metrics.
    avg_recent_spearman = None  # Moving-average Spearman used to dynamically adjust proxy_steps.
    
    # per epoch average metrics
    avg_time, avg_pearson_corr, avg_spearman_corr, avg_loss, avg_acc= 0.0, 0.0, 0.0, 0.0, 0.0
    # Initialize the Spearman history queue before training starts.
    proxy_spearman_history = deque(maxlen=args.proxy_spearman_window)

    if args.pretrained_model_path == "models/pre-trained_model.bin":
        backbone = "ET-BERT"
    else:
        backbone = "TrafficFormer"

    if args.method == "HISS":  # epoch 1 use erm to update classifier and also update proxy
        if args.hiss_rbf == "topk_mean":
            rbf_type_name = f"{args.hiss_rbf}={args.hiss_topk}"
        elif args.hiss_rbf == "normalized_sum":
            rbf_type_name = f"{args.hiss_rbf}={args.hiss_gamma}"
        elif args.hiss_rbf == "softmax":
            rbf_type_name = f"{args.hiss_rbf}={args.hiss_tau}"
        else:
            rbf_type_name = args.hiss_rbf
        save_path = f"./{backbone}_Results/Results_norm_{args.method}_update_proxy_{args.proxy_spearman_low}_seed={args.seed}_{DATASET}_seq={args.seq_length}_{args.batch_size}_{args.learning_rate}_{args.rm_lr}_lambda={args.hiss_lambda_base}_{args.hiss_memory_capacity}_{rbf_type_name}"
    elif args.method == "HISS-no-penalty":
        save_path = f"./{backbone}_Results/Results_{args.method}_update_proxy_{args.proxy_spearman_low}_seed={args.seed}_{DATASET}_seq={args.seq_length}_{args.batch_size}_{args.learning_rate}_{args.rm_lr}"
    elif args.method == "TDRO":
        save_path = f"./{backbone}_Results/Results_{args.method}_{DATASET}_seq={args.seq_length}_{args.batch_size}_lr={args.learning_rate}_seed={args.seed}_rho={args.tdro_rho}_lambda={args.tdro_lambda}"
    elif args.method == "Focal":
        save_path = f"./{backbone}_Results/Results_{args.method}_{DATASET}_seq={args.seq_length}_{args.batch_size}_lr={args.learning_rate}_seed={args.seed}_gamma={args.focal_gamma}_alpha={args.focal_alpha}"
    else:
        save_path = f"./{backbone}_Results/Results_{args.method}_{DATASET}_seq={args.seq_length}_{args.batch_size}_lr={args.learning_rate}_seed={args.seed}"
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(f"{save_path}/models", exist_ok=True)  # save finetuned models
    os.makedirs(f"{save_path}/logs", exist_ok=True)  # save logs

    log_config = {
        'loss': f"{save_path}/logs/{args.method}_avg_loss.txt",
        'time': f"{save_path}/logs/{args.method}_time.txt",
        'corr': f"{save_path}/logs/{args.method}_corr.txt",
        'train_list': f"{save_path}/logs/{args.method}_train_acc_list.txt"
    }
    if args.method == "TDRO":
        log_config['tdro_alpha'] = f"{save_path}/logs/{args.method}_alpha.txt"

    # Clear each log file path before training starts.
    for file_path in log_config.values():
        with open(file_path, 'w') as f:
            pass

    if args.only_test:
        if args.test_path is not None:
            print("Test set evaluation.")
            if torch.cuda.device_count() > 1:
                model.module.load_state_dict(torch.load(save_path+'/'+args.output_model_path,map_location='cuda:1'))
            else:
                model.load_state_dict(torch.load(save_path+'/'+args.output_model_path))
            with torch.no_grad():
                acc, confusion,CVaR_list,acc_list,prec_list,rec_list,f1_list,rem = evaluate(args, read_dataset(args, args.test_path), True)

            r1 = np.array([acc]) # Note: the original code used [ac] here; it should be [acc].
            r2 = np.array(confusion)
            r3 = np.array(CVaR_list)
            r4 = np.array(acc_list)
            r5 = np.array(prec_list)
            r6 = np.array(rec_list)
            r7 = np.array(f1_list)
            r8 = np.array(rem)
            np.savetxt(save_path+"/acc.txt", r1) # Fix the filename to avoid overwriting.
            np.savetxt(save_path+"/confusion.txt", r2)
            np.savetxt(save_path+"/CVaR.txt", r3)
            np.savetxt(save_path+"/acc_list.txt", r4)
            np.savetxt(save_path+"/prec_list.txt", r5)
            np.savetxt(save_path+"/rec_list.txt", r6)
            np.savetxt(save_path+"/f1_list.txt", r7)
            np.savetxt(save_path+"/rem.txt", r8)
        else:
            raise ValueError('please input the test_path')
        return
    print("Start training.")
    total_steps = args.train_steps
    monitor = TrainingEfficiencyMonitor(
        experiment_name="exp1",
        method_name=args.method,
        output_dir=f"./{backbone}_efficiency_results_{DATASET}"
        ).start()
    std = 1.
    for epoch in range(1, args.epochs_num + 1):
        random.shuffle(trainset)
        src = torch.LongTensor([example[0] for example in trainset])
        tgt = torch.LongTensor([example[1] for example in trainset])
        seg = torch.LongTensor([example[2] for example in trainset])

        if args.soft_targets:
            soft_tgt = torch.FloatTensor([example[3] for example in trainset])
        else:
            soft_tgt = None
        model.train()

        # Compute the number of completed steps before the current epoch.
        # Compute the number of batches in the current epoch based on the specific batch_loader batch size.

        # Default batch size for ERM and GroupDRO.
        batches_per_epoch_default = instances_num // batch_size + (1 if instances_num % batch_size > 0 else 0)

        # Risk-optimization batch size for MC-CVaR, HISS-no-penalty, and HISS.
        ro_batch_size = robust_candidate_batch_size
        batches_per_epoch_ro = instances_num // ro_batch_size + (1 if instances_num % ro_batch_size > 0 else 0)


        if args.method == 'ERM' or args.method == 'GroupDRO' or args.method == 'TDRO' or args.method == 'Focal':
            current_batches_per_epoch = batches_per_epoch_default
            steps_in_prev_epochs = (epoch - 1) * current_batches_per_epoch
        elif args.method == 'HISS' or args.method == 'HISS-no-penalty':
            # HISS warm-up: epoch 1 uses standard batch size B, then switches to candidate-pool batch size.
            if epoch == 1:
                current_batches_per_epoch = batches_per_epoch_default
                steps_in_prev_epochs = 0
            else:
                current_batches_per_epoch = batches_per_epoch_ro
                steps_in_prev_epochs = batches_per_epoch_default + (epoch - 2) * batches_per_epoch_ro
        else: # MC-CVaR or robust subset methods
            current_batches_per_epoch = batches_per_epoch_ro
            steps_in_prev_epochs = (epoch - 1) * current_batches_per_epoch

        # ----------------------------------------------------

        if args.method == 'ERM':
            for i, (src_batch, tgt_batch, seg_batch, soft_tgt_batch) in enumerate(batch_loader(batch_size, src, tgt, seg, soft_tgt)):
                start_time = time.perf_counter()
                loss,_ = train_model(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch, soft_tgt_batch)
                total_loss += loss.item()
                end_time = time.perf_counter()
                total_time += end_time-start_time
                if (i + 1) % args.report_steps == 0:

                    # Compute and print timing information for ERM.
                    current_global_step = steps_in_prev_epochs + i + 1
                    avg_time_per_step = total_time / args.report_steps
                    remaining_steps = total_steps - current_global_step
                    estimated_time_remaining_seconds = max(0, remaining_steps) * avg_time_per_step
                    estimated_time_remaining = str(timedelta(seconds=estimated_time_remaining_seconds)).split('.')[0]

                    print("Epoch id: {}, Training steps: {}, Avg loss: {:.3f}, Avg time: {:.6f}s, Remaining: {}".\
                        format(epoch, i + 1, total_loss / args.report_steps, avg_time_per_step, estimated_time_remaining))

                    avg_loss = total_loss / args.report_steps
                    avg_time = total_time / args.report_steps
                    with open(f"{save_path}/logs/{args.method}_avg_loss.txt", "a") as f1:
                        f1.write(f"{avg_loss}\n")
                    with open(f"{save_path}/logs/{args.method}_time.txt", "a") as f3:
                        f3.write(f"{avg_time}\n")
                    total_loss = 0.0
                    total_time = 0.0 # Reset the timer.
                    # ----------------------------------------------------

        elif args.method == 'Focal':
            for i, (src_batch, tgt_batch, seg_batch, soft_tgt_batch) in enumerate(batch_loader(batch_size, src, tgt, seg, soft_tgt)):
                start_time = time.perf_counter()
                loss,_ = train_model(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch, soft_tgt_batch)
                total_loss += loss.item()
                end_time = time.perf_counter()
                total_time += end_time-start_time
                if (i + 1) % args.report_steps == 0:

                    current_global_step = steps_in_prev_epochs + i + 1
                    avg_time_per_step = total_time / args.report_steps
                    remaining_steps = total_steps - current_global_step
                    estimated_time_remaining_seconds = max(0, remaining_steps) * avg_time_per_step
                    estimated_time_remaining = str(timedelta(seconds=estimated_time_remaining_seconds)).split('.')[0]

                    print("Epoch id: {}, Training steps: {}, Avg loss: {:.3f}, Avg time: {:.6f}s, Remaining: {}".\
                        format(epoch, i + 1, total_loss / args.report_steps, avg_time_per_step, estimated_time_remaining))

                    avg_loss = total_loss / args.report_steps
                    avg_time = total_time / args.report_steps
                    with open(f"{save_path}/logs/{args.method}_avg_loss.txt", "a") as f1:
                        f1.write(f"{avg_loss}\n")
                    with open(f"{save_path}/logs/{args.method}_time.txt", "a") as f3:
                        f3.write(f"{avg_time}\n")
                    total_loss = 0.0
                    total_time = 0.0

        elif args.method == 'MC-CVaR':
            for i, (src_batch, tgt_batch, seg_batch, soft_tgt_batch) in enumerate(batch_loader(int(batch_size/(1-args.CVaR_alpha)), src, tgt, seg, soft_tgt)):
                start_time = time.perf_counter()
                with torch.no_grad():
                    _,loss_ = train_model(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch, soft_tgt_batch,inference=True)
                _,index = torch.topk(loss_.squeeze(),k=int((1-args.CVaR_alpha)*len(tgt_batch)))
                src_batch = src_batch.to(args.device)
                tgt_batch = tgt_batch.to(args.device)
                seg_batch = seg_batch.to(args.device)
                if soft_tgt_batch is not None:
                    soft_tgt_batch = soft_tgt_batch.to(args.device)
                src_batch = src_batch[index]
                tgt_batch = tgt_batch[index]
                seg_batch = seg_batch[index]
                if soft_tgt_batch != None:
                    soft_tgt_batch = soft_tgt_batch[index]
                loss,loss_ = train_model(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch, soft_tgt_batch)
                total_loss += loss.item()
                end_time = time.perf_counter()
                total_time += end_time-start_time
                if (i + 1) % args.report_steps == 0:

                    # Compute and print timing information for MC-CVaR.
                    current_global_step = steps_in_prev_epochs + i + 1
                    avg_time_per_step = total_time / args.report_steps
                    remaining_steps = total_steps - current_global_step
                    estimated_time_remaining_seconds = max(0, remaining_steps) * avg_time_per_step
                    estimated_time_remaining = str(timedelta(seconds=estimated_time_remaining_seconds)).split('.')[0]

                    print("Epoch id: {}, Training steps: {}, Avg loss: {:.3f}, Avg time: {:.6f}s, Remaining: {}".\
                        format(epoch, i + 1, total_loss / args.report_steps, avg_time_per_step, estimated_time_remaining))

                    avg_loss = total_loss / args.report_steps
                    avg_time = total_time / args.report_steps
                    with open(f"{save_path}/logs/{args.method}_avg_loss.txt", "a") as f1:
                        f1.write(f"{avg_loss}\n")
                    with open(f"{save_path}/logs/{args.method}_time.txt", "a") as f3:
                        f3.write(f"{avg_time}\n")
                    total_loss = 0.0
                    total_time = 0.0 # Reset the timer.
                    # ----------------------------------------------------

        elif args.method == 'Random':
            candidate_batch_size = int(batch_size/(1-args.CVaR_alpha))
            for i, (src_batch, tgt_batch, seg_batch, soft_tgt_batch) in enumerate(batch_loader(candidate_batch_size, src, tgt, seg, soft_tgt)):
                start_time = time.perf_counter()
                random_k = min(batch_size, len(tgt_batch))
                index = torch.randperm(len(tgt_batch))[:random_k]
                src_batch = src_batch.to(args.device)
                tgt_batch = tgt_batch.to(args.device)
                seg_batch = seg_batch.to(args.device)
                if soft_tgt_batch is not None:
                    soft_tgt_batch = soft_tgt_batch.to(args.device)
                index = index.to(args.device)
                src_batch = src_batch[index]
                tgt_batch = tgt_batch[index]
                seg_batch = seg_batch[index]
                if soft_tgt_batch != None:
                    soft_tgt_batch = soft_tgt_batch[index]
                loss,loss_ = train_model(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch, soft_tgt_batch)
                total_loss += loss.item()
                end_time = time.perf_counter()
                total_time += end_time-start_time
                if (i + 1) % args.report_steps == 0:

                    current_global_step = steps_in_prev_epochs + i + 1
                    avg_time_per_step = total_time / args.report_steps
                    remaining_steps = total_steps - current_global_step
                    estimated_time_remaining_seconds = max(0, remaining_steps) * avg_time_per_step
                    estimated_time_remaining = str(timedelta(seconds=estimated_time_remaining_seconds)).split('.')[0]

                    print("Epoch id: {}, Training steps: {}, Avg loss: {:.3f}, Avg time: {:.6f}s, Remaining: {}".\
                        format(epoch, i + 1, total_loss / args.report_steps, avg_time_per_step, estimated_time_remaining))

                    avg_loss = total_loss / args.report_steps
                    avg_time = total_time / args.report_steps
                    with open(f"{save_path}/logs/{args.method}_avg_loss.txt", "a") as f1:
                        f1.write(f"{avg_loss}\n")
                    with open(f"{save_path}/logs/{args.method}_time.txt", "a") as f3:
                        f3.write(f"{avg_time}\n")
                    total_loss = 0.0
                    total_time = 0.0

        elif args.method == 'GroupDRO':
            for i, (src_batch, tgt_batch, seg_batch, soft_tgt_batch) in enumerate(batch_loader(batch_size, src, tgt, seg, soft_tgt)):
                start_time = time.perf_counter()
                model.zero_grad()

                src_batch = src_batch.to(args.device)
                tgt_batch = tgt_batch.to(args.device)
                seg_batch = seg_batch.to(args.device)
                if soft_tgt_batch is not None:
                    soft_tgt_batch = soft_tgt_batch.to(args.device)

                _, loss_, _ = model(src_batch, tgt_batch, seg_batch, soft_tgt_batch)
                loss = torch.sum((1.*loss_.detach()).softmax(0)*loss_)
                if torch.cuda.device_count() > 1:
                    loss_ = loss_.view(-1)
                    weights = (loss_.detach() / args.gdro_tau).softmax(0)
                    loss = torch.sum(weights * loss_)
                if args.fp16:
                    with args.amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                optimizer.step()
                scheduler.step()

                total_loss += loss.item()
                end_time = time.perf_counter()
                total_time += end_time-start_time
                if (i + 1) % args.report_steps == 0:

                    # Compute and print timing information for GroupDRO.
                    current_global_step = steps_in_prev_epochs + i + 1
                    avg_time_per_step = total_time / args.report_steps
                    remaining_steps = total_steps - current_global_step
                    estimated_time_remaining_seconds = max(0, remaining_steps) * avg_time_per_step
                    estimated_time_remaining = str(timedelta(seconds=estimated_time_remaining_seconds)).split('.')[0]

                    print("Epoch id: {}, Training steps: {}, Avg loss: {:.3f}, Avg time: {:.6f}s, Remaining: {}".\
                        format(epoch, i + 1, total_loss / args.report_steps, avg_time_per_step, estimated_time_remaining))

                    avg_loss = total_loss / args.report_steps
                    avg_time = total_time / args.report_steps

                    with open(f"{save_path}/logs/{args.method}_avg_loss.txt", "a") as f1:
                        f1.write(f"{avg_loss}\n")
                    with open(f"{save_path}/logs/{args.method}_time.txt", "a") as f3:
                        f3.write(f"{avg_time}\n")
                    total_loss = 0.0
                    total_time = 0.0 # Reset the timer.
                    # ----------------------------------------------------
        
        elif args.method == 'TDRO':
            for i, (src_batch, tgt_batch, seg_batch, soft_tgt_batch) in enumerate(batch_loader(batch_size, src, tgt, seg, soft_tgt)):
                start_time = time.perf_counter()
                model.zero_grad()

                src_batch = src_batch.to(args.device)
                tgt_batch = tgt_batch.to(args.device)
                seg_batch = seg_batch.to(args.device)
                if soft_tgt_batch is not None:
                    soft_tgt_batch = soft_tgt_batch.to(args.device)

                _, loss_, _ = model(src_batch, tgt_batch, seg_batch, soft_tgt_batch)
                if torch.cuda.device_count() > 1:
                    loss_ = loss_.view(-1)

                tdro_alpha = get_tdro_alpha(model)
                loss = tdro_softplus_loss(
                    per_sample_loss=loss_,
                    alpha=tdro_alpha,
                    rho=args.tdro_rho,
                    lam=args.tdro_lambda
                )

                if args.fp16:
                    with args.amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                optimizer.step()
                scheduler.step()

                total_loss += loss.item()
                end_time = time.perf_counter()
                total_time += end_time - start_time
                if (i + 1) % args.report_steps == 0:

                    current_global_step = steps_in_prev_epochs + i + 1
                    avg_time_per_step = total_time / args.report_steps
                    remaining_steps = total_steps - current_global_step
                    estimated_time_remaining_seconds = max(0, remaining_steps) * avg_time_per_step
                    estimated_time_remaining = str(timedelta(seconds=estimated_time_remaining_seconds)).split('.')[0]
                    current_alpha = get_tdro_alpha(model).detach().item()

                    print("Epoch id: {}, Training steps: {}, Avg loss: {:.3f}, TDRO alpha: {:.3f}, Avg time: {:.6f}s, Remaining: {}".\
                        format(epoch, i + 1, total_loss / args.report_steps, current_alpha, avg_time_per_step, estimated_time_remaining))

                    avg_loss = total_loss / args.report_steps
                    avg_time = total_time / args.report_steps

                    with open(f"{save_path}/logs/{args.method}_avg_loss.txt", "a") as f1:
                        f1.write(f"{avg_loss}\n")
                    with open(f"{save_path}/logs/{args.method}_time.txt", "a") as f3:
                        f3.write(f"{avg_time}\n")
                    with open(f"{save_path}/logs/{args.method}_alpha.txt", "a") as f4:
                        f4.write(f"{current_alpha}\n")
                    total_loss = 0.0
                    total_time = 0.0
                    # ----------------------------------------------------

        elif args.method == 'OHTM':
            for i, (src_batch, tgt_batch, seg_batch, soft_tgt_batch) in enumerate(batch_loader(int(batch_size/(1-args.CVaR_alpha)), src, tgt, seg, soft_tgt)):
                start_time = time.perf_counter()
                with torch.no_grad():
                    src_batch = src_batch.to(args.device)
                    seg_batch = seg_batch.to(args.device)
                    tgt_batch = tgt_batch.to(args.device)
                    if soft_tgt_batch != None:
                        soft_tgt_batch = soft_tgt_batch.to(args.device)

                    if torch.cuda.device_count() > 1:
                        emb = model.module.embedding(src_batch,seg_batch)
                    else:
                        emb = model.embedding(src_batch,seg_batch)

                emb = emb.mean(dim=1)   # (64, 64, 128, 768) → (64, 128, 768)
                emb = emb[:, 0, :]      # (64, 128, 768)     → (64, 768)
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)

                n, d = emb.shape
                selected_indices = []
                mask = torch.ones(n, dtype=torch.bool, device=emb.device)
                residuals = emb.clone()
                squared_errors = (emb ** 2).sum(dim=1)  # Using the actual norm does not change the algorithmic logic, but makes the first step meaningful.

                for _ in range(batch_size):
                    curr_candidates = torch.where(mask, squared_errors, torch.tensor(-1.0, device=emb.device))
                    best_idx = torch.argmax(curr_candidates).item()

                    selected_indices.append(best_idx)
                    mask[best_idx] = False

                    e = residuals[best_idx] / torch.sqrt(squared_errors[best_idx] + 1e-9)
                    e = e.view(1, -1)

                    dots = torch.matmul(residuals, e.t()).squeeze()
                    residuals = residuals - dots.view(-1, 1) * e
                    squared_errors = torch.clamp(squared_errors - dots**2, min=0.0)

                index = selected_indices
                src_batch = src_batch[index]
                tgt_batch = tgt_batch[index]
                seg_batch = seg_batch[index]
                emb = emb[index]
                if soft_tgt_batch != None:
                    soft_tgt_batch = soft_tgt_batch[index]
                loss,loss_ = train_model(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch, soft_tgt_batch)
                total_loss += loss.item()
                end_time = time.perf_counter()
                total_time += end_time-start_time
                if (i + 1) % args.report_steps == 0:

                    current_global_step = steps_in_prev_epochs + i + 1
                    avg_time_per_step = total_time / args.report_steps
                    remaining_steps = total_steps - current_global_step
                    estimated_time_remaining_seconds = max(0, remaining_steps) * avg_time_per_step
                    estimated_time_remaining = str(timedelta(seconds=estimated_time_remaining_seconds)).split('.')[0]

                    print("Epoch id: {}, Training steps: {}, Avg loss: {:.3f}, Avg time: {:.6f}s, Remaining: {}".\
                        format(epoch, i + 1, total_loss / args.report_steps, avg_time_per_step, estimated_time_remaining))
                    avg_loss = total_loss / args.report_steps
                    avg_time = total_time / args.report_steps

                    with open(f"{save_path}/logs/{args.method}_avg_loss.txt", "a") as f1:
                        f1.write(f"{avg_loss}\n")
                    with open(f"{save_path}/logs/{args.method}_time.txt", "a") as f3:
                        f3.write(f"{avg_time}\n")
                    total_loss = 0.0
                    total_time = 0.0 # Reset the timer.

        elif args.method == 'HISS-no-penalty':
            use_warmup_epoch = (epoch <= args.warmup_erm)
            hiss_batch_size = batch_size if use_warmup_epoch else robust_candidate_batch_size
            for i, (src_batch, tgt_batch, seg_batch, soft_tgt_batch) in enumerate(batch_loader(hiss_batch_size, src, tgt, seg, soft_tgt)):
                start_time = time.perf_counter()
                if use_warmup_epoch:
                    loss, loss_ = train_model(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch, soft_tgt_batch)
                    # Update the proxy during warm-up as well.
                    src_batch = src_batch.to(args.device)
                    seg_batch = seg_batch.to(args.device)
                    tgt_batch = tgt_batch.to(args.device)
                    if soft_tgt_batch != None:
                        soft_tgt_batch = soft_tgt_batch.to(args.device)
                    if torch.cuda.device_count() > 1:
                        emb_module = model.module.embedding
                    else:
                        emb_module = model.embedding
                    was_training = emb_module.training
                    emb_module.eval()
                    with torch.no_grad():
                        emb = emb_module(src_batch, seg_batch)
                    if was_training:
                        emb_module.train()
                    rng_state_before_proxy_forward = capture_rng_state()
                    risk = riskmodel(emb, tgt_batch, soft_tgt_batch).squeeze()
                    restore_rng_state(rng_state_before_proxy_forward)
                    selected_risk = risk
                    selected_emb = emb
                    current_proxy_steps = 2  # Update the proxy for more steps during warm-up.
                else:
                    src_batch = src_batch.to(args.device)
                    seg_batch = seg_batch.to(args.device)
                    tgt_batch = tgt_batch.to(args.device)
                    if soft_tgt_batch != None:
                        soft_tgt_batch = soft_tgt_batch.to(args.device)
                    if torch.cuda.device_count() > 1:
                        emb_module = model.module.embedding
                    else:
                        emb_module = model.embedding
                    was_training = emb_module.training
                    emb_module.eval()
                    with torch.no_grad():
                        emb = emb_module(src_batch, seg_batch)
                    if was_training:
                        emb_module.train()
                    rng_state_before_proxy_forward = capture_rng_state()
                    risk = riskmodel(emb, tgt_batch, soft_tgt_batch).squeeze()
                    # =========================================================
                    # Restore the RNG state.
                    # This keeps the random state for the subsequent train_model(...) call the same as if the proxy forward pass had not run.
                    # =========================================================
                    restore_rng_state(rng_state_before_proxy_forward)

                    score = risk.detach()

                    _, index = torch.topk(
                        score.squeeze(),
                        k=int((1 - args.CVaR_alpha) * len(tgt_batch))
                    )

                    selected_risk = risk[index]
                    src_batch = src_batch[index]
                    tgt_batch = tgt_batch[index]
                    seg_batch = seg_batch[index]
                    selected_emb = emb[index]

                    if soft_tgt_batch != None:
                        soft_tgt_batch = soft_tgt_batch[index]

                    loss, loss_ = train_model(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch, soft_tgt_batch)

                    pred_np = selected_risk.detach().cpu().numpy()
                    true_np = loss_.detach().cpu().numpy()

                    pearson_dict = compute_pearson_corr(pred_np, true_np)
                    spearman_dict = compute_spearman_corr(pred_np, true_np)
                    k_eff = max(1, int(0.5 * len(pred_np)))  # Use the top 50% as the effective k value.
                    prec_k = precision_at_k(pred_np, true_np, k=k_eff)

                    pearson_corr = torch.tensor(
                        pearson_dict["pearson_r"],
                        device=loss_.device,
                        dtype=loss_.dtype
                    )
                    spearman_corr = torch.tensor(
                        spearman_dict["spearman_rho"],
                        device=loss_.device,
                        dtype=loss_.dtype
                    )
                    total_pearson_corr += pearson_corr.item()
                    total_spearman_corr += spearman_corr.item()
                    total_prec_k += prec_k
                    metric_count += 1

                    # ---- Update the moving window. ----
                    proxy_spearman_history.append(float(spearman_corr.item()))

                    # ---- Determine proxy_steps for the current iteration. ----
                    if len(proxy_spearman_history) < args.proxy_schedule_warmup:
                        current_proxy_steps = args.proxy_steps_high
                    else:
                        avg_recent_spearman = get_spearman_moving_average(proxy_spearman_history)
                        current_proxy_steps = schedule_proxy_steps(avg_recent_spearman, args)
                

                # =========================================================
                # Isolate the RNG during the online proxy update as well.
                # Otherwise, dropout inside the proxy update will still advance the random state.
                # =========================================================
                proxy_target = loss_.detach()
                for _ in range(current_proxy_steps):
                    rng_state_before_proxy_update = capture_rng_state()
                    optim.zero_grad()
                    risk_pred = riskmodel(selected_emb, tgt_batch, soft_tgt_batch).squeeze()
                    riskloss = pairwise_ranking_loss(
                        pred_scores=risk_pred,
                        true_scores=proxy_target,
                        max_pairs=None,
                        min_delta=args.min_delta,
                    )
                    riskloss.backward()
                    optim.step()

                    restore_rng_state(rng_state_before_proxy_update)

                total_loss += loss.item()
                end_time = time.perf_counter()
                total_time += end_time - start_time

                if (i + 1) % args.report_steps == 0:

                    current_global_step = steps_in_prev_epochs + i + 1
                    avg_time_per_step = total_time / args.report_steps
                    remaining_steps = total_steps - current_global_step
                    estimated_time_remaining_seconds = max(0, remaining_steps) * avg_time_per_step
                    estimated_time_remaining = str(timedelta(seconds=estimated_time_remaining_seconds)).split('.')[0]

                    if use_warmup_epoch:
                        print("Epoch id: {}, Training steps: {}, Avg loss: {:.3f}, Avg time: {:.6f}s, Remaining: {}".format(
                            epoch, i + 1, total_loss / args.report_steps, avg_time_per_step, estimated_time_remaining
                        ))
                    else:
                        avg_pearson_corr = total_pearson_corr / metric_count if metric_count > 0 else 0.0
                        avg_spearman_corr = total_spearman_corr / metric_count if metric_count > 0 else 0.0
                        avg_prec_k = total_prec_k / metric_count if metric_count > 0 else 0.0
                        print(
                                "Epoch id: {}, Training steps: {}, Avg loss: {:.3f}, "
                                "Avg pearson corr: {:.3f}, Avg spearman corr: {:.3f}, Avg precision@k: {:.3f}, "
                                "Recent spearman(avg): {}, Current proxy_steps: {}, "
                                "Avg time: {:.6f}s, Remaining: {}".format(
                                    epoch,
                                    i + 1,
                                    total_loss / args.report_steps,
                                    avg_pearson_corr,
                                    avg_spearman_corr,
                                    avg_prec_k,
                                    "None" if len(proxy_spearman_history) < args.proxy_schedule_warmup else f"{avg_recent_spearman:.3f}",
                                    current_proxy_steps,
                                    avg_time_per_step,
                                    estimated_time_remaining
                                )
                            )

                        with open(f"{save_path}/logs/{args.method}_corr.txt", "a") as f2:
                            f2.write(f"{avg_pearson_corr}\t{avg_spearman_corr}\t{avg_prec_k}\n")
                        with open(f"{save_path}/logs/{args.method}_proxy_schedule.txt", "a") as f:
                            f.write(
                                f"{epoch}\t{i+1}\t"
                                f"{'None' if len(proxy_spearman_history) < args.proxy_schedule_warmup else avg_recent_spearman}\t"
                                f"{current_proxy_steps}\n"
                            )

                    avg_loss = total_loss / args.report_steps
                    avg_time = total_time / args.report_steps

                    with open(f"{save_path}/logs/{args.method}_avg_loss.txt", "a") as f1:
                        f1.write(f"{avg_loss}\n")
                    with open(f"{save_path}/logs/{args.method}_time.txt", "a") as f3:
                        f3.write(f"{avg_time}\n")

                    total_loss = 0.0
                    total_pearson_corr = 0.0
                    total_spearman_corr = 0.0
                    total_prec_k = 0.0
                    total_time = 0.0
                    metric_count = 0
                    avg_recent_spearman = None
                    # ----------------------------------------------------

        elif args.method == 'HISS':
            use_warmup_epoch = (epoch <= args.warmup_erm)
            hiss_batch_size = batch_size if use_warmup_epoch else robust_candidate_batch_size
            for i, (src_batch, tgt_batch, seg_batch, soft_tgt_batch) in enumerate(batch_loader(hiss_batch_size, src, tgt, seg, soft_tgt)):
                start_time = time.perf_counter()
                if not hasattr(riskmodel, "_hiss_memory_rounds"):
                    memory_rounds = max(1, int(args.hiss_memory_capacity))
                    riskmodel._hiss_memory_rounds = deque(maxlen=memory_rounds)
                if use_warmup_epoch:
                    loss, loss_ = train_model(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch, soft_tgt_batch)
                    # Update the proxy during warm-up as well.
                    src_batch = src_batch.to(args.device)
                    seg_batch = seg_batch.to(args.device)
                    tgt_batch = tgt_batch.to(args.device)
                    if soft_tgt_batch != None:
                        soft_tgt_batch = soft_tgt_batch.to(args.device)
                    if torch.cuda.device_count() > 1:
                        emb_module = model.module.embedding
                    else:
                        emb_module = model.embedding
                    was_training = emb_module.training
                    emb_module.eval()
                    with torch.no_grad():
                        emb = emb_module(src_batch, seg_batch)
                    if was_training:
                        emb_module.train()
                    rng_state_before_proxy_forward = capture_rng_state()
                    risk = riskmodel(emb, tgt_batch, soft_tgt_batch).squeeze()
                    restore_rng_state(rng_state_before_proxy_forward)
                    selected_risk = risk
                    selected_emb = emb
                    current_proxy_steps = 2  # Update the proxy for more steps during warm-up.
                else:
                    src_batch = src_batch.to(args.device)
                    seg_batch = seg_batch.to(args.device)
                    tgt_batch = tgt_batch.to(args.device)
                    if soft_tgt_batch != None:
                        soft_tgt_batch = soft_tgt_batch.to(args.device)
                    if torch.cuda.device_count() > 1:
                        emb_module = model.module.embedding
                    else:
                        emb_module = model.embedding
                    was_training = emb_module.training
                    emb_module.eval()
                    with torch.no_grad():
                        emb = emb_module(src_batch, seg_batch)
                    if was_training:
                        emb_module.train()
                    rng_state_before_proxy_forward = capture_rng_state()
                    risk = riskmodel(emb, tgt_batch, soft_tgt_batch).squeeze()
                    # =========================================================
                    # Restore the RNG state.
                    # This keeps the random state for the subsequent train_model(...) call the same as if the proxy forward pass had not run.
                    # =========================================================
                    restore_rng_state(rng_state_before_proxy_forward)

                    selector_embeddings = pool_selector_embeddings(emb.detach())
                    sigma = compute_rbf_bandwidth(selector_embeddings, min_sigma=args.hiss_min_sigma)

                    if len(riskmodel._hiss_memory_rounds) == 0:
                        novelty_penalty = torch.zeros(
                            selector_embeddings.size(0),
                            device=selector_embeddings.device,
                            dtype=selector_embeddings.dtype
                        )
                    else:
                        memory_embeddings = torch.cat(list(riskmodel._hiss_memory_rounds), dim=0).to(selector_embeddings.device)
                        novelty_penalty = compute_rbf_novelty_penalty(
                            selector_embeddings,
                            memory_embeddings,
                            sigma,
                            reduction=args.hiss_rbf,
                            topk=args.hiss_topk,
                            gamma=args.hiss_gamma,
                            tau=args.hiss_tau
                        )

                    lambda_value = float(args.hiss_lambda_base)

                    risk_min = risk.detach().min()
                    risk_max = risk.detach().max()
                    denom = (risk_max - risk_min).clamp(min=1e-8)
                    risk_norm = (risk.detach() - risk_min) / denom
                    score = risk_norm - lambda_value * novelty_penalty

                    _, index = torch.topk(
                        score.squeeze(),
                        k=int((1 - args.CVaR_alpha) * len(tgt_batch))
                    )

                    selected_risk = risk[index]
                    src_batch = src_batch[index]
                    tgt_batch = tgt_batch[index]
                    seg_batch = seg_batch[index]
                    selected_emb = emb[index]
                    selected_selector_embeddings = selector_embeddings[index]

                    if soft_tgt_batch != None:
                        soft_tgt_batch = soft_tgt_batch[index]

                    loss, loss_ = train_model(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch, soft_tgt_batch)

                    riskmodel._hiss_memory_rounds.append(selected_selector_embeddings.detach().cpu())

                    pred_np = selected_risk.detach().cpu().numpy()
                    true_np = loss_.detach().cpu().numpy()

                    pearson_dict = compute_pearson_corr(pred_np, true_np)
                    spearman_dict = compute_spearman_corr(pred_np, true_np)
                    k_eff = max(1, int(0.5 * len(pred_np)))  # Use the top 50% as the effective k value.
                    prec_k = precision_at_k(pred_np, true_np, k=k_eff)

                    pearson_corr = torch.tensor(
                        pearson_dict["pearson_r"],
                        device=loss_.device,
                        dtype=loss_.dtype
                    )
                    spearman_corr = torch.tensor(
                        spearman_dict["spearman_rho"],
                        device=loss_.device,
                        dtype=loss_.dtype
                    )
                    total_pearson_corr += pearson_corr.item()
                    total_spearman_corr += spearman_corr.item()
                    total_prec_k += prec_k
                    metric_count += 1

                    # ---- Update the moving window. ----
                    proxy_spearman_history.append(float(spearman_corr.item()))

                    # ---- Determine proxy_steps for the current iteration. ----
                    if len(proxy_spearman_history) < args.proxy_schedule_warmup:
                        current_proxy_steps = args.proxy_steps_high
                    else:
                        avg_recent_spearman = get_spearman_moving_average(proxy_spearman_history)
                        current_proxy_steps = schedule_proxy_steps(avg_recent_spearman, args)

                # =========================================================
                # Isolate the RNG during the online proxy update as well.
                # Otherwise, dropout inside the proxy update will still advance the random state.
                # =========================================================
                proxy_target = loss_.detach()
                for _ in range(current_proxy_steps):
                    rng_state_before_proxy_update = capture_rng_state()
                    optim.zero_grad()
                    risk_pred = riskmodel(selected_emb, tgt_batch, soft_tgt_batch).squeeze()
                    riskloss = pairwise_ranking_loss(
                        pred_scores=risk_pred,
                        true_scores=proxy_target,
                        max_pairs=None,
                        min_delta=args.min_delta,
                    )
                    riskloss.backward()
                    optim.step()

                    restore_rng_state(rng_state_before_proxy_update)

                total_loss += loss.item()
                end_time = time.perf_counter()
                total_time += end_time - start_time

                if (i + 1) % args.report_steps == 0:

                    current_global_step = steps_in_prev_epochs + i + 1
                    avg_time_per_step = total_time / args.report_steps
                    remaining_steps = total_steps - current_global_step
                    estimated_time_remaining_seconds = max(0, remaining_steps) * avg_time_per_step
                    estimated_time_remaining = str(timedelta(seconds=estimated_time_remaining_seconds)).split('.')[0]

                    if use_warmup_epoch:
                        print("Epoch id: {}, Training steps: {}, Avg loss: {:.3f}, Avg time: {:.6f}s, Remaining: {}".format(
                            epoch, i + 1, total_loss / args.report_steps, avg_time_per_step, estimated_time_remaining
                        ))
                    else:
                        avg_pearson_corr = total_pearson_corr / metric_count if metric_count > 0 else 0.0
                        avg_spearman_corr = total_spearman_corr / metric_count if metric_count > 0 else 0.0
                        avg_prec_k = total_prec_k / metric_count if metric_count > 0 else 0.0
                        print(
                                "Epoch id: {}, Training steps: {}, Avg loss: {:.3f}, "
                                "Avg pearson corr: {:.3f}, Avg spearman corr: {:.3f}, Avg precision@k: {:.3f}, "
                                "Recent spearman(avg): {}, Current proxy_steps: {}, "
                                "Avg time: {:.6f}s, Remaining: {}".format(
                                    epoch,
                                    i + 1,
                                    total_loss / args.report_steps,
                                    avg_pearson_corr,
                                    avg_spearman_corr,
                                    avg_prec_k,
                                    "None" if len(proxy_spearman_history) < args.proxy_schedule_warmup else f"{avg_recent_spearman:.3f}",
                                    current_proxy_steps,
                                    avg_time_per_step,
                                    estimated_time_remaining
                                )
                            )

                        with open(f"{save_path}/logs/{args.method}_corr.txt", "a") as f2:
                            f2.write(f"{avg_pearson_corr}\t{avg_spearman_corr}\t{avg_prec_k}\n")
                        with open(f"{save_path}/logs/{args.method}_proxy_schedule.txt", "a") as f:
                            f.write(
                                f"{epoch}\t{i+1}\t"
                                f"{'None' if len(proxy_spearman_history) < args.proxy_schedule_warmup else avg_recent_spearman}\t"
                                f"{current_proxy_steps}\n"
                            )

                    avg_loss = total_loss / args.report_steps
                    avg_time = total_time / args.report_steps

                    with open(f"{save_path}/logs/{args.method}_avg_loss.txt", "a") as f1:
                        f1.write(f"{avg_loss}\n")
                    with open(f"{save_path}/logs/{args.method}_time.txt", "a") as f3:
                        f3.write(f"{avg_time}\n")

                    total_loss = 0.0
                    total_pearson_corr = 0.0
                    total_spearman_corr = 0.0
                    total_prec_k = 0.0
                    total_time = 0.0
                    metric_count = 0
                    avg_recent_spearman = None
                    # ----------------------------------------------------

        monitor.record_checkpoint(f"epoch_{epoch}")  
        with torch.no_grad():
            result = evaluate(args, read_dataset(args, args.dev_path))   # correct / len(dataset), confusion, CVaR_list, acc_list, prec_list, rec_list, f1_list, rem
        line_str = "\t".join(map(str, result[3]))
        with open(f"{save_path}/logs/{args.method}_train_acc_list.txt", "a") as f4:
            f4.write(f"{line_str}\n")
        if result[0] > best_result:
            best_result = result[0]  # Save the best model based on average accuracy, not CVaR-conditional accuracy.
            save_model(model, f"{save_path}/{args.output_model_path}")

    # Evaluation phase.
    monitor.stop()
    if args.test_path is not None:
        print("Test set evaluation.")
        if torch.cuda.device_count() > 1:
            model.module.load_state_dict(torch.load(save_path+'/'+args.output_model_path,map_location='cuda:1'))
        else:
            model.load_state_dict(torch.load(save_path+'/'+args.output_model_path))
        with torch.no_grad():
            acc, confusion,CVaR_list,acc_list,prec_list,rec_list,f1_list,rem = evaluate(args, read_dataset(args, args.test_path), True)

        r1 = np.array([acc])  # Note: the original code used [ac] here; it should be [acc].
        r2 = np.array(confusion)
        r3 = np.array(CVaR_list)
        r4 = np.array(acc_list)
        r5 = np.array(prec_list)
        r6 = np.array(rec_list)
        r7 = np.array(f1_list)
        r8 = np.array(rem)
        np.savetxt(save_path+"/acc.txt", r1) # Fix the filename to avoid overwriting.
        np.savetxt(save_path+"/confusion.txt", r2)
        np.savetxt(save_path+"/CVaR.txt", r3)
        np.savetxt(save_path+"/acc_list.txt", r4)
        np.savetxt(save_path+"/prec_list.txt", r5)
        np.savetxt(save_path+"/rec_list.txt", r6)
        np.savetxt(save_path+"/f1_list.txt", r7)
        np.savetxt(save_path+"/rem.txt", r8)

if __name__ == "__main__":
    main()
