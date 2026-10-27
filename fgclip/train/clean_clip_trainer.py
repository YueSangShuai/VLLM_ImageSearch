import copy
import torch
import numpy as np
from sklearn.metrics import accuracy_score

from torch.utils.data import Sampler

from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
)
from transformers.trainer_callback import TrainerCallback
from typing import Dict, Optional
from tqdm import tqdm
import torch.nn.functional as F


def compute_ma_metric(labels: np.ndarray, predictions: np.ndarray, valid_mask: np.ndarray) -> float:
    """Compute mean Accuracy (mA) metric for attribute recognition.

    mA = (1/N) * sum over attributes (TP + TN) / (TP + TN + FP + FN)
    """
    num_attrs = labels.shape[1]
    acc_per_attr = []

    for i in range(num_attrs):
        valid_idx = valid_mask[:, i] == 1
        if valid_idx.sum() == 0:
            continue

        y_true = labels[valid_idx, i]
        y_pred = predictions[valid_idx, i]

        tp = ((y_true == 1) & (y_pred == 1)).sum()
        tn = ((y_true == 0) & (y_pred == 0)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        fn = ((y_true == 1) & (y_pred == 0)).sum()

        acc = (tp + tn) / (tp + tn + fp + fn + 1e-6)
        acc_per_attr.append(acc)

    return np.mean(acc_per_attr) if acc_per_attr else 0.0


class CustomProgressCallback(TrainerCallback):
    """自定义进度条回调，按照 ultralytics 风格动态显示所有损失"""
    def __init__(self):
        super().__init__()
        self.loss_names = []  # 动态损失名
        self.loss_values = []  # 动态损失值
        self._header_printed = False

    def _get_gpu_mem(self):
        """获取 GPU 内存使用"""
        if torch.cuda.is_available():
            return f"{torch.cuda.memory_reserved() / 1E9:.2f}G"
        return "0.00G"

    def _print_header(self):
        """打印表头"""
        # Epoch    GPU_mem    clsloss   dist_fea   dist_log  Instances       Size
        header = f"{'Epoch':>11}    {'GPU_mem':>8}"
        for name in self.loss_names:
            header += f"  {name:>10}"
        header += f"  {'Instances':>10}  {'Size':>6}"
        print(header)
        self._header_printed = True

    def on_train_begin(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self._total_epochs = int(args.num_train_epochs)
            self._steps_per_epoch = state.max_steps // self._total_epochs if self._total_epochs > 0 else state.max_steps
            self._header_printed = False
            self.loss_names = []
            self.loss_values = []
            self.eval_bar = None
            print(f"[CALLBACK] on_train_begin: max_steps={state.max_steps}, steps_per_epoch={self._steps_per_epoch}")

            # 设置初始描述，等待第一个 step 后再更新
            init_desc = f"  1/{self._total_epochs}        0.00G      0.0000      0.0000"
            self.training_bar = tqdm(
                total=self._steps_per_epoch,
                desc=init_desc,
                dynamic_ncols=True,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
            )

    def on_epoch_begin(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            if self.training_bar is not None:
                self.training_bar.close()
            self._header_printed = False
            # 设置初始描述，等待第一个 step 后再更新
            epoch_num = int(state.epoch) + 1 if hasattr(state, 'epoch') else 1
            init_desc = f"  {epoch_num}/{self._total_epochs}        0.00G      0.0000      0.0000"
            self.training_bar = tqdm(
                total=self._steps_per_epoch,
                desc=init_desc,
                dynamic_ncols=True,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
            )

    def on_epoch_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero and self.training_bar is not None:
            # Print final stats for this epoch
            self.training_bar.close()
            self.training_bar = None

    def on_evaluate(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            # 验证完成后关闭进度条
            if hasattr(self, 'eval_bar') and self.eval_bar is not None:
                self.eval_bar.close()
                self.eval_bar = None

    def on_prediction_step(self, args, state, control, **kwargs):
        if state.is_world_process_zero and hasattr(self, 'eval_bar') and self.eval_bar is not None:
            self.eval_bar.update()

    def on_step_end(self, args, state, control, **kwargs):
        pass  # 进度更新已在 compute_loss 中通过 _update_progress_callback 完成

    def on_train_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero and self.training_bar is not None:
            self.training_bar.close()
            self.training_bar = None


class CLIPTrainer(Trainer):

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        return super()._get_train_sampler()

    def create_optimizer(self):
        """Setup the optimizer."""
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]

            if self.args.text_model_lr is not None:
                text_model_parameters = [name for name, _ in opt_model.named_parameters() if "text_model" in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in text_model_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in text_model_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in text_model_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.text_model_lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in text_model_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.text_model_lr,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer

    def _nested_gather(self, tensors, name=None):
        """Override to handle list/tuple returns from distributed_concat."""
        result = super()._nested_gather(tensors, name)
        if isinstance(result, (list, tuple)) and len(result) == 1:
            return result[0]
        return result

    def _update_progress_callback(self):
        """更新 CustomProgressCallback 中的损失值和进度条"""
        # 只在 rank 0 执行
        if not (self.state.is_world_process_zero if hasattr(self.state, 'is_world_process_zero') else True):
            return

        for callback in self.callback_handler.callbacks:
            if isinstance(callback, CustomProgressCallback):
                callback.loss_values = [self._clip_loss.item(), self._classification_loss.item()]
                if not callback._header_printed:
                    callback.loss_names = ['clip_loss', 'classification_loss']
                    header = f"\n{'Epoch':>11}    {'GPU_mem':>8}"
                    for name in callback.loss_names:
                        header += f"  {name:>10}"
                    print(header)
                    callback._header_printed = True

                if hasattr(callback, 'training_bar') and callback.training_bar is not None:
                    state = self.state
                    epoch_str = f"{int(state.epoch) + 1}/{callback._total_epochs}"
                    mem_str = callback._get_gpu_mem()
                    desc = f"{epoch_str:>11}    {mem_str:>8}"
                    for val in callback.loss_values:
                        desc += f"  {val:>10.4f}"
                    callback.training_bar.set_description(desc)
                    callback.training_bar.update()
                break

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Override compute_loss to handle classification loss properly."""
        labels = inputs.get("labels")
        outputs = model(**inputs)

        # Handle CLIPOutput or similar structured outputs
        if hasattr(outputs, 'loss') and outputs.loss is not None:
            clip_loss = outputs.loss
        elif isinstance(outputs, dict) and 'loss' in outputs:
            clip_loss = outputs['loss']
        else:
            clip_loss = outputs[0] if isinstance(outputs, (tuple, list)) else outputs

        # Handle NaN in clip_loss
        if torch.isnan(clip_loss) or torch.isinf(clip_loss):
            clip_loss = torch.tensor(0.0, device=next(model.parameters()).device)

        self._clip_loss = clip_loss.detach()
        self._classification_loss = torch.tensor(0.0, device=self._clip_loss.device)

        # Add classification loss if present
        classification_logits = getattr(outputs, 'classification_logits', None) or outputs.get('classification_logits')
        classification_labels = inputs.get('classification_labels', {})
        loss = clip_loss
        if classification_logits and classification_labels:
            cls_loss = self.compute_classification_loss(
                classification_logits,
                classification_labels,
                inputs.get('classification_valid_mask', {})
            )
            # Handle NaN in classification loss
            if torch.isnan(cls_loss) or torch.isinf(cls_loss):
                cls_loss = torch.tensor(0.0, device=next(model.parameters()).device)
            self._classification_loss = cls_loss.detach()
            loss = clip_loss + cls_loss

        # Ensure loss is a tensor with scalar value
        if not isinstance(loss, torch.Tensor):
            loss = torch.tensor(loss, device=list(model.parameters())[0].device)
        loss = loss.mean()

        # 更新进度回调中的损失值
        self._update_progress_callback()

        return (loss, outputs, num_items_in_batch) if return_outputs else loss

    def _save_checkpoint(self, model, trial, metrics=None):
        super(CLIPTrainer, self)._save_checkpoint(model, trial)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        super(CLIPTrainer, self)._save(output_dir, state_dict)

    def compute_classification_loss(self, classification_logits: Dict, classification_labels: Dict,
                                     classification_valid_mask: Dict) -> torch.Tensor:
        """Compute multi-label classification loss, skipping invalid labels (-1)."""
        total_loss = None
        total_valid = 0

        for attr_name, logits in classification_logits.items():
            if attr_name not in classification_labels:
                continue

            labels = classification_labels[attr_name]
            valid_mask = classification_valid_mask[attr_name]

            # Handle different output shapes: (batch, 2) for 2-class or (batch, 1) for binary
            if logits.dim() == 2 and logits.shape[-1] == 2:
                # Two-class classification: use focal loss
                log_probs = F.log_softmax(logits, dim=1)
                probs = torch.exp(log_probs)
                target_log_probs = log_probs[range(logits.size(0)), labels.long()]
                target_probs = probs[range(logits.size(0)), labels.long()]
                focal_weight = (1 - target_probs) ** 2.0
                loss = focal_weight * target_log_probs
                loss = -loss  # shape (B,)
            else:
                # Binary classification: use focal loss for binary
                probs = torch.sigmoid(logits.squeeze(-1))
                loss = -((1 - probs) ** 2.0 * torch.log(probs + 1e-8) * labels +
                         probs ** 2.0 * torch.log(1 - probs + 1e-8) * (1 - labels))

            masked_loss = loss * valid_mask
            valid_count = valid_mask.sum()

            
            if valid_count > 0:
                total_loss = masked_loss.sum() if total_loss is None else total_loss + masked_loss.sum()
                total_valid += valid_count

        if total_valid > 0:
            return total_loss / total_valid
        return torch.tensor(0.0, device=next(iter(classification_logits.values())).device if classification_logits else 'cpu')

    def prediction_step(self, model, inputs, prediction_loss_only=False, ignore_keys=None):
        """Override prediction_step to support classification."""
        has_classification = 'classification_labels' in inputs and inputs['classification_labels'] is not None

        if not has_classification:
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

        # Forward pass
        with torch.no_grad():
            outputs = model(
                text_short=inputs.get('text_short'),
                text_long=inputs.get('text_long'),
                image=inputs.get('image'),
                add_box_loss=inputs.get('add_box_loss', False),
                use_hard_neg=inputs.get('use_hard_neg', False),
                classification_labels=inputs.get('classification_labels'),
                classification_valid_mask=inputs.get('classification_valid_mask'),
            )

        loss = outputs.loss
        classification_logits = outputs.classification_logits

        # 收集预测结果用于最终指标计算
        if classification_logits is not None and inputs.get('classification_labels') is not None:
            # 初始化收集列表（如果不存在）
            if not hasattr(self, '_eval_logits') or self._eval_logits is None:
                self._eval_logits = {}
                self._eval_labels = {}
                self._eval_masks = {}

            for attr_name, logits in classification_logits.items():
                if logits is not None:
                    logits_np = logits.detach().cpu()
                    if attr_name not in self._eval_logits:
                        self._eval_logits[attr_name] = []
                        self._eval_labels[attr_name] = []
                        self._eval_masks[attr_name] = []
                    self._eval_logits[attr_name].append(logits_np)

                    labels = inputs['classification_labels'].get(attr_name)
                    masks = inputs['classification_valid_mask'].get(attr_name)
                    if labels is not None:
                        self._eval_labels[attr_name].append(labels.detach().cpu())
                    if masks is not None:
                        self._eval_masks[attr_name].append(masks.detach().cpu())

            cls_loss = self.compute_classification_loss(
                classification_logits,
                inputs['classification_labels'],
                inputs['classification_valid_mask']
            )
            loss = loss + cls_loss * getattr(self.args, 'classification_loss_weight', 1.0)

        return (loss, None, None)

    def evaluation_loop(self, dataloader, description, prediction_loss_only=None, ignore_keys=None,
                       metric_key_prefix="eval"):
        """Override evaluation_loop to compute classification metrics."""
        # 创建验证进度条
        if hasattr(self, 'callback_handler'):
            for callback in self.callback_handler.callbacks:
                if isinstance(callback, CustomProgressCallback):
                    if self.state.is_world_process_zero:
                        eval_len = len(dataloader) if hasattr(dataloader, '__len__') else 0
                        callback.eval_bar = tqdm(
                            total=eval_len,
                            desc="Evaluating",
                            dynamic_ncols=True,
                            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
                        )
                    break

        output = super().evaluation_loop(dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix)

        # Compute classification metrics if available
        if hasattr(self, '_eval_logits') and self._eval_logits:
            metrics = self.compute_classification_metrics()
            output.metrics.update(metrics)
            self._eval_logits = None
            self._eval_labels = None
            self._eval_masks = None

        return output

    def compute_classification_metrics(self) -> Dict:
        """Compute per-class accuracy and mA metrics."""
        if not hasattr(self, '_eval_logits') or not self._eval_logits:
            return {}

        metrics = {}
        all_class_acc = []
        per_class_acc = {}

        for attr_name in self._eval_logits:
            logits_list = self._eval_logits[attr_name]
            labels_list = self._eval_labels.get(attr_name, [])
            masks_list = self._eval_masks.get(attr_name, [])

            if not logits_list or not labels_list:
                continue

            # 拼接所有 batch 的数据
            all_logits = torch.cat(logits_list, dim=0)  # (N, 2) for binary
            all_labels = torch.cat(labels_list, dim=0)  # (N,)
            all_masks = torch.cat(masks_list, dim=0) if masks_list else torch.ones_like(all_labels)

            # 对于二分类，取 class 1 的 logits 进行预测
            if all_logits.dim() == 2 and all_logits.shape[1] == 2:
                preds = (torch.sigmoid(all_logits[:, 1]) > 0.5).long()
            else:
                preds = (torch.sigmoid(all_logits.squeeze(-1)) > 0.5).long()

            # 只在有效样本上计算准确率
            valid_mask = all_masks == 1
            if valid_mask.sum() > 0:
                correct = (preds == all_labels) & valid_mask
                acc = correct.float().sum() / valid_mask.float().sum()
                all_class_acc.append(acc.item())
                per_class_acc[attr_name] = acc.item()
                # 只有在 log_eval_metrics 为 True 时才添加到 metrics 字典
                if self.args.log_eval_metrics:
                    metrics[f'eval_{attr_name}_acc'] = acc.item()

        # 计算 mA (mean Accuracy)
        if all_class_acc:
            metrics['eval_mA'] = sum(all_class_acc) / len(all_class_acc)

        # 打印每个类别的准确率
        if self.state.is_world_process_zero and self.args.print_eval_metrics:
            print("\n" + "=" * 60)
            print("Per-class Accuracy:")
            for attr_name, acc in per_class_acc.items():
                print(f"  {attr_name}: {acc:.4f}")
            print(f"\nmA (mean Accuracy): {metrics.get('eval_mA', 0):.4f}")
            print("=" * 60 + "\n")

        return metrics