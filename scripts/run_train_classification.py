#!/usr/bin/env python3
"""Launch FG-CLIP classification training from YAML config."""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

import argparse
import yaml
import subprocess
import sys
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Run FG-CLIP classification training from YAML config")
    parser.add_argument("--config", type=str, default="./scripts/train_classification.yaml",
                        help="Path to YAML configuration file")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # GPU config - set environment variables
    env = os.environ.copy()
    if 'gpu' in config and 'cuda_visible_devices' in config['gpu']:
        env["CUDA_VISIBLE_DEVICES"] = config['gpu']['cuda_visible_devices']

    # Build command - use -- before training script to separate DeepSpeed args from script args
    cmd = ["deepspeed"]

    # DeepSpeed GPU include - must be before --
    if 'gpu' in config and 'deepspeed_include' in config['gpu']:
        cmd += [f"--include={config['gpu']['deepspeed_include']}"]

    cmd += ["--", "fgclip/train/train_fgclip.py"]

    # Model config
    cmd += [f"--base_model={config['model']['base_model']}"]
    cmd += [f"--model_name_or_path={config['model']['model_name_or_path']}"]
    cmd += [f"--from_openai={str(config['model']['from_openai']).lower()}"]

    # Data config - join list to space-separated string
    train_jsonfiles = " ".join(config['data']['train_jsonfiles'])
    cmd += [f"--data_path={train_jsonfiles}"]
    if 'val_jsonfiles' in config['data'] and config['data']['val_jsonfiles']:
        val_jsonfiles = " ".join(config['data']['val_jsonfiles'])
        cmd += [f"--eval_data_path={val_jsonfiles}"]
    cmd += [f"--image_folder={config['data']['image_root']}"]

    # Classifier config
    cmd += [f"--classifier_yaml={config['classifier']['yaml_path']}"]

    # Training config
    cmd += [f"--output_dir={config['training']['output_dir']}"]
    cmd += [f"--num_train_epochs={config['training']['num_epochs']}"]
    cmd += [f"--per_device_train_batch_size={config['training']['per_device_train_batch_size']}"]
    cmd += [f"--per_device_eval_batch_size={config['training']['per_device_eval_batch_size']}"]
    cmd += [f"--gradient_accumulation_steps={config['training']['gradient_accumulation_steps']}"]
    cmd += [f"--learning_rate={config['training']['learning_rate']}"]
    cmd += [f"--weight_decay={config['training']['weight_decay']}"]
    cmd += [f"--warmup_steps={config['training']['warmup_steps']}"]
    cmd += [f"--lr_scheduler_type={config['training']['lr_scheduler_type']}"]
    cmd += [f"--evaluation_strategy={config['training']['evaluation_strategy']}"]
    cmd += [f"--save_strategy={config['training']['save_strategy']}"]
    cmd += [f"--save_steps={config['training']['save_steps']}"]
    cmd += [f"--save_total_limit={config['training']['save_total_limit']}"]
    cmd += [f"--logging_steps={config['training']['logging_steps']}"]
    cmd += [f"--dataloader_num_workers={config['training']['dataloader_num_workers']}"]
    cmd += [f"--dataloader_pin_memory={str(config['training']['dataloader_pin_memory']).lower()}"]
    cmd += [f"--bf16={str(config['training']['bf16']).lower()}"]
    cmd += [f"--disable_tqdm=true"]
    cmd += [f"--tf32={str(config['training']['tf32']).lower()}"]
    cmd += [f"--gradient_checkpointing={str(config['training']['gradient_checkpointing']).lower()}"]

    # Freeze config
    cmd += [f"--freeze_vision={str(config['freeze']['vision']).lower()}"]
    cmd += [f"--freeze_text={str(config['freeze']['text']).lower()}"]

    # Classification config
    cmd += [f"--use_classification={str(config['classification']['use_classification']).lower()}"]
    cmd += [f"--classification_loss_weight={config['classification']['classification_loss_weight']}"]

    # Bbox config
    cmd += [f"--add_box_loss={str(config['bbox']['add_box_loss']).lower()}"]
    cmd += [f"--use_hard_neg={str(config['bbox']['use_hard_neg']).lower()}"]

    # Model architecture config
    cmd += [f"--base_image_size={config['model_arch']['base_image_size']}"]
    cmd += [f"--base_seq_length={config['model_arch']['base_seq_length']}"]
    cmd += [f"--max_seq_length={config['model_arch']['max_seq_length']}"]
    cmd += [f"--save_safetensors={str(config['model_arch']['save_safetensors']).lower()}"]

    # Other config
    cmd += [f"--lazy_preprocess={str(config['other']['lazy_preprocess']).lower()}"]
    cmd += [f"--report_to={config['other']['report_to']}"]
    cmd += [f"--skip_invalid_labels={str(config['other']['skip_invalid_labels']).lower()}"]
    cmd += [f"--print_eval_metrics={str(config['other'].get('print_eval_metrics', True)).lower()}"]
    cmd += [f"--log_eval_metrics={str(config['other'].get('log_eval_metrics', False)).lower()}"]

    # DeepSpeed config
    cmd += [f"--deepspeed={config['deepspeed']['config_path']}"]

    # GPU world size
    if 'gpu' in config and 'train_use_word_size' in config['gpu']:
        cmd += [f"--train_use_word_size={config['gpu']['train_use_word_size']}"]
    elif 'deepspeed' in config and 'train_use_word_size' in config['deepspeed']:
        cmd += [f"--train_use_word_size={config['deepspeed']['train_use_word_size']}"]

    # Print command
    print("=" * 80)
    print("Running command:")
    print(" ".join(cmd))
    print("=" * 80)

    # Run command with PYTHONPATH
    env["PYTHONPATH"] = f"{os.getcwd()}:{env.get('PYTHONPATH', '')}"
    result = subprocess.run(cmd, env=env)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()