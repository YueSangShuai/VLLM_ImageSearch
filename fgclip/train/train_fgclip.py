import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List

import torch
import random

import glob
import transformers

from torch.utils.data import Dataset
from fgclip.train.clean_clip_trainer import CLIPTrainer, CustomProgressCallback


import torch.distributed as dist
import math

import copy
import os
import json
import torch
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url
from torchvision import transforms
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision.transforms.functional import InterpolationMode
from einops import rearrange

from random import choice
from PIL import Image

import gzip
from io import BytesIO
import base64
from torch.utils.data import  IterableDataset
import random

from fgclip.model.clip_strc.fgclip import FGCLIPModel
# Load pretrained model, tokenizer, and image processor
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig, CLIPConfig
import numpy as np

from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)

import gc
import yaml

local_rank = None


def load_classifier_config(yaml_path: str) -> Dict:
    """从 YAML 加载分类器配置"""
    if yaml_path is None or not os.path.exists(yaml_path):
        return None
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config.get('classiffer_config', None)


def rank0_print(*args):
    if local_rank == 0:
        print(*args)

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    base_model: Optional[str] = field(default=None)
    download_root: Optional[str] = field(default=None)
    log_scale: float = 4.6052

@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    eval_data_path: Optional[str] = field(default=None,
                           metadata={"help": "Path to the evaluation data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'
    image_grid_pinpoints: Optional[str] = field(default=None)
    max_seq_length: int = 77*4-60
    base_seq_length: int = 77
    base_image_size: int = 224
    add_box_loss: bool = field(default=False)
    use_hard_neg: bool = field(default=False)
    classifier_yaml: Optional[str] = field(default=None, metadata={"help": "Path to classifier YAML config"})
    use_classification: bool = field(default=False)
    classification_loss_weight: float = field(default=1.0)
    skip_invalid_labels: bool = field(default=True)


    

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    from_openai: bool = field(default=False)
    train_use_word_size: int = 8
    text_model_lr: Optional[float] = None
    freeze_vision: bool = field(default=False)
    freeze_text: bool = field(default=False)
    print_eval_metrics: bool = field(default=True, metadata={"help": "Whether to print per-class evaluation metrics"})
    log_eval_metrics: bool = field(default=False, metadata={"help": "Whether to log per-class metrics to output dictionary"})


from datetime import datetime
    
def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()

    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


class LazySupervisedBboxDataset(Dataset):
    """Dataset for Stage2."""

    def __init__(self, data_path: str,
                 data_args: DataArguments,
                 img_preprocess=None,tokenizer=None):
        super(LazySupervisedBboxDataset, self).__init__()

        if data_path.endswith('.json') or data_path.endswith('.jsonl'):
            list_data_dict = json.load(open(data_path, "r", encoding="utf-8"))
        elif data_path.endswith('.txt'):
            lines = open(data_path, "r", encoding="utf-8").readlines()
            list_data_dict = []
            for line in lines:
                json_file = line.rstrip()
                list_data_dict += json.load(open(json_file, "r",encoding="utf-8"))
        else:
            json_files = glob.glob(os.path.join(data_path, '*.json'))
            list_data_dict = []
            for json_file in json_files:
                list_data_dict += json.load(open(json_file, "r",encoding="utf-8"))

            jsonl_files = glob.glob(os.path.join(data_path, '*.jsonl'))
            for jsonl_file in jsonl_files:
                list_data_dict += json.load(open(jsonl_file, "r",encoding="utf-8"))
        

        
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.max_anns = 4

        self.data_args = data_args
        self.preprocess = img_preprocess
        self.image_root = data_args.image_folder
        self.max_length = data_args.max_seq_length
        self.base_length = data_args.base_seq_length
        self.base_image_size = data_args.base_image_size
        self.add_box_loss = data_args.add_box_loss
        self.use_hard_neg = data_args.use_hard_neg

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:

        item = self.list_data_dict[i]

        caption = item["caption"]
        caption_short = "a photo of "+item["short_caption"]        

        image_path = item["f_path"]
        image_path = image_path.replace("grit-20m/data-12m/","")
        image_name = os.path.join(self.image_root,image_path)
        
        image = Image.open(image_name).convert("RGB")
        
        image = image.resize((self.base_image_size, self.base_image_size))

        image_tensor = self.preprocess.preprocess(image, return_tensors='pt')['pixel_values'][0]

        text =  torch.tensor(self.tokenizer([caption], max_length=self.max_length, padding="max_length", truncation=True).input_ids, dtype=torch.long, device=image_tensor.device)
        short_text = torch.tensor(self.tokenizer([caption_short], max_length=self.base_length, padding="max_length", truncation=True).input_ids, dtype=torch.long, device=image_tensor.device)        

        if self.add_box_loss:
            box_texts = []
            bbox_info = item["bbox_info"]

            total_num = self.max_anns
            valid_num = min(len(bbox_info), self.max_anns)
            boxes_template = torch.zeros((total_num, 4), device=image_tensor.device)
            width, height = image.size
            for i in range(total_num):
                if i<valid_num:
                    bbox_data = bbox_info[i]
                    box = bbox_data["bbox"]
                    box_caption = random.choice([bbox_data["short_expr"], bbox_data["long_expr"]])

                else:
                    box = [0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.000000000]
                    box_caption = ""
                    

                box_tensor = torch.tensor(box[:4])
                boxes_template[i] = box_tensor

                if box[0] > box[2] or box[1] > box[3]:
                    raise ValueError("Box coordinates are invalid.")

                box_text = torch.tensor(self.tokenizer([box_caption], max_length=self.base_length, padding="max_length", truncation=True).input_ids, dtype=torch.long, device=image_tensor.device)        
                box_texts.append(box_text)


            box_texts = torch.cat(box_texts,dim=0)
            bbox_num = torch.tensor([valid_num], device=image_tensor.device)
                    
        if self.use_hard_neg:
            hard_texts = []
           
            bbox_info = item["bbox_info"]

            width, height = image.size
            total_num = self.max_anns
            valid_num = min(len(bbox_info), self.max_anns)
            hard_boxes = torch.zeros((total_num, 4), device=image_tensor.device)
            valid_hard = 0
            for i in range(total_num):
                if i<valid_num:
                    bbox_data = bbox_info[i]
                    box = bbox_data["bbox"]
                    box_caption = bbox_data["short_expr"]
                    
                    box_tensor = torch.tensor(box[:4])

                    if box[0] > box[2] or box[1] > box[3]:
                        raise ValueError("Box coordinates are invalid.")
    
                    if bbox_data["flag_short_neg"] == 1:
                        cur_texts = [box_caption]
                        hard_negs = bbox_data["short_expr_negs"]
                        for key in hard_negs.keys():
                            cur_texts.append(hard_negs[key])
                        box_text = torch.tensor(self.tokenizer(cur_texts, max_length=self.base_length, padding="max_length", truncation=True).input_ids, dtype=torch.long, device=image_tensor.device)        
                        hard_texts.append(box_text)

                        hard_boxes[valid_hard] = box_tensor
                        valid_hard = valid_hard+1


            valid_hard = torch.tensor([valid_hard], device=image_tensor.device)

            if len(hard_texts) > 0:
                hard_texts = torch.cat(hard_texts,dim=0)
            else:
                hard_texts = None

        data_dict = {}
        data_dict['image'] = image_tensor
        data_dict['text'] = text
        data_dict['short_text'] = short_text
        data_dict['add_box_loss'] = self.add_box_loss
        data_dict['use_hard_neg'] = self.use_hard_neg

        if self.add_box_loss:
            data_dict['box_texts'] = box_texts
            data_dict['box_infos'] = boxes_template
            data_dict['box_nums'] = bbox_num
        if self.use_hard_neg:
            data_dict['hard_texts'] = hard_texts
            data_dict['hard_infos'] = hard_boxes
            data_dict['hard_nums'] = valid_hard
            
        return data_dict




@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        
        batch = {}
        images = [instance['image'] for instance in instances]
        batch['image'] = torch.stack(images)
        texts = [instance['text'] for instance in instances]
        batch['text_long'] = torch.cat(texts,dim=0)
        short_texts = [instance['short_text'] for instance in instances]
        batch['text_short'] = torch.cat(short_texts,dim=0)
        
        batch["add_box_loss"] = instances[0]["add_box_loss"]
        batch["use_hard_neg"] = instances[0]["use_hard_neg"]
        
        if batch["add_box_loss"]:
            box_texts = [instance['box_texts'] for instance in instances]
            batch['box_texts'] = torch.cat(box_texts,dim=0)
            box_infos = [instance['box_infos'] for instance in instances]
            batch['box_infos'] = torch.cat(box_infos,dim=0)
            box_nums = [instance['box_nums'] for instance in instances]
            batch['box_nums'] = torch.cat(box_nums, dim=0)
        if batch["use_hard_neg"] :
            hard_texts = []
            for instance in instances:
                if instance['hard_texts'] != None:
                    hard_texts.append(instance['hard_texts'])

            batch['hard_texts'] = torch.cat(hard_texts,dim=0)
            hard_infos = [instance['hard_infos'] for instance in instances]
            batch['hard_infos'] = torch.cat(hard_infos,dim=0)
            hard_nums = [instance['hard_nums'] for instance in instances]
            batch['hard_nums'] = torch.cat(hard_nums, dim=0)                

        return batch


class PedestrianAttributeDataset(Dataset):
    """Dataset for Pedestrian Attribute Classification with dynamic attribute loading from YAML."""

    def __init__(self, data_path: str, data_args: DataArguments,
                 img_preprocess=None, tokenizer=None, classifier_config: Dict = None):
        super(PedestrianAttributeDataset, self).__init__()

        # Get distributed training info
        if dist.is_available() and dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1

        # Load multiple JSON files
        list_data_dict = []
        # Support space-separated multiple paths
        if ' ' in data_path:
            # Split by space to support multiple json files
            json_paths = data_path.split(' ')
        else:
            json_paths = [data_path]

        for data_path_item in json_paths:
            if data_path_item.endswith('.json') or data_path_item.endswith('.jsonl'):
                list_data_dict += json.load(open(data_path_item, "r", encoding="utf-8"))
            elif data_path_item.endswith('.txt'):
                lines = open(data_path_item, "r", encoding="utf-8").readlines()
                for line in lines:
                    json_file = line.rstrip()
                    list_data_dict += json.load(open(json_file, "r", encoding="utf-8"))
            else:
                json_files = glob.glob(os.path.join(data_path_item, '*.json'))
                for json_file in json_files:
                    list_data_dict += json.load(open(json_file, "r", encoding="utf-8"))

                jsonl_files = glob.glob(os.path.join(data_path_item, '*.jsonl'))
                for jsonl_file in jsonl_files:
                    list_data_dict += json.load(open(jsonl_file, "r", encoding="utf-8"))

        # Distribute data across ranks (distributed data loading)
        total_samples = len(list_data_dict)
        if self.world_size > 1:
            # Shuffle data first, then split
            np.random.seed(42)
            indices = np.arange(total_samples)
            np.random.shuffle(indices)
            list_data_dict = [list_data_dict[i] for i in indices]

            # Split data into shards for each rank
            samples_per_rank = math.ceil(total_samples / self.world_size)
            start_idx = self.rank * samples_per_rank
            end_idx = min(start_idx + samples_per_rank, total_samples)
            list_data_dict = list_data_dict[start_idx:end_idx]

        
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.preprocess = img_preprocess
        self.max_length = data_args.max_seq_length
        self.base_length = data_args.base_seq_length
        self.base_image_size = data_args.base_image_size
        self.classifier_config = classifier_config
        self.use_classification = data_args.use_classification
        self.skip_invalid = data_args.skip_invalid_labels

        # Parse classifier config to get attribute names
        self.classification_attr_names = []
        if classifier_config is not None and classifier_config.get('open', False):
            classes = classifier_config.get('classes', {})
            self.classification_attr_names = list(classes.keys())

        # Filter out samples without any valid attributes
        if self.skip_invalid and self.classification_attr_names:
            original_count = len(self.list_data_dict)
            self.list_data_dict = [
                item for item in self.list_data_dict
                if any(item.get(attr, -1) != -1 for attr in self.classification_attr_names)
            ]
            print(f"Filtered {original_count - len(self.list_data_dict)} samples without valid attributes, remaining: {len(self.list_data_dict)}")

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        item = self.list_data_dict[i]

        # Debug: print first few items (flush to ensure output)
        
        # Load image
        image_path = item["image_path"]
        if not os.path.isabs(image_path):
            image_path = os.path.join(self.data_args.image_folder, image_path)
        image = Image.open(image_path).convert("RGB")
        image = image.resize((self.base_image_size, self.base_image_size))
        image_tensor = self.preprocess.preprocess(image, return_tensors='pt')['pixel_values'][0]

        # Load caption from text_path
        text_path = item["text_path"]
        with open(text_path, 'r', encoding='utf-8') as f:
            caption = f.read().strip()

        short_caption = caption[:100] if len(caption) > 100 else caption
        caption_with_prefix = "a photo of " + short_caption

        text = torch.tensor(
            self.tokenizer([caption], max_length=self.max_length, padding="max_length", truncation=True).input_ids,
            dtype=torch.long, device=image_tensor.device
        )
        short_text = torch.tensor(
            self.tokenizer([caption_with_prefix], max_length=self.base_length, padding="max_length", truncation=True).input_ids,
            dtype=torch.long, device=image_tensor.device
        )

        data_dict = {
            'image': image_tensor,
            'text': text,
            'short_text': short_text,
            'add_box_loss': False,
            'use_hard_neg': False,
        }

        # Load classification labels
        if self.use_classification and self.classification_attr_names:
            classification_labels = {}
            classification_valid_mask = {}

            all_valid = True
            for attr_name in self.classification_attr_names:
                if attr_name in item:
                    label_value = item[attr_name]
                    if label_value == -1:
                        classification_labels[attr_name] = torch.tensor(0, dtype=torch.float, device=image_tensor.device)
                        classification_valid_mask[attr_name] = torch.tensor(0, dtype=torch.float, device=image_tensor.device)
                        all_valid = False
                    else:
                        classification_labels[attr_name] = torch.tensor(label_value, dtype=torch.float, device=image_tensor.device)
                        classification_valid_mask[attr_name] = torch.tensor(1, dtype=torch.float, device=image_tensor.device)
                else:
                    classification_labels[attr_name] = torch.tensor(0, dtype=torch.float, device=image_tensor.device)
                    classification_valid_mask[attr_name] = torch.tensor(0, dtype=torch.float, device=image_tensor.device)
                    all_valid = False

            data_dict['classification_labels'] = classification_labels
            data_dict['classification_valid_mask'] = classification_valid_mask
            data_dict['all_valid'] = all_valid

        return data_dict


class DataCollatorForClassification(object):
    """Collate examples for classification fine-tuning."""

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:

        batch = {}
        images = [instance['image'] for instance in instances]
        batch['image'] = torch.stack(images)
        texts = [instance['text'] for instance in instances]
        batch['text_long'] = torch.cat(texts, dim=0)
        short_texts = [instance['short_text'] for instance in instances]
        batch['text_short'] = torch.cat(short_texts, dim=0)

        batch["add_box_loss"] = False
        batch["use_hard_neg"] = False

        # Classification data
        if 'classification_labels' in instances[0]:
            classification_labels = {}
            classification_valid_mask = {}
            attr_names = list(instances[0]['classification_labels'].keys())

            for attr_name in attr_names:
                labels = [inst['classification_labels'][attr_name] for inst in instances]
                masks = [inst['classification_valid_mask'][attr_name] for inst in instances]
                classification_labels[attr_name] = torch.stack(labels)
                classification_valid_mask[attr_name] = torch.stack(masks)

            batch['classification_labels'] = classification_labels
            batch['classification_valid_mask'] = classification_valid_mask

        return batch


def make_classification_data_module(data_args, img_preprocess, tokenizer, classifier_config: Dict = None) -> Dict:
    """Make dataset and collator for classification fine-tuning."""

    train_dataset = PedestrianAttributeDataset(
        data_path=data_args.data_path,
        data_args=data_args,
        img_preprocess=img_preprocess,
        tokenizer=tokenizer,
        classifier_config=classifier_config,
    )

    eval_dataset = None
    if data_args.eval_data_path:
        eval_dataset = PedestrianAttributeDataset(
            data_path=data_args.eval_data_path,
            data_args=data_args,
            img_preprocess=img_preprocess,
            tokenizer=tokenizer,
            classifier_config=classifier_config,
        )

    data_collator = DataCollatorForClassification()
    return dict(train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
                classifier_config=classifier_config,
                classification_attr_names=train_dataset.classification_attr_names)


def make_supervised_data_module(data_args, img_preprocess, tokenizer, classifier_config: Dict = None) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    if data_args.use_classification and classifier_config is not None:
        return make_classification_data_module(data_args, img_preprocess, tokenizer, classifier_config)

    train_dataset = LazySupervisedBboxDataset(
                                data_path=data_args.data_path,
                                data_args=data_args,
                                img_preprocess=img_preprocess,tokenizer=tokenizer,)

    data_collator = DataCollatorForSupervisedDataset()
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank

    # For single GPU training with DeepSpeed, only rank 0 proceeds
    if training_args.train_use_word_size == 1 and local_rank != 0:
                exit(0)

    # NOTE: FP16 is supported via DeepSpeed

    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    # Load classifier config if provided
    classifier_config = load_classifier_config(data_args.classifier_yaml)

    tokenizer = AutoTokenizer.from_pretrained(model_args.base_model)

    image_processor = CLIPImageProcessor.from_pretrained(model_args.base_model)

    model = FGCLIPModel.from_pretrained(model_args.model_name_or_path, classifier_config=classifier_config)


    config = model.config

    model.logit_scale_finegraind = torch.nn.Parameter(torch.ones([]) * model.logit_scale)
    model.logit_scale_hardneg = torch.nn.Parameter(torch.ones([]) * model.logit_scale)

    # NOTE If only the second phase is trained, from_openai must be set to True
    if training_args.from_openai:
        print("copy and resize")
        model.resize_postion_embeding()
        model.copy_weight()
        print("fine")

    # Freeze vision/text model if specified
    if training_args.freeze_vision:
        for p in model.vision_model.parameters():
            p.requires_grad = False

    if training_args.freeze_text:
        for p in model.text_model.parameters():
            p.requires_grad = False

    data_module = make_supervised_data_module(
        data_args=data_args,
        img_preprocess=image_processor,
        tokenizer=tokenizer,
        classifier_config=classifier_config,
    )

    model.to(dtype=compute_dtype, device=training_args.device)

    # NOTE Set up for two front-passes
    training_args.gradient_checkpointing_kwargs = {"use_reentrant":False}

    trainer = CLIPTrainer(model=model,
                        args=training_args,
                        train_dataset=data_module['train_dataset'],
                        data_collator=data_module['data_collator'],
                        eval_dataset=data_module.get('eval_dataset'),
                        callbacks=[CustomProgressCallback])

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        # 找到最新的 checkpoint
        checkpoints = list(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))
        latest_checkpoint = max(checkpoints, key=lambda p: int(p.name.split('-')[1]))
        trainer.train(resume_from_checkpoint=latest_checkpoint)
    else:
        trainer.train()
    trainer.save_state()
    
    safe_save_model_for_hf_trainer(trainer=trainer,output_dir=training_args.output_dir)




if __name__ == "__main__":
    train()
