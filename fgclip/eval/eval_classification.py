import os
import torch
import argparse
import json
import glob
import numpy as np
from tqdm import tqdm
from PIL import Image
import yaml

from fgclip.model.clip_strc.fgclip import FGCLIPModel
from transformers import CLIPImageProcessor, AutoTokenizer, CLIPConfig, CLIPTextConfig, CLIPVisionConfig
from torch.utils.data import Dataset, DataLoader


def load_classifier_config(yaml_path):
    """Load classifier configuration from YAML file."""
    if yaml_path is None or not os.path.exists(yaml_path):
        return None
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config.get('classiffer_config', None)


class ClassificationEvalDataset(Dataset):
    """Dataset for classification evaluation."""

    def __init__(self, data_path, image_folder, img_preprocess, tokenizer,
                 classifier_config, max_length=77, image_size=224):
        super().__init__()

        # Load data
        if data_path.endswith('.json'):
            self.list_data_dict = json.load(open(data_path, "r", encoding="utf-8"))
        elif data_path.endswith('.txt'):
            lines = open(data_path, "r", encoding="utf-8").readlines()
            self.list_data_dict = []
            for line in lines:
                json_file = line.rstrip()
                self.list_data_dict += json.load(open(json_file, "r", encoding="utf-8"))
        else:
            json_files = glob.glob(os.path.join(data_path, '*.json'))
            self.list_data_dict = []
            for json_file in json_files:
                self.list_data_dict += json.load(open(json_file, "r", encoding="utf-8"))

        self.image_folder = image_folder
        self.preprocess = img_preprocess
        self.tokenizer = tokenizer
        self.classifier_config = classifier_config
        self.max_length = max_length
        self.image_size = image_size

        # Get attribute names
        self.classification_attr_names = []
        if classifier_config and classifier_config.get('open', False):
            classes = classifier_config.get('classes', {})
            self.classification_attr_names = list(classes.keys())

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, idx):
        item = self.list_data_dict[idx]

        # Load image
        image_path = item["image_path"]
        if not os.path.isabs(image_path):
            image_path = os.path.join(self.image_folder, image_path)
        image = Image.open(image_path).convert("RGB")
        image = image.resize((self.image_size, self.image_size))
        image_tensor = self.preprocess.preprocess(image, return_tensors='pt')['pixel_values'][0]

        # Load caption
        text_path = item["text_path"]
        with open(text_path, 'r', encoding='utf-8') as f:
            caption = f.read().strip()
        short_caption = caption[:100] if len(caption) > 100 else caption

        text = torch.tensor(
            self.tokenizer([caption], max_length=self.max_length, padding="max_length", truncation=True).input_ids,
            dtype=torch.long
        )
        short_text = torch.tensor(
            self.tokenizer(["a photo of " + short_caption], max_length=self.max_length, padding="max_length", truncation=True).input_ids,
            dtype=torch.long
        )

        result = {
            'image': image_tensor,
            'text_long': text,
            'text_short': short_text,
        }

        # Load labels
        if self.classification_attr_names:
            labels = {}
            valid_mask = {}
            for attr_name in self.classification_attr_names:
                if attr_name in item:
                    value = item[attr_name]
                    if value == -1:
                        labels[attr_name] = 0
                        valid_mask[attr_name] = 0
                    else:
                        labels[attr_name] = value
                        valid_mask[attr_name] = 1
                else:
                    labels[attr_name] = 0
                    valid_mask[attr_name] = 0
            result['labels'] = labels
            result['valid_mask'] = valid_mask

        return result


def compute_ma_metric(all_labels, all_preds, all_masks):
    """Compute mean Accuracy (mA) metric."""
    num_attrs = len(all_labels[0])
    acc_per_attr = []

    for i in range(num_attrs):
        labels = [item[i] for item in all_labels]
        preds = [item[i] for item in all_preds]
        masks = [item[i] for item in all_masks]

        tp = sum(1 for l, p, m in zip(labels, preds, masks) if m == 1 and l == 1 and p == 1)
        tn = sum(1 for l, p, m in zip(labels, preds, masks) if m == 1 and l == 0 and p == 0)
        fp = sum(1 for l, p, m in zip(labels, preds, masks) if m == 1 and l == 0 and p == 1)
        fn = sum(1 for l, p, m in zip(labels, preds, masks) if m == 1 and l == 1 and p == 0)

        acc = (tp + tn) / (tp + tn + fp + fn + 1e-6)
        acc_per_attr.append(acc)

    return np.mean(acc_per_attr)


def evaluate_model(args):
    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_base)
    image_processor = CLIPImageProcessor.from_pretrained(args.model_base)

    # Load classifier config
    classifier_config = load_classifier_config(args.classifier_yaml)

    model = FGCLIPModel.from_pretrained(args.model_path, classifier_config=classifier_config)
    model = model.cuda()
    model.eval()

    # Create dataset
    dataset = ClassificationEvalDataset(
        data_path=args.data_path,
        image_folder=args.image_folder,
        img_preprocess=image_processor,
        tokenizer=tokenizer,
        classifier_config=classifier_config,
        max_length=args.max_length,
        image_size=args.image_size
    )

    print(f"Evaluation dataset size: {len(dataset)}")

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Collect predictions
    all_labels = []
    all_preds = []
    all_masks = []

    attr_names = dataset.classification_attr_names

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images = batch['image'].cuda()
            text_long = batch['text_long'].cuda()
            text_short = batch['text_short'].cuda()

            outputs = model(
                text_short=text_short,
                text_long=text_long,
                image=images,
            )

            # Get classification predictions
            if outputs.classification_logits:
                for attr_name in attr_names:
                    if attr_name in outputs.classification_logits:
                        logits = outputs.classification_logits[attr_name]
                        probs = torch.sigmoid(logits.squeeze(-1))
                        preds = (probs > 0.5).long().cpu().numpy()

                        labels = batch['labels'][attr_name].numpy()
                        masks = batch['valid_mask'][attr_name].numpy()

                        for i in range(len(preds)):
                            all_labels.append(int(labels[i]))
                            all_preds.append(int(preds[i]))
                            all_masks.append(int(masks[i]))

    # Compute metrics
    if all_labels:
        # Reshape to batch format
        batch_size = args.batch_size
        num_attrs = len(attr_names)

        labels_array = np.array(all_labels).reshape(-1, num_attrs)
        preds_array = np.array(all_preds).reshape(-1, num_attrs)
        masks_array = np.array(all_masks).reshape(-1, num_attrs)

        ma_metric = compute_ma_metric(labels_array, preds_array, masks_array)

        print(f"\n=== Classification Evaluation Results ===")
        print(f"mA (mean Accuracy): {ma_metric:.4f}")

        # Per-attribute accuracy
        print("\nPer-attribute accuracy:")
        for i, attr_name in enumerate(attr_names):
            tp = ((labels_array[:, i] == 1) & (preds_array[:, i] == 1) & (masks_array[:, i] == 1)).sum()
            tn = ((labels_array[:, i] == 0) & (preds_array[:, i] == 0) & (masks_array[:, i] == 1)).sum()
            fp = ((labels_array[:, i] == 0) & (preds_array[:, i] == 1) & (masks_array[:, i] == 1)).sum()
            fn = ((labels_array[:, i] == 1) & (preds_array[:, i] == 0) & (masks_array[:, i] == 1)).sum()

            acc = (tp + tn) / (tp + tn + fp + fn + 1e-6)
            print(f"  {attr_name}: {acc:.4f}")
    else:
        print("No valid predictions collected.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default="qihoo360/fg-clip-base")
    parser.add_argument("--data-path", type=str, required=True, help="Path to data JSON/YAML file")
    parser.add_argument("--image-folder", type=str, required=True)
    parser.add_argument("--classifier-yaml", type=str, required=True, help="Path to classifier YAML config")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-length", type=int, default=77)

    args = parser.parse_args()
    evaluate_model(args)