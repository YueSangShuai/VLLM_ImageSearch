"""Wrapper to load FGCLIP model by fixing relative imports."""
import sys
import types

MARS_FGCLIP_PATH = "/data/yuesang/LLM/MARS/fg-clip"
sys.path.insert(0, MARS_FGCLIP_PATH)

modeling_clip = types.ModuleType("modeling_clip")
exec(open(f"{MARS_FGCLIP_PATH}/modeling_clip.py").read(), modeling_clip.__dict__)
sys.modules["modeling_clip"] = modeling_clip

modeling_fgclip = types.ModuleType("modeling_fgclip")
modeling_fgclip_sys = types.ModuleType("modeling_fgclip")
sys.modules["modeling_fgclip"] = modeling_fgclip

exec("""
import sys
import torch
import torch.nn as nn
import math
from typing import Any, Optional, Tuple, Union
import torch.distributed.nn as nn_dist
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from typing import Tuple, Union
import modeling_clip as _clip
from torch import nn, einsum
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange, Reduce
import math
from torchvision.ops import roi_align

CLIPConfig = _clip.CLIPConfig
CLIPTextConfig = _clip.CLIPTextConfig
CLIPVisionConfig = _clip.CLIPVisionConfig
CLIPModel = _clip.CLIPModel
CLIPTextTransformer = _clip.CLIPTextTransformer
CLIPVisionTransformer = _clip.CLIPVisionTransformer
CLIPOutput = _clip.CLIPOutput
CLIPAttention = _clip.CLIPAttention
CLIPMLP = _clip.CLIPMLP

class FGCLIPConfig(CLIPConfig):
    model_type = "clip"

class FGCLIPModel(CLIPModel):
    config_class = FGCLIPConfig
    main_input_name = "text_long"

    def __init__(self, config):
        super(CLIPModel, self).__init__(config)

        if not isinstance(config.text_config, CLIPTextConfig):
            raise ValueError("config.text_config is expected to be of type CLIPTextConfig")

        if not isinstance(config.vision_config, CLIPVisionConfig):
            raise ValueError("config.vision_config is expected to be of type CLIPVisionConfig")

        text_config = config.text_config
        vision_config = config.vision_config
        text_config.eos_token_id = 49407
        text_config.pad_token_id = 49407
        text_config.bos_token_id = 49406

        self.projection_dim = config.projection_dim
        self.text_embed_dim = text_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size

        self.text_model = CLIPTextTransformer(text_config)
        self.vision_model = CLIPVisionTransformer(vision_config)
        self.visual_projection = nn.Linear(self.vision_embed_dim, self.projection_dim, bias=False)
        self.text_projection = nn.Linear(self.text_embed_dim, self.projection_dim, bias=False)
        self.text_filip_projection = nn.Linear(self.text_embed_dim, self.projection_dim, bias=False)

        self.logit_scale = nn.Parameter(torch.tensor(self.config.logit_scale_init_value))
        self.logit_scale_finegraind = nn.Parameter(torch.tensor(self.config.logit_scale_init_value))
        self.logit_scale_hardneg = nn.Parameter(torch.tensor(self.config.logit_scale_init_value))

        self.embed_dim = text_config.hidden_size
        self.world_size = 0

        self.post_init()

    def get_image_features(self, pixel_values=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = vision_outputs[1]
        image_features = self.visual_projection(pooled_output)
        return image_features

    def get_text_features(self, input_ids=None, attention_mask=None, position_ids=None, output_attentions=None, output_hidden_states=None, return_dict=None, walk_short_pos=True, use_bbox=False):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        pos_flag = walk_short_pos or use_bbox

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            walk_short_pos=pos_flag,
        )
        pooled_output = text_outputs[1]

        if walk_short_pos:
            text_features = self.text_projection(pooled_output)
        else:
            text_features = self.text_filip_projection(pooled_output)

        return text_features

    def forward_without_attn(self, x):
        residual = x
        x = self.vision_model.encoder.layers[-1].layer_norm1(x)
        x = F.linear(input=x, weight=self.vision_model.encoder.layers[-1].self_attn.v_proj.weight, bias=self.vision_model.encoder.layers[-1].self_attn.v_proj.bias)
        x = self.vision_model.encoder.layers[-1].self_attn.out_proj(x)
        x = residual + x
        residual = x
        x = self.vision_model.encoder.layers[-1].layer_norm2(x)
        x = self.vision_model.encoder.layers[-1].mlp(x)
        x = residual + x
        return x

    def get_image_dense_features(self, pixel_values=None, output_attentions=None, output_hidden_states=None, return_dict=None, interpolate_pos_encoding=False, box_info=None):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )
        bs = pixel_values.shape[0]
        length = vision_outputs[0].shape[1] - 1
        h = int(math.sqrt(length))
        w = h
        feature_map = vision_outputs.hidden_states[-2]
        feature_map = self.forward_without_attn(feature_map)[:, 1:]
        feature_map = self.vision_model.post_layernorm(feature_map)
        feature_map = self.visual_projection(feature_map)
        return feature_map
""", modeling_fgclip.__dict__)

from modeling_fgclip import FGCLIPModel
from modeling_clip import CLIPConfig

__all__ = ["FGCLIPModel", "CLIPConfig"]