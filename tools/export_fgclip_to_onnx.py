#!/usr/bin/env python3
"""
Export FG-CLIP model to separate vision and text ONNX models with accuracy validation.
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from safetensors.torch import load_file
from transformers.configuration_utils import PretrainedConfig
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from torch.nn import functional as FF


def _expand_mask(mask, dtype, tgt_len=None):
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len
    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    inverted_mask = 1.0 - expanded_mask
    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


def _make_causal_mask(input_ids_shape, dtype, device, past_key_values_length=0):
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)
    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


class CLIPTextConfig(PretrainedConfig):
    model_type = "clip_text_model"

    def __init__(self, vocab_size=49408, hidden_size=512, intermediate_size=2048, projection_dim=512,
                 num_hidden_layers=12, num_attention_heads=8, max_position_embeddings=77,
                 hidden_act="quick_gelu", layer_norm_eps=1e-5, attention_dropout=0.0,
                 initializer_range=0.02, initializer_factor=1.0, pad_token_id=1, bos_token_id=49406, eos_token_id=49407, **kwargs):
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.projection_dim = projection_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor
        self.attention_dropout = attention_dropout


class CLIPVisionConfig(PretrainedConfig):
    model_type = "clip_vision_model"

    def __init__(self, hidden_size=768, intermediate_size=3072, projection_dim=512, num_hidden_layers=12,
                 num_attention_heads=12, num_channels=3, image_size=224, patch_size=32,
                 hidden_act="quick_gelu", layer_norm_eps=1e-5, attention_dropout=0.0,
                 initializer_range=0.02, initializer_factor=1.0, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.projection_dim = projection_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act


class CLIPConfig(PretrainedConfig):
    model_type = "clip"

    def __init__(self, text_config=None, vision_config=None, projection_dim=512, logit_scale_init_value=2.6592, **kwargs):
        super().__init__(**kwargs)
        if text_config is None:
            text_config = {}
        if vision_config is None:
            vision_config = {}
        self.text_config = CLIPTextConfig(**text_config)
        self.vision_config = CLIPVisionConfig(**vision_config)
        self.projection_dim = projection_dim
        self.logit_scale_init_value = logit_scale_init_value
        self.initializer_factor = 1.0

    @classmethod
    def from_json_file(cls, json_file):
        import json
        with open(json_file, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)


class CLIPVisionEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))
        self.patch_embedding = nn.Conv2d(in_channels=config.num_channels, out_channels=self.embed_dim,
                                          kernel_size=self.patch_size, stride=self.patch_size, bias=False)
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)), persistent=False)

    def forward(self, pixel_values, interpolate_pos_encoding=False):
        batch_size = pixel_values.shape[0]
        _, _, height, width = pixel_values.shape
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        else:
            embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings

    def interpolate_pos_encoding(self, embeddings, height, width):
        num_patches = embeddings.shape[1] - 1
        num_positions = self.position_embedding.weight.shape[0] - 1
        if num_patches == num_positions and height == width:
            return self.position_embedding(self.position_ids)
        new_height = height // self.patch_size
        new_width = width // self.patch_size
        origin_patch_shape = (self.image_size // self.patch_size, self.image_size // self.patch_size)
        h, w = new_height, new_width
        positional_embedding_pre = self.position_embedding.weight.data
        length, dim = positional_embedding_pre.shape
        positional_embedding_pre = positional_embedding_pre.view(1, length, dim)
        rescaled_positional_embedding = embeddings.new_zeros(1, 1 + h * w, dim)
        rescaled_positional_embedding[0, 0] = positional_embedding_pre[0, 0]
        pe_2d = positional_embedding_pre[0, 1:].T.contiguous().view(1, -1, *origin_patch_shape)
        pe_2d = F.interpolate(pe_2d, (h, w), mode='bicubic', align_corners=False).view(-1, h * w)
        rescaled_positional_embedding[0, 1:] = pe_2d.T.contiguous()
        return rescaled_positional_embedding


class CLIPTextEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        embed_dim = config.hidden_size
        self.token_embedding = nn.Embedding(config.vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(config.max_position_embeddings, embed_dim)
        self.position_embedding_res = nn.Embedding(248, embed_dim)
        self.position_embedding_ori = nn.Embedding(248, embed_dim)
        self.mask1 = torch.zeros([248, 1])
        self.mask1[:20, :] = 1
        self.mask2 = torch.zeros([248, 1])
        self.mask2[20:, :] = 1
        self.register_buffer("position_ids", torch.arange(248).expand((1, -1)), persistent=False)

    def forward(self, input_ids=None, position_ids=None, inputs_embeds=None, walk_short_pos=False):
        seq_length = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]
        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)
        if walk_short_pos:
            position_embeddings = self.position_embedding(position_ids)
            embeddings = inputs_embeds + position_embeddings
        else:
            position_embeddings_res = self.position_embedding_res(position_ids)
            position_embeddings_ori = self.position_embedding_ori(position_ids)
            embeddings = inputs_embeds + (position_embeddings_ori * self.mask1.to(inputs_embeds.device)).type(inputs_embeds.dtype).to(inputs_embeds.device) + \
                        (position_embeddings_res * self.mask2.to(inputs_embeds.device)).type(inputs_embeds.dtype).to(inputs_embeds.device)
        return embeddings


class CLIPAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout = config.attention_dropout
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _shape(self, tensor, seq_len, bsz):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, hidden_states, attention_mask=None, causal_attention_mask=None, output_attentions=False):
        bsz, tgt_len, embed_dim = hidden_states.size()
        query_states = self.q_proj(hidden_states) * self.scale
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)
        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        if causal_attention_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + causal_attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        if attention_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        if output_attentions:
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None
        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.bmm(attn_probs, value_states)
        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, tgt_len, embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output, attn_weights_reshaped


class CLIPMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class CLIPEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = CLIPAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = CLIPMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(self, hidden_states, attention_mask=None, causal_attention_mask=None, output_attentions=False):
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(hidden_states, attention_mask, causal_attention_mask, output_attentions)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs


class CLIPEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([CLIPEncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, inputs_embeds, attention_mask=None, causal_attention_mask=None, output_attentions=False, output_hidden_states=False, return_dict=False):
        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            layer_outputs = encoder_layer(hidden_states, attention_mask, causal_attention_mask, output_attentions)
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)
        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions)


class CLIPTextTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        self.embeddings = CLIPTextEmbeddings(config)
        self.encoder = CLIPEncoder(config)
        self.final_layer_norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.eos_token_id = config.eos_token_id

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, output_attentions=False, output_hidden_states=False, return_dict=False, walk_short_pos=False):
        if input_ids is None:
            raise ValueError("You have to specify input_ids")
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids, walk_short_pos=walk_short_pos)
        causal_attention_mask = _make_causal_mask(input_shape, hidden_states.dtype, device=hidden_states.device)
        if attention_mask is not None:
            attention_mask = _expand_mask(attention_mask, hidden_states.dtype)
        encoder_outputs = self.encoder(inputs_embeds=hidden_states, attention_mask=attention_mask, causal_attention_mask=causal_attention_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.final_layer_norm(last_hidden_state)
        pooled_output = last_hidden_state[torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device), (input_ids.to(dtype=torch.int, device=last_hidden_state.device) == self.eos_token_id).int().argmax(dim=-1)]
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]
        return BaseModelOutputWithPooling(last_hidden_state=last_hidden_state, pooler_output=pooled_output, hidden_states=encoder_outputs.hidden_states, attentions=encoder_outputs.attentions)


class CLIPVisionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        self.embeddings = CLIPVisionEmbeddings(config)
        self.pre_layrnorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)
        self.encoder = CLIPEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(self, pixel_values=None, output_attentions=False, output_hidden_states=False, return_dict=False, interpolate_pos_encoding=False):
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")
        hidden_states = self.embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)
        hidden_states = self.pre_layrnorm(hidden_states)
        encoder_outputs = self.encoder(inputs_embeds=hidden_states, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        last_hidden_state = encoder_outputs[0]
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.post_layernorm(pooled_output)
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]
        return BaseModelOutputWithPooling(last_hidden_state=last_hidden_state, pooler_output=pooled_output, hidden_states=encoder_outputs.hidden_states, attentions=encoder_outputs.attentions)


class FGCLIPModel(nn.Module):
    def __init__(self, config):
        super().__init__()
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
        self.logit_scale = nn.Parameter(torch.tensor(config.logit_scale_init_value))
        self.embed_dim = text_config.hidden_size

    def post_init(self):
        pass

    def get_image_features(self, pixel_values=None, output_attentions=False, output_hidden_states=False, return_dict=False, interpolate_pos_encoding=False):
        vision_outputs = self.vision_model(pixel_values=pixel_values, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, interpolate_pos_encoding=interpolate_pos_encoding)
        pooled_output = vision_outputs[1]
        image_features = self.visual_projection(pooled_output)
        return image_features

    def get_text_features(self, input_ids=None, attention_mask=None, position_ids=None, output_attentions=False, output_hidden_states=False, return_dict=False, walk_short_pos=True, use_bbox=False):
        pos_flag = walk_short_pos or use_bbox
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, walk_short_pos=pos_flag)
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


class VisionEncoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.vision_model = model.vision_model
        self.visual_projection = model.visual_projection

    def forward(self, pixel_values, interpolate_pos_encoding=False):
        vision_outputs = self.vision_model(pixel_values=pixel_values, return_dict=True, interpolate_pos_encoding=interpolate_pos_encoding)
        pooled_output = vision_outputs.pooler_output
        image_features = self.visual_projection(pooled_output)
        return image_features


class TextEncoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.text_model = model.text_model
        self.text_projection = model.text_projection

    def forward(self, input_ids, attention_mask=None):
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True, walk_short_pos=True)
        pooled_output = text_outputs.pooler_output
        text_features = self.text_projection(pooled_output)
        return text_features


def load_fgclip_model(model_path):
    config = CLIPConfig.from_json_file(os.path.join(model_path, "config.json"))
    model = FGCLIPModel(config)
    state_dict = load_file(os.path.join(model_path, "model.safetensors"))
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def export_vision_model(model, output_path, opset=14):
    print(f"Exporting vision model to {output_path}...")
    wrapper = VisionEncoderWrapper(model)
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(wrapper, dummy_input, output_path, export_params=True, opset_version=opset,
                      input_names=["pixel_values"], output_names=["image_features"],
                      dynamic_axes={"pixel_values": {0: "batch_size"}, "image_features": {0: "batch_size"}})
    print(f"Vision model exported successfully!")


def export_text_model(model, output_path, opset=14):
    print(f"Exporting text model to {output_path}...")
    wrapper = TextEncoderWrapper(model)
    max_seq_length = model.text_model.config.max_position_embeddings
    dummy_input_ids = torch.randint(0, 49408, (1, max_seq_length))
    dummy_attention_mask = torch.ones(1, max_seq_length, dtype=torch.long)
    torch.onnx.export(wrapper, (dummy_input_ids, dummy_attention_mask), output_path, export_params=True, opset_version=opset,
                      input_names=["input_ids", "attention_mask"], output_names=["text_features"],
                      dynamic_axes={"input_ids": {0: "batch_size", 1: "sequence_length"},
                                    "attention_mask": {0: "batch_size", 1: "sequence_length"},
                                    "text_features": {0: "batch_size"}})
    print(f"Text model exported successfully!")


def verify_vision_model(original_model, onnx_path, atol=1e-4):
    print(f"\nVerifying vision model from {onnx_path}...")
    import onnx
    import onnxruntime as ort
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model structure is valid!")
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    test_images = [torch.randn(1, 3, 224, 224), torch.randn(2, 3, 224, 224)]
    max_diff = 0.0
    for i, pixel_values in enumerate(test_images):
        with torch.no_grad():
            pytorch_output = original_model.get_image_features(pixel_values=pixel_values)
        ort_inputs = {session.get_inputs()[0].name: pixel_values.numpy()}
        onnx_output = session.run(None, ort_inputs)[0]
        diff = np.abs(pytorch_output.numpy() - onnx_output).max()
        max_diff = max(max_diff, diff)
        print(f"  Test {i+1}: Max absolute difference = {diff:.6e}")
    passed = max_diff < atol
    print(f"Vision verification {'PASSED' if passed else 'FAILED'} (max_diff={max_diff:.6e}, atol={atol})")
    return passed


def verify_text_model(original_model, onnx_path, atol=1e-4):
    print(f"\nVerifying text model from {onnx_path}...")
    import onnx
    import onnxruntime as ort
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model structure is valid!")
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    max_seq_length = original_model.text_model.config.max_position_embeddings
    test_texts = [
        (torch.randint(0, 49408, (1, max_seq_length)), torch.ones(1, max_seq_length, dtype=torch.long)),
        (torch.randint(0, 49408, (2, max_seq_length)), torch.ones(2, max_seq_length, dtype=torch.long)),
    ]
    max_diff = 0.0
    for i, (input_ids, attention_mask) in enumerate(test_texts):
        with torch.no_grad():
            pytorch_output = original_model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
        ort_inputs = {session.get_inputs()[0].name: input_ids.numpy(), session.get_inputs()[1].name: attention_mask.numpy()}
        onnx_output = session.run(None, ort_inputs)[0]
        diff = np.abs(pytorch_output.numpy() - onnx_output).max()
        max_diff = max(max_diff, diff)
        print(f"  Test {i+1}: Max absolute difference = {diff:.6e}")
    passed = max_diff < atol
    print(f"Text verification {'PASSED' if passed else 'FAILED'} (max_diff={max_diff:.6e}, atol={atol})")
    return passed


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Export FG-CLIP to ONNX models")
    parser.add_argument("--model_path", type=str, default="/data/yuesang/LLM/MARS/fg-clip", help="Path to FG-CLIP model")
    parser.add_argument("--output_dir", type=str, default="/data/yuesang/LLM/FG-CLIP/onnx_exports", help="Output directory")
    parser.add_argument("--opset", type=int, default=14, help="ONNX opset version")
    parser.add_argument("--verify", action="store_true", default=True, help="Verify accuracy after export")
    parser.add_argument("--atol", type=float, default=1e-4, help="Absolute tolerance for verification")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Loading FG-CLIP model from {args.model_path}...")
    model = load_fgclip_model(args.model_path)
    print("Model loaded successfully!")

    vision_onnx_path = os.path.join(args.output_dir, "vision_model.onnx")
    text_onnx_path = os.path.join(args.output_dir, "text_model.onnx")

    export_vision_model(model, vision_onnx_path, args.opset)
    export_text_model(model, text_onnx_path, args.opset)

    if args.verify:
        print("\n" + "="*50)
        print("Starting verification...")
        print("="*50)
        vision_passed = verify_vision_model(model, vision_onnx_path, args.atol)
        text_passed = verify_text_model(model, text_onnx_path, args.atol)
        print("\n" + "="*50)
        if vision_passed and text_passed:
            print("ALL VERIFICATIONS PASSED!")
        else:
            print("SOME VERIFICATIONS FAILED!")
            import sys
            sys.exit(1)
        print("="*50)

    print(f"\nExport complete! Files saved to:")
    print(f"  Vision: {vision_onnx_path}")
    print(f"  Text: {text_onnx_path}")


if __name__ == "__main__":
    main()