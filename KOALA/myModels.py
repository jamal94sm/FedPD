import numpy as np
import transformers
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from MyBaselines.local_models import *



##############################################################################################################
##############################################################################################################

def load_clip_model(args):
    model_name = args.Foundation_model
    model = transformers.CLIPModel.from_pretrained(model_name)
    processor = transformers.CLIPProcessor.from_pretrained(model_name, use_fast=False)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, use_fast=False)

    if "BN" in args.setup: 
        print("Unfreeze LayerNorm layers in the image encoder")

        # Unfreeze LayerNorm layers in the image encoder
        for module in model.vision_model.modules():
            if isinstance(module, torch.nn.LayerNorm):
                module.train()  # Set to training mode
                for param in module.parameters():
                    param.requires_grad = True
                    
    return model, processor, tokenizer

##############################################################################################################
##############################################################################################################

class Image_prompting_plus_Fm(nn.Module):
    def __init__(self, FM, processor, tokenizer, num_classes, name_classes, args):
        super(Image_prompting_plus_Fm, self).__init__()
        self.FM = FM.to(args.device)
        self.tokenizer = tokenizer
        self.processor = processor
        self.num_classes = num_classes
        self.name_classes = name_classes
        self.args = args

        for p in self.FM.parameters():
            p.requires_grad = False

        self.logit_scale = nn.Parameter(torch.tensor(self.FM.config.logit_scale_init_value))
        self.load_descriptions()
        self.generate_text_rep()
        self.generate_basic_text_rep()

        hidden_size = self.FM.vision_model.config.hidden_size
        self.soft_prompts = nn.Parameter(torch.randn(1, 4, hidden_size) * 0.02) # 4= num_prompts
        self.prompt_pos_embed = nn.Parameter(torch.zeros(1, 4, hidden_size))
        nn.init.trunc_normal_(self.prompt_pos_embed, std=0.02)

    def load_descriptions(self):
        df = pd.read_csv("Descriptions_Dataset.csv")
        df['descriptions'] = df['descriptions'].str.strip('\'"')
        self.descript_dataset = {
            'descriptions': list(df['descriptions'].values),
            'label': list(df['label'].values)
        }

    @torch.no_grad()
    def generate_text_rep(self):
        if "M" in self.args.setup:
            class_descriptions = self.descript_dataset['descriptions']
            self.labels = torch.tensor(self.descript_dataset['label'], device=self.args.device)
        else:
            class_descriptions = [self.args.prompt_template.format(name) for name in self.name_classes]
            self.labels = torch.arange(self.num_classes, device=self.args.device)

        tok = self.tokenizer(class_descriptions, padding=True, truncation=True, return_tensors="pt").to(self.args.device)
        self.text_rep = F.normalize(self.FM.get_text_features(tok["input_ids"]), dim=-1)

    @torch.no_grad()
    def generate_basic_text_rep(self):
        if "mean" in self.args.setup and "M" in self.args.setup:
            self.basic_text_rep = torch.stack([
                self.text_rep[(self.labels == n).nonzero(as_tuple=True)[0]].mean(dim=0)
                for n in range(self.num_classes)
            ])
        else:
            class_prompts = [self.args.prompt_template.format(name) for name in self.name_classes]
            tok = self.tokenizer(class_prompts, padding=True, truncation=True, return_tensors="pt").to(self.args.device)
            self.basic_text_rep = F.normalize(self.FM.get_text_features(tok["input_ids"]), dim=-1)

    def patchify(self, img, patch_size):
        B, C, H, W = img.shape
        assert H % patch_size == 0 and W % patch_size == 0
        patches = img.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        patches = patches.contiguous().view(B, C, -1, patch_size, patch_size)
        patches = patches.permute(0, 2, 1, 3, 4).reshape(B, patches.shape[1], -1)
        return patches

    def _embed_image_with_prompts(self, pixel_values):
        vision = self.FM.vision_model
        device = pixel_values.device
        dtype = pixel_values.dtype
        B = pixel_values.shape[0]

        # 1) Patch embedding -> (B, N, hidden)
        patch_embeds = vision.embeddings.patch_embedding(pixel_values)
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)  # (B, N, H)

        # 2) Class token -> (B, 1, hidden)
        cls_token = vision.embeddings.class_embedding.to(dtype=dtype)
        cls_tokens = cls_token.expand(B, 1, -1)

        # 3) Visual soft prompts -> (B, P, hidden)
        prompt_tokens = self.soft_prompts.to(device=device, dtype=dtype).expand(B, -1, -1)

        # 4) Positional embeddings (handle both nn.Embedding and Tensor/Parameter)
        pos_module = vision.embeddings.position_embedding
        if isinstance(pos_module, nn.Embedding):
            # Build base position ids for [CLS] + patches
            seq_len_base = 1 + patch_embeds.size(1)
            position_ids = torch.arange(seq_len_base, device=device).unsqueeze(0).expand(B, -1)  # (B, 1+N)
            base_pos = pos_module(position_ids).to(dtype=dtype)  # (B, 1+N, hidden)
        else:
            # pos_module is a Tensor/Parameter with shape (1, 1+N, hidden)
            base_pos = pos_module.to(device=device, dtype=dtype).expand(B, -1, -1)  # (B, 1+N, hidden)

        pos_cls = base_pos[:, :1, :]                 # (B, 1, hidden)
        pos_patches = base_pos[:, 1:, :]             # (B, N, hidden)
        pos_prompts = self.prompt_pos_embed.to(device=device, dtype=dtype).expand(B, -1, -1)  # (B, P, hidden)

        # 5) Compose tokens and add positions: [CLS] + [PROMPTS] + [PATCHES]
        tokens = torch.cat([cls_tokens, prompt_tokens, patch_embeds], dim=1)  # (B, 1+P+N, hidden)
        pos = torch.cat([pos_cls, pos_prompts, pos_patches], dim=1)           # (B, 1+P+N, hidden)
        hidden_states = tokens + pos

        # 6) Vision transformer forward
        hidden_states = vision.pre_layrnorm(hidden_states)
        encoder_out = vision.encoder(hidden_states)[0]        # (B, 1+P+N, hidden)
        pooled = vision.post_layernorm(encoder_out[:, 0, :])  # CLS -> (B, hidden)

        # 7) Project to CLIP space and normalize
        image_embeds = self.FM.visual_projection(pooled)      # (B, proj_dim)
        return F.normalize(image_embeds, dim=-1)


    def forward(self, x, inference=None):
        if isinstance(x, torch.Tensor):
            pixel_values = x.to(self.args.device)
        else:
            pixel_values = self.processor(images=x, return_tensors="pt")["pixel_values"].to(self.args.device)

        img_rep = self._embed_image_with_prompts(pixel_values)
        img_rep = img_rep / img_rep.norm(p=2, dim=-1, keepdim=True)

        
        
        #indices = [  np.random.choice((self.labels == n).nonzero(as_tuple=True)[0]) for n in range(self.num_classes)  ] 
        indices = [int((self.labels == n).nonzero(as_tuple=True)[0][torch.randint(0, (self.labels == n).sum(), (1,))]) for n in range(self.num_classes)]
        selected_text_rep = self.text_rep[indices]


        
        if inference:
            text_rep = self.basic_text_rep/self.basic_text_rep.norm(p=2, dim=-1, keepdim=True)
        else: 
            text_rep = selected_text_rep / selected_text_rep.norm(p=2, dim=-1, keepdim=True)
         
            
         
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * img_rep @ text_rep.t()
        return logits
        

##############################################################################################################
##############################################################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import DatasetDict

class clip_plus_linear_head(nn.Module):
    def __init__(self, FM, processor, tokenizer, num_classes, name_classes, device):
        super(clip_plus_linear_head, self).__init__()
        self.FM = FM.to(device)
        self.tokenizer = tokenizer
        self.processor = processor
        self.num_classes = num_classes
        self.name_classes = name_classes
        self.device = device

        for p in self.FM.parameters():
            p.requires_grad = False

        self.logit_scale = nn.Parameter(torch.tensor(self.FM.config.logit_scale_init_value).to(device))

        embedding_dim = self.FM.get_image_features(torch.randn(1, 3, 224, 224).to(device)).shape[-1]
        self.linear_head = nn.Linear(embedding_dim, num_classes).to(device)

    def forward(self, x):
        inputs = self.processor(images=x, return_tensors="pt").to(self.device)
        image_features = self.FM.get_image_features(**inputs)
        image_features = F.normalize(image_features, dim=-1)
        logits = self.linear_head(image_features) * self.logit_scale.exp()
        return logits

    def inference(self, dataset_dict: DatasetDict):
        self.eval()
        logits_list = []

        for sample in dataset_dict["train"]:
            image = sample["image"]
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)

            with torch.no_grad():
                image_features = self.FM.get_image_features(**inputs)
                image_features = F.normalize(image_features, dim=-1)
                logits = self.linear_head(image_features) * self.logit_scale.exp()

            logits_list.append(logits.squeeze(0).cpu())  # remove batch dim and move to CPU

        return torch.stack(logits_list)  # shape: [num_samples, num_classes]

##############################################################################################################
##############################################################################################################



