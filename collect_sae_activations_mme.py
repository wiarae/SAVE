import os
import torch
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch.distributed as dist
from safetensors.torch import save_file
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
from sae_auto_interp.sae import Sae
from typing import Dict, Sequence
import torch.nn as nn
from eval_data_loader import POPEDataSetEval
import argparse

import json, re
from typing import List, Optional

def build_filter_mask_from_jsonl(
    jsonl_path: str,
    substrings: Optional[List[str]] = None,
    regex: Optional[str] = None,
    *,
    filter_field: str = "image",      # "image" | "query" | "both"
    case_insensitive: bool = True,    # 기본: 대소문자 무시
) -> List[bool]:
    """
    OR-필터(합집합) 전용:
      - substrings 중 하나라도 포함이면 통과
      - regex가 매치되면 통과
      - 둘 다 주면 (substrings 통과) OR (regex 통과)
      - 아무 것도 안 주면 전부 True
    """
    import re, json

    def norm(s: str) -> str:
        return s.lower() if case_insensitive else s

    subs = [norm(s) for s in (substrings or [])]
    pat = re.compile(regex, re.IGNORECASE if (regex and case_insensitive) else 0) if regex else None

    mask: List[bool] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            img = norm(obj.get("image", ""))
            qry = norm(obj.get("query", ""))

            # 검사 대상 문자열 만들기
            hay = []
            if filter_field in ("image", "both"):
                hay.append(img)
            if filter_field in ("query", "both"):
                hay.append(qry)
            haystack = " || ".join(hay) if hay else ""

            cond_sub = any(sub in haystack for sub in subs) if subs else False
            cond_re  = bool(pat.search(haystack)) if pat else False

            if not subs and not pat:
                ok = True
            else:
                ok = cond_sub or cond_re  # ← 합집합
            mask.append(ok)
    return mask



class FeatureAnswerCache:
    def __init__(
        self,
        model: LlavaNextForConditionalGeneration,
        processor: LlavaNextProcessor,
        submodule_dict: Dict[str, Sae],
        batch_size: int,
        shard_size: int,
        save_dir: str,
        filter_mask: Optional[Sequence[bool]] = None,
    ):
        self.model = model
        self.processor = processor
        self.submodule_dict = submodule_dict
        self.batch_size = batch_size
        self.shard_size = shard_size
        self.save_dir = save_dir
        self.name_to_module = {
            name: model.language_model.get_submodule(name)
            for name in submodule_dict.keys()
        }
        self.module_to_name = {v: k for k, v in self.name_to_module.items()}
        self.rank = dist.get_rank() if dist.is_initialized() else 0

        self.correct_feature_locations = defaultdict(list)
        self.correct_feature_activations = defaultdict(list)
        self.incorrect_feature_locations = defaultdict(list)
        self.incorrect_feature_activations = defaultdict(list)
        self.filter_mask = filter_mask
        self.example_idx = 0 
        

    def run(self, dataset: Dataset):
        def collate_fn(batch: Sequence[Dict]):
            return {
                "image": [item["image"].convert("RGB") for item in batch],
                "query": [item["query"] for item in batch],
                "label": [item["label"] for item in batch],
            }

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            drop_last=False,
            shuffle=False,
            num_workers=0,
        )

        device = self.model.device
        total_batches = len(dataloader)
        save_every = 3000
        rank_zero = not dist.is_initialized() or dist.get_rank() == 0

        with tqdm(total=total_batches, desc="Caching features (answer split)", disable=not rank_zero) as pbar:
            for batch_number, batch in enumerate(dataloader):
                # ---- 필터: batch_size=1 가정 ----
                if self.filter_mask is not None:
                    # dataloader 순서 == jsonl 순서 (shuffle=False)
                    if self.example_idx < len(self.filter_mask) and not self.filter_mask[self.example_idx]:
                        self.example_idx += 1
                        pbar.update(1)
                        continue
                    self.example_idx += 1
                # -------------------------------

                images = batch["image"]
                queries = batch["query"]
                labels = batch["label"]

                conversations = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": queries[0]},
                        ],
                    },
                ]
                prompts = self.processor.apply_chat_template(conversations, add_generation_prompt=True)
                inputs = self.processor(images, prompts, return_tensors="pt").to(device)

                with torch.no_grad():
                    buffer = {}

                    def hook(module: torch.nn.Module, _, outputs):
                        if isinstance(outputs, tuple):
                            outputs = outputs[0]
                        name = self.module_to_name[module]
                        buffer[name] = outputs

                    handles = [
                        mod.register_forward_hook(hook)
                        for mod in self.name_to_module.values()
                    ]

                    try:
                        outputs = self.model.generate(**inputs, max_new_tokens=2)
                    finally:
                        for handle in handles:
                            handle.remove()

                    decoded = self.processor.decode(outputs[0], skip_special_tokens=True).lower()
                    print(decoded)
                    pred_label = decoded

                    split_token = '[/inst]'
                    if split_token in pred_label:
                        pred_label = pred_label.split(split_token, 1)[-1].strip()
                        pred_label = pred_label.strip(',')
                    else:
                        pred_label = pred_label.strip()

                    gt_label = labels[0]
                    is_correct = (pred_label.lower() == gt_label.lower())
                
                    for module_path, latents in buffer.items():
                        sae = self.submodule_dict[module_path]
                        latents = latents.float()
                        _, features = sae(latents)

                        topk = 256
                        values, indices = torch.topk(features, k=topk, dim=-1)
                        sparse_features = torch.zeros_like(features)
                        sparse_features.scatter_(-1, indices, values)

                        result = sparse_features[0].cpu()

                        if is_correct:
                            self.correct_feature_locations[module_path].append(result)
                            self.correct_feature_activations[module_path].append(result)
                        else:
                            self.incorrect_feature_locations[module_path].append(result)
                            self.incorrect_feature_activations[module_path].append(result)

                        if (batch_number + 1) % save_every == 0:
                            self._save_temp(module_path, batch_number)

                torch.cuda.empty_cache()
                pbar.update(1)

        self._save_final()
        if dist.is_initialized():
            dist.barrier()

    def _save_temp(self, module_path: str, batch_number: int):
        # Save correct
        if len(self.correct_feature_locations[module_path]) > 0:
            correct_dir = os.path.join(self.save_dir, "correct", module_path)
            os.makedirs(correct_dir, exist_ok=True)
            output_file = os.path.join(correct_dir, f"Rank{self.rank}_{batch_number}.safetensors")
            data = {
                "locations": torch.stack(self.correct_feature_locations[module_path], dim=0),
                "activations": torch.stack(self.correct_feature_activations[module_path], dim=0),
            }
            save_file(data, output_file)

        # Save incorrect (key: (module_path, label_pair))
        if len(self.incorrect_feature_locations[module_path]) > 0:
            incorrect_dir = os.path.join(self.save_dir, "incorrect", module_path)
            os.makedirs(incorrect_dir, exist_ok=True)
            output_file = os.path.join(incorrect_dir, f"Rank{self.rank}_{batch_number}.safetensors")
            data = {
                "locations": torch.stack(self.incorrect_feature_locations[module_path], dim=0),
                "activations": torch.stack(self.incorrect_feature_activations[module_path], dim=0),
            }
            save_file(data, output_file)


    def _save_final(self):
        for module_path in self.correct_feature_locations.keys():
            self._save_temp(module_path, batch_number="final")

def remove_module_prefix(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v  # remove 'module.' prefix
        else:
            new_state_dict[k] = v
    return new_state_dict

class SAE(nn.Module):
    def __init__(self):
        super(SAE, self).__init__()
        self.fc1 = nn.Linear(4096, int(8 * 4096), bias=True)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(int(8 * 4096), 4096, bias=True)
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        W_d = torch.randn(self.fc2.weight.size())
        W_d = W_d / W_d.norm(dim=1, keepdim=True) * 0.1
        self.fc2.weight.data = W_d
        self.fc1.weight.data = W_d.t()

    def forward(self, x):
        x = x.float()
        features = self.relu1(self.fc1(x))
        x_hat = self.fc2(features)
        return x_hat, features

    def encode(self, x, topk=256):
        x = x.float()
        with torch.no_grad():
            hidden = self.relu1(self.fc1(x))  # (B, 32768)
            values, indices = torch.topk(hidden, k=topk, dim=-1)
            mask = torch.zeros_like(hidden)
            mask.scatter_(-1, indices, values)
            return mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="llava-hf/llava-v1.6-mistral-7b-hf", help="Model name")
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--layer", type=int, default=24)

    # ↓↓↓ 수정: nargs="+" 추가
    parser.add_argument("--filter-substr", type=str, nargs="+", default=None,
                        help="jsonl의 'image' 경로에 포함되어야 할 부분 문자열들 (예: landmark tower)")
    parser.add_argument("--filter-regex", type=str, default=None,
                        help="jsonl의 'image' 경로에 매칭될 정규식 (선택)")
   


    args = parser.parse_args()

    processor = LlavaNextProcessor.from_pretrained(args.model_name)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        args.model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).to("cuda")

    pope_path = 'mme_for_gen.jsonl'
    dataset = POPEDataSetEval(
        pope_path=pope_path,
        data_path="MME_Benchmark_release_version/MME_Benchmark",
        trans=processor.image_processor,
    )

    # --- 필터 마스크 구성 (jsonl 순서 기준) ---
    filter_mask = None
    if args.filter_substr or args.filter_regex:
        filter_mask = build_filter_mask_from_jsonl(
            jsonl_path=pope_path,
            substrings=args.filter_substr,
            regex=args.filter_regex, 
        )
    # ----------------------------------------

    layer = args.layer
    file = f"SAE/sae_final_layer{layer}_8x_norm.pkl"

    sae = SAE().to("cuda")
    raw_state_dict = torch.load(file, map_location='cpu')
    clean_state_dict = remove_module_prefix(raw_state_dict)
    sae.load_state_dict(clean_state_dict)

    layer = args.layer
    layers = [8, 12, 16]
    file = f"SAE/sae_final_layer{layer}_8x_norm.pkl"

    sae = SAE().to("cuda")
    raw_state_dict = torch.load(file, map_location='cpu')
    clean_state_dict = remove_module_prefix(raw_state_dict)
    sae.load_state_dict(clean_state_dict)
    for layer in layers:
        submodule_dict = {
            f"model.layers.{layer}": sae  # ← 해당 layer 이름과 일치해야 함
        }
        save_dir = f"{args.save_dir}/{args.filter_substr[0]}/{layer}"
        cacher = FeatureAnswerCache(
            model=model,
            processor=processor,
            submodule_dict=submodule_dict,
            batch_size=1,
            shard_size=0,
            save_dir=save_dir,
            filter_mask=filter_mask,  # ← 전달
        )

        cacher.run(dataset)

