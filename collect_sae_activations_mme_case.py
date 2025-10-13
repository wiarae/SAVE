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

def build_filter_mask_from_jsonl(jsonl_path: str,
                                 substrings: Optional[List[str]] = None,
                                 regex: Optional[str] = None) -> List[bool]:
    """
    jsonl 파일을 순서대로 읽으며, 각 항목의 'image' 필드가 조건을 만족하면 True, 아니면 False.
    substrings 중 하나라도 포함되거나 regex가 매칭되면 True로 간주.
    둘 다 None이면 전부 True.
    """
    mask = []
    pat = re.compile(regex) if regex else None
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            img = obj.get("image", "")
            ok = True
            if substrings:
                ok = any(sub in img for sub in substrings)
            if pat:
                ok = ok or bool(pat.search(img))
            mask.append(ok)
    return mask


def _norm_yn(text: str) -> str:
    """모델 출력/GT 라벨을 'yes'/'no'로 정규화(가능하면 접두부 매칭)."""
    t = (text or "").strip().lower()
    if t.startswith("yes"):
        return "yes"
    if t.startswith("no"):
        return "no"
    return t  # 예외적으로 yes/no가 아닐 경우 원문 반환(드물지만 유지)


class FeatureAnswerCache:
    CASE_KEYS = ("both_wrong", "both_correct", "both_yes", "both_no")
    CASE_DIRS = {
        "both_wrong": "case1_both_wrong",
        "both_correct": "case2_both_correct",
        "both_yes": "case3_both_yes",
        "both_no": "case4_both_no",
    }

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

        # 케이스별 버킷: {case_key: {module_path: [Tensor(pair=2, D), ...]}}
        self.case_feature_locations = {
            k: defaultdict(list) for k in self.CASE_KEYS
        }
        self.case_feature_activations = {
            k: defaultdict(list) for k in self.CASE_KEYS
        }

        self.filter_mask = filter_mask
        self.example_idx = 0

        # 2개 예시를 쌍으로 묶기 위한 버퍼
        self._pair_buffer = []  # 각 원소: {"pred":"yes/no", "gt":"yes/no", "features_by_module":{module:tensor(D)}}

    def _classify_pair(self, p1: str, g1: str, p2: str, g2: str) -> str:
        """
        네 가지 케이스 중 하나를 반환.
        - both_wrong: 두 질문 모두 오답
        - both_correct: 두 질문 모두 정답
        - both_yes: 예측이 모두 yes
        - both_no: 예측이 모두 no
        (POPE류의 상보 yes/no 라벨쌍 가정 시 상호배타)
        """
        p1n, g1n, p2n, g2n = _norm_yn(p1), _norm_yn(g1), _norm_yn(p2), _norm_yn(g2)
        if p1n == g1n and p2n == g2n:
            return "both_correct"
        if p1n != g1n and p2n != g2n:
            return "both_wrong"
        if p1n == "yes" and p2n == "yes":
            return "both_yes"
        if p1n == "no" and p2n == "no":
            return "both_no"
        # 이외 조합(이상치)은 일단 혼합 판단: yes/no 불일치 -> 케이스 분류 애매, 여기서는 보수적으로 FP 성향으로 수렴
        return "both_yes" if p1n == "yes" or p2n == "yes" else "both_no"

    def _flush_pair_if_ready(self, batch_number: int):
        """버퍼에 2개가 쌓이면 케이스 판정 후 케이스 버킷에 저장."""
        if len(self._pair_buffer) < 2:
            return

        a, b = self._pair_buffer[0], self._pair_buffer[1]
        case_key = self._classify_pair(a["pred"], a["gt"], b["pred"], b["gt"])

        # 모듈별로 (2, D) 텐서를 쌓아서 보관
        for module_path in self.submodule_dict.keys():
            fa = a["features_by_module"][module_path]  # (D,)
            fb = b["features_by_module"][module_path]  # (D,)
            pair_tensor = torch.stack([fa, fb], dim=0)  # (2, D)

            self.case_feature_locations[case_key][module_path].append(pair_tensor)
            self.case_feature_activations[case_key][module_path].append(pair_tensor)

        # 버퍼 비우기
        self._pair_buffer.clear()

    def run(self, dataset: Dataset):
        def collate_fn(batch: Sequence[Dict]):
            return {
                "image": [item["image"].convert("RGB") for item in batch],
                "query": [item["query"] for item in batch],
                "label": [item["label"] for item in batch],
                # 주어진 데이터 구조 상 경로/키를 모르므로 페어링은 순서 기반으로 처리
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

        with tqdm(total=total_batches, desc="Caching features (paired answer split)", disable=not rank_zero) as pbar:
            for batch_number, batch in enumerate(dataloader):
                # ---- 필터: batch_size=1 가정 ----
                if self.filter_mask is not None:
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
                    pred_label = decoded

                    split_token = '[/inst]'
                    if split_token in pred_label:
                        pred_label = pred_label.split(split_token, 1)[-1].strip()
                        pred_label = pred_label.strip(',')
                    else:
                        pred_label = pred_label.strip()

                    gt_label = labels[0]
                    pred_n = _norm_yn(pred_label)
                    gt_n = _norm_yn(gt_label)

                    # 모듈별 sparse feature 계산
                    features_by_module = {}
                    for module_path, latents in buffer.items():
                        sae = self.submodule_dict[module_path]
                        latents = latents.float()
                        _, features = sae(latents)

                        topk = 256
                        values, indices = torch.topk(features, k=topk, dim=-1)
                        sparse_features = torch.zeros_like(features)
                        sparse_features.scatter_(-1, indices, values)

                        result = sparse_features[0].cpu()  # (D,)
                        features_by_module[module_path] = result

                    # 현재 예시를 페어 버퍼에 적재 후, 2개가 되면 케이스 저장
                    self._pair_buffer.append({"pred": pred_n, "gt": gt_n, "features_by_module": features_by_module})
                    self._flush_pair_if_ready(batch_number)

                    # 주기적 임시 저장
                    if (batch_number + 1) % save_every == 0:
                        self._save_temp(batch_number)

                torch.cuda.empty_cache()
                pbar.update(1)

        # 에폭 종료 시 남은 1개(비정상) 버퍼는 폐기하거나 로그만 남길 수 있음. 여기서는 무시.
        self._save_final()
        if dist.is_initialized():
            dist.barrier()

    def _save_temp(self, batch_number: int):
        """
        케이스/모듈별 누적을 임시 safetensors로 저장.
        파일 구조: {save_dir}/{case_dir}/{module_path}/Rank{rank}_{batch_number}.safetensors
        텐서 shape: (N, 2, D)
        """
        for case_key in self.CASE_KEYS:
            case_dir = self.CASE_DIRS[case_key]
            for module_path, loc_list in self.case_feature_locations[case_key].items():
                if len(loc_list) == 0:
                    continue
                out_dir = os.path.join(self.save_dir, case_dir, module_path)
                os.makedirs(out_dir, exist_ok=True)
                output_file = os.path.join(out_dir, f"Rank{self.rank}_{batch_number}.safetensors")
                data = {
                    "locations": torch.stack(loc_list, dim=0),   # (N, 2, D)
                    "activations": torch.stack(self.case_feature_activations[case_key][module_path], dim=0),  # (N, 2, D)
                }
                save_file(data, output_file)

    def _save_final(self):
        # 최종 저장(배치번호 표기를 'final'로 통일)
        self._save_temp(batch_number="final")


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
    parser.add_argument("--save-dir", type=str, default="sae_cache_mme_casewise")
    parser.add_argument("--layer", type=int, required=True)

    # ↓↓↓ 수정: nargs="+" 추가
    parser.add_argument("--substr", type=str, nargs="+", default=None,
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
    if args.substr or args.filter_regex:
        filter_mask = build_filter_mask_from_jsonl(
            jsonl_path=pope_path,
            substrings=args.substr,
            regex=args.filter_regex
        )
    # ----------------------------------------

    layer = args.layer
    filter_substrs = ["color", "existence", "count"]
    layers = [8, 12, 16, 20]
    file = f"SAE/sae_final_layer{layer}_8x_norm.pkl"

    sae = SAE().to("cuda")
    raw_state_dict = torch.load(file, map_location='cpu')
    clean_state_dict = remove_module_prefix(raw_state_dict)
    sae.load_state_dict(clean_state_dict)
    for sub_str in filter_substrs:
        for layer in layers:
            submodule_dict = {
                f"model.layers.{layer}": sae  # ← 해당 layer 이름과 일치해야 함
            }
            save_dir = f"{args.save_dir}/{sub_str}/{layer}"
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
