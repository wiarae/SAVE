"""
Steering Evaluation + CHAIR Evaluation (multi-feature steering)
"""
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import os
import json
import random
from baukit import Trace
import warnings
import argparse

warnings.filterwarnings("ignore")

random.seed(42)
torch.set_grad_enabled(False)


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
        features = self.relu1(self.fc1(x))
        x_hat = self.fc2(features)
        return x_hat, features


def remove_module_prefix(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            new_state_dict[key[7:]] = value
        else:
            new_state_dict[key] = value
    return new_state_dict


def main(args):
    layer = args.layer
    file = f"SAE/sae_final_layer{layer}_8x_norm.pkl"

    if args.substr == "lure":
        layer_to_feature = {8: 5824, 12: 8830, 16: 16342, 20: 12525, 24: 2229}
        hal_feature = {8: 10300, 12: 8875, 16: 9702, 20: 31370, 24: 23278}
  
    else:
        raise ValueError(f"Unknown substr: {args.substr}")

    if args.feature_idxs is not None and len(args.feature_idxs) > 0:
        feature_idxs = args.feature_idxs               
    else:
        feature_idxs = [layer_to_feature[layer]]       

    hal_feature_idx = hal_feature[layer]             
    STEER_ALPHA = args.steer_alpha
    models, saes = [], []

    device = "cuda"
    model = LlavaNextForConditionalGeneration.from_pretrained(
        "llava-hf/llava-v1.6-mistral-7b-hf",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    ).to(device)
    sae = SAE().half().to(device)
    state = torch.load(file, map_location="cpu")
    sae.load_state_dict(remove_module_prefix(state))
    models.append(model.eval())
    saes.append(sae.eval())

    processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

    # CHAIR evaluation
    os.makedirs(args.save_dir, exist_ok=True)
    with open("chair/chair.jsonl", "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f.readlines()]

    results = []
    for i, data in tqdm(enumerate(lines), total=len(lines)):
        image_id = data["image_id"]
        image_path = f"data/val2014/COCO_val2014_{str(image_id).zfill(12)}.jpg"
        prompt = "user\n<image>\nPlease describe this image in detail.\n\nassistant\n"
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            continue

        inputs = processor(images=image, text=[prompt], return_tensors="pt", padding=True).to(device)
        hal = args.hal
        faith = args.faith
        both = args.both
        alpha_ablation = args.alpha_ablation

        def intervene(output):
   
            x = output[0] if isinstance(output, (tuple, list)) else output  # (B, T, D)
            B, T, D_ = x.shape

            dev = x.device
            dt = x.dtype

            vecs = []
            for idx in feature_idxs:
                direction_i = sae.fc2.weight[:, idx].to(device=dev, dtype=dt)  # (D,)
                vec_i = F.normalize(direction_i, dim=0)
                vecs.append(vec_i)

            if len(vecs) == 1:
                faith_vec = vecs[0]
            else:
                faith_vec = torch.stack(vecs, dim=0).mean(dim=0)
                faith_vec = F.normalize(faith_vec, dim=0)

            hal_vec = None
            if hal or both:
                assert hal_feature_idx is not None, "Require hal_feature_idx."
                hal_direction = sae.fc2.weight[:, hal_feature_idx].to(device=dev, dtype=dt)
                hal_vec = F.normalize(hal_direction, dim=0)  # (D,)

            x_new = x.clone()

            # broadcast: (1, 1, D)
            faith_vec_b = faith_vec.view(1, 1, -1)
            hal_vec_b = hal_vec.view(1, 1, -1) if hal_vec is not None else None


            if alpha_ablation:
                x_norm = x_new.norm(dim=-1, keepdim=True)  # (B, T, 1)
                eps = 1e-6
                gamma = 0.2

                d_norm_faith = faith_vec.norm().clamp_min(eps)
                alpha_faith = gamma * x_norm / d_norm_faith  # (B, T, 1)

                if hal_vec_b is not None:
                    d_norm_hal = hal_vec.norm().clamp_min(eps)
                    alpha_hal = gamma * x_norm / d_norm_hal   # (B, T, 1)
                else:
                    alpha_hal = None
            else:
                alpha_faith = STEER_ALPHA
                alpha_hal = STEER_ALPHA if hal_vec_b is not None else None

            if both and hal_vec_b is not None:
                x_new = x_new + alpha_faith * faith_vec_b
                x_new = x_new - alpha_hal * hal_vec_b
            elif hal and hal_vec_b is not None:
                x_new = x_new - alpha_hal * hal_vec_b
            elif faith:
                x_new = x_new + alpha_faith * faith_vec_b

            if isinstance(output, tuple):
                return (x_new,) + output[1:]
            else:
                return x_new

        with Trace(models[0].language_model.model.layers[layer - 1], edit_output=intervene):
            output = models[0].generate(**inputs, max_new_tokens=500)

        decoded = processor.tokenizer.batch_decode(output, skip_special_tokens=True)[0]

        if "assistant\n\n" in decoded:
            response = decoded.split("assistant\n\n", 1)[-1].strip()
        else:
            response = decoded.strip()
        print(response)
        results.append({"id": image_id, "response": response})

    active_opts = []
    if args.faith:
        active_opts.append("faith")
    if args.hal:
        active_opts.append("hal")
    if args.both:
        active_opts.append("both")
    if args.alpha_ablation:
        active_opts.append("alphaAblation")

    opt_str = "_".join(active_opts) if active_opts else "none"

    feature_idx_str = "-".join(str(i) for i in feature_idxs)
    k = len(feature_idxs)
    save_path = f"{args.save_dir}/layer_{layer}_{STEER_ALPHA}_{k}_{opt_str}_{args.substr}.jsonl"

    with open(save_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(
        f"CHAIR evaluation completed with features {feature_idxs} at layer {layer}."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steer-alpha", type=float, default=10.0)
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--save-dir", type=str, default="chair_result")
    parser.add_argument("--substr", type=str, default="lure")
    parser.add_argument("--hal", action="store_true")
    parser.add_argument("--faith", action="store_true")
    parser.add_argument("--both", action="store_true")
    parser.add_argument("--alpha-ablation", action="store_true")

    parser.add_argument(
        "--feature-idxs",
        type=int,
        nargs="+",
        default=None,
        help="Faith feature indices to steer (if not set, use layer_to_feature[layer])",
    )

    args = parser.parse_args()
    main(args)
