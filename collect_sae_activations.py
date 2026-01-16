import os
import torch
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch.distributed as dist
from safetensors.torch import save_file
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
from typing import Dict, Sequence
import torch.nn as nn
import argparse

class LureDataset(Dataset):
    def __init__(self, pope_path, data_path, trans):
        self.pope_path = pope_path
        self.data_path = data_path
        # print(data_path)
        self.trans = trans

        image_list, query_list, label_list, question_id = [], [], [], []


        for q in open(pope_path, 'r'):
            line = json.loads(q)
            image_list.append(line['image'])
            query_list.append(line['text'])
            label_list.append(line['label'])
            question_id.append(line['question_id'])

        # for i in range(len(label_list)):
        #     if label_list[i] == 'no':
        #         label_list[i] = 0
        #     else:
        #         label_list[i] = 1

        assert len(image_list) == len(query_list)
        assert len(image_list) == len(label_list)

        self.image_list = image_list
        self.query_list = query_list
        self.label_list = label_list
        self.question_id = question_id

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, index):
        image_path = os.path.join(self.data_path, self.image_list[index])
        raw_image = Image.open(image_path).convert("RGB")
        # image = self.trans(raw_image)
        image = raw_image
        query = self.query_list[index]
        label = self.label_list[index]
        question_id = self.question_id[index]
        return {"image": image, "query": query, "label": label, "question_id": question_id}

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
        
        
class FeatureAnswerCache:
    def __init__(
        self,
        model: LlavaNextForConditionalGeneration,
        processor: LlavaNextProcessor,
        submodule_dict: Dict[str, SAE],
        batch_size: int,
        shard_size: int,
        save_dir: str,
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
                    print(pred_label)
                    gt_label = labels[0]
                    print(gt_label)
                    is_correct = (pred_label.lower() == gt_label.lower())
                    print(is_correct)
                    topk=256
                    for module_path, latents in buffer.items():
                        sae = self.submodule_dict[module_path]

                        # (1) SAE forward 
                        latents = latents.float()  
                        _, features = sae(latents)  

                        # (2) Top-k sparse activation
                        topk = 256  
                        values, indices = torch.topk(features, k=topk, dim=-1)  # shape: (B, k)
                        sparse_features = torch.zeros_like(features)
                        sparse_features.scatter_(-1, indices, values)  # shape: (B, d_sae)

                        # (3) Batch index 0 
                        result = sparse_features[0].cpu()
                        print(result.shape)
                        if is_correct:
                            self.correct_feature_locations[module_path].append(result)
                            self.correct_feature_activations[module_path].append(result)
                        else:
                            self.incorrect_feature_locations[module_path].append(result)
                            self.incorrect_feature_activations[module_path].append(result)

                        if (batch_number + 1) % save_every == 0:
                            self._save_temp(module_path, batch_number)
                        print(len(self.correct_feature_locations[module_path]))
                        print(len(self.incorrect_feature_locations[module_path]))
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="llava-hf/llava-v1.6-mistral-7b-hf", help="Model name")
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--layer", type=int, required=True)

    args = parser.parse_args()

    processor = LlavaNextProcessor.from_pretrained(args.model_name)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        args.model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).to("cuda")
    lure_path = 'lure_for_gen.jsonl'
    dataset = LureDataset(
        lure_path=lure_path,
        data_path="train2014",
        trans=processor.image_processor,
    )

    layer=args.layer
    
    file = f"SAE/sae_final_layer{layer}_8x_norm.pkl"
    sae = SAE()
    sae = SAE().to("cuda")
    sae = sae.to("cuda")
    raw_state_dict = torch.load(file, map_location='cpu')
    clean_state_dict = remove_module_prefix(raw_state_dict)
    sae.load_state_dict(clean_state_dict)
    submodule_dict = {
        f"model.layers.{layer}": sae  
    }
    cacher = FeatureAnswerCache(
        model=model,
        processor=processor,
        submodule_dict=submodule_dict,
        batch_size=1,
        shard_size=0,
        save_dir=args.save_dir,
    )

    cacher.run(dataset)
