import torch
import os
from safetensors.torch import load_file
import argparse
import json

def find_best_separation_features(
    save_dir: str,
    correct_dir: str,
    incorrect_dir: str,
    module_path: str,
    topk: int = 10
):
   
    correct_file = os.path.join(correct_dir, f"{module_path}/Rank0_final.safetensors")
    incorrect_file = os.path.join(incorrect_dir, f"{module_path}/Rank0_final.safetensors")

    correct_data = load_file(correct_file)
    incorrect_data = load_file(incorrect_file)

  
    correct_acts = correct_data["activations"]          # e.g. [51, 32768]
    incorrect_acts = incorrect_data["activations"]
    correct_acts = correct_acts[:incorrect_acts.shape[0]] 
    print("correct_acts shape:", correct_acts.shape)
    print("incorrect_acts shape:", incorrect_acts.shape)

    if correct_acts.dim() == 3:
        # [B, T, D]
        num_features = correct_acts.shape[2]
        correct_feats = (correct_acts > 0).nonzero(as_tuple=False)  # [N, 3]
        incorrect_feats = (incorrect_acts > 0).nonzero(as_tuple=False)

        correct_feat_indices = correct_feats[:, 2]
        incorrect_feat_indices = incorrect_feats[:, 2]

    elif correct_acts.dim() == 2:
        # [B, D]
        num_features = correct_acts.shape[1]
        correct_feats = (correct_acts > 0).nonzero(as_tuple=False)  # [N, 2]
        incorrect_feats = (incorrect_acts > 0).nonzero(as_tuple=False)

        correct_feat_indices = correct_feats[:, 1]
        incorrect_feat_indices = incorrect_feats[:, 1]

    else:
        raise ValueError(f"Unexpected activations shape: {correct_acts.shape}")

    correct_counts = torch.zeros(num_features, dtype=torch.float32)
    incorrect_counts = torch.zeros(num_features, dtype=torch.float32)

    for feat in correct_feat_indices:
        correct_counts[feat] += 1
    for feat in incorrect_feat_indices:
        incorrect_counts[feat] += 1

    correct_total = correct_acts.shape[0]
    incorrect_total = incorrect_acts.shape[0]

    p_correct = correct_counts / (correct_total + 1e-8)
    p_incorrect = incorrect_counts / (incorrect_total + 1e-8)

    separation_pos = p_correct - p_incorrect     
    separation_neg = p_incorrect - p_correct     

    topk_pos_scores, topk_pos_indices = torch.topk(separation_pos, topk)
    topk_neg_scores, topk_neg_indices = torch.topk(separation_neg, topk)

    print("🔝 Top features (correct > incorrect):")
    for rank, (feat, score) in enumerate(zip(topk_pos_indices, topk_pos_scores), 1):
        print(
            f"Top {rank}: feature {int(feat)}, "
            f"Δ={float(score):.4f}, "
            f"C={int(correct_counts[feat])}, I={int(incorrect_counts[feat])}, "
            f"p_c={float(p_correct[feat]):.4f}, p_i={float(p_incorrect[feat]):.4f}"
        )

    print("\n🔻 Bottom features (incorrect > correct):")
    for rank, (feat, score) in enumerate(zip(topk_neg_indices, topk_neg_scores), 1):
        print(
            f"Bottom {rank}: feature {int(feat)}, "
            f"Δ=-{float(score):.4f}, "
            f"C={int(correct_counts[feat])}, I={int(incorrect_counts[feat])}, "
            f"p_c={float(p_correct[feat]):.4f}, p_i={float(p_incorrect[feat]):.4f}"
        )

    result = {
            "correct_indices": topk_pos_indices.tolist(),
            "hallucinated_indices": topk_neg_indices.tolist()
        }

    save_path = os.path.join(save_dir, "best_separation_feature.json")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"💾 Saved indices to {save_path}")

    return (
        topk_pos_indices.tolist(),
        topk_pos_scores.tolist(),
        topk_neg_indices.tolist(),
        topk_neg_scores.tolist(),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", type=str, default="sae_cache")
    parser.add_argument("--layer", type=int, default=24)
    parser.add_argument("--substr", type=str, default="lure")
    parser.add_argument("--model", type=str, default="llava_1_6")
    
    args = parser.parse_args()

    save_dir = f"{args.save_dir}/{args.substr}_layer{args.layer}"
    correct_dir = f"{save_dir}/correct"
    incorrect_dir = f"{save_dir}/incorrect"
    module_path = f"model.layers.{args.layer}"

    find_best_separation_features(
        save_dir, correct_dir, incorrect_dir, module_path, topk=10
    )

