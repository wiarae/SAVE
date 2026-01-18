# SAVE

![SAVE framework overview](save.png)

**SAVE** is a comprehensive framework designed to **collect**, **identify**, and **steer** the internal features of **Multimodal LLMs** using **Sparse Autoencoders (SAEs)**.
By leveraging **SAVE**, researchers can pinpoint and manipulate specific neural features responsible for visual grounding and hallucinations, ultimately enhancing the reliability and interpretability of large-scale multimodal systems.

## 📋 Requirements

Install the required Python environment using Conda:

```bash
conda env create -f environment.yml
conda activate save
```
* **Model:** `llava-hf/llava-v1.6-mistral-7b-hf`
* **SAE Weights:** Download `.pkl` checkpoints for your target layers from [flybamboo/vlm-saes](https://huggingface.co/flybamboo/vlm-saes/tree/main) and place them in the `SAE/` directory.
* **Datasets:** `lure_for_gen.jsonl` (POPE), COCO images, and `chair/chair.jsonl` for evaluation.
   COCO Train 2014 & Val 2014 images

## 🚀 Step 1: Collect SAE activations

Extract Top-k sparse activations from the model. Features are automatically categorized into `correct/` and `incorrect/` folders based on the model's prediction accuracy.

```bash
python collect_sae_activations.py --layer 24 --save-dir "./sae_cache"

```
Results will be saved in: `./sae_cache/lure_layer24/`

---

## 🔍 Step 2: Identify Features

Identify the specific SAE features that distinguish correct responses from hallucinations. This script calculates the activation probability difference ().

```bash
python identify_features.py --layer 24 --save-dir "./sae_cache"

```

**Output:** Generates `best_separation_feature.json` containing:

* `correct_indices`: Features promoting reliable visual grounding.
* `hallucinated_indices`: Features correlated with hallucinations.

---

## ☸️ Step 3: Activation Steering (CHAIR Evaluation)

Apply "Steering" to the model's latent space to actively improve faithfulness or suppress hallucinations during long-form image description.

### Running the Steering Evaluation:

```bash
python chair_steer.py \
    --layer 24 \
    --steer-alpha 5.0 \
    --both \
    --save-dir "./chair_results"

```

### Key Steering Modes:

* `--faith`: Enhances "grounding" features (adds the  direction of correct features).
* `--hal`: Suppresses "hallucination" features (subtracts the  direction of hallucinated features).
* `--both`: Performs both enhancement and suppression simultaneously.

---

## 📂 Project Structure

```text
.
├── SAE/                     # Place downloaded .pkl files here
├── sae_cache/               # Extracted features (Step 1 & 2)
├── chair/                   # CHAIR evaluation dataset
├── collect_sae_activations.py        # Step 1 Script
├── identify_features.py      # Step 2 Script
└── chair_steer.py            # Step 3 Script

```

---

