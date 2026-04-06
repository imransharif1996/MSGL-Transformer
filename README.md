# MSGL-Transformer

**MSGL-Transformer: A Multi-Scale Global-Local Transformer for Rodent Social Behavior Recognition**

Muhammad Imran Sharif, Doina Caragea  
Department of Computer Science, Kansas State University, Manhattan, Kansas, 66502, USA

---

## Overview

MSGL-Transformer is a lightweight transformer-based model for recognizing rodent social behaviors from pose-based temporal sequences. The model captures behavioral dynamics at multiple temporal scales through:

- **Short-range causal attention** — captures rapid motion cues (first ⌊T/2⌋ frames)
- **Medium-range causal attention** — captures gradual behavioral patterns (all T frames)
- **Global bidirectional attention** — captures long-range dependencies (all T+1 tokens)
- **Behavior-Aware Modulation (BAM)** — emphasizes behavior-relevant temporal features

---

## Datasets

### RatSI
- 5 behavior classes: Approaching, Following, Moving Away, Social Nose Contact, Solitary
- 12D pose input (6 keypoints x 2 coordinates)
- 9 cross-validation splits
- Available at: https://mlorbach.gitlab.io/datasets/

### CalMS21
- 4 behavior classes: Attack, Investigation, Mount, Other
- 28D pose input (7 keypoints x 2 mice x 2 coordinates)
- Official train/test split
- Available at: https://data.caltech.edu/records/s0vdx-0k302

---

## Results

### RatSI (9-fold Cross-Validation)

| Metric | Value |
|--------|-------|
| Mean Accuracy | 75.4% |
| Mean F1-Score | 0.745 |
| Best Split Accuracy | 81.48% |

### CalMS21 (Task 1)

| Metric | Value |
|--------|-------|
| Accuracy | 87.1% |
| F1-Score | 0.8745 |
| Attack F1 | 0.5829 |
| vs HSTWFormer | +10.7% |

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Usage

### For CalMS21

```bash
python train.py \
  --data_dir /path/to/calms21/task1_classic_classification \
  --output_dir ./results
```

### For RatSI

The same `model.py` architecture works for RatSI with different parameters:

```python
from model import MSGLTransformer

# RatSI: 12D input, 5 classes
model = MSGLTransformer(input_dim=12, num_classes=5, seq_len=35)

# CalMS21: 28D input, 4 classes
model = MSGLTransformer(input_dim=28, num_classes=4, seq_len=35)
```

---

## Model Architecture

| Parameter | Value |
|-----------|-------|
| Sequence length (T) | 35 |
| Model dimension (d) | 64 |
| Attention heads | 4 |
| Transformer layers | 2 |
| Feed-forward dim | 128 |
| Dropout | 0.2 |

---

## Computing Resources

Experiments were conducted on the Beocat High-Performance Computing cluster at Kansas State University using NVIDIA A100 GPU. We thank the Beocat team for providing computational resources.

## Citation

If you use this code, please cite our paper:

```
@article{sharif2026msgl,
  title={MSGL-Transformer: A Multi-Scale Global-Local Transformer for Rodent Social Behavior Recognition},
  author={Sharif, Muhammad Imran and Caragea, Doina},
  journal={Scientific Reports},
  year={2026}
}
```

---

## Acknowledgment

This research was partially sponsored by the Cognitive and Neurobiological Approaches to Plasticity (CNAP) Center of Biomedical Research Excellence (COBRE) of the National Institutes of Health (NIH) under grant number P20GM113109.
