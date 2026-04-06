"""
Dataset Loader for MSGL-Transformer
=====================================
Paper: MSGL-Transformer: A Multi-Scale Global-Local Transformer
       for Rodent Social Behavior Recognition
Authors: Muhammad Imran Sharif, Doina Caragea
Institution: Kansas State University

This file provides the dataset loader for CalMS21 Task 1.

---------------------------------------------------------------
FOR CalMS21 USERS:
---------------------------------------------------------------
- Input: JSON files (calms21_task1_train.json, calms21_task1_test.json)
- Keypoints shape: (num_frames, 2_mice, 2_xy, 7_keypoints) -> flattened to (num_frames, 28)
- Labels: 0=attack, 1=investigation, 2=mount, 3=other
- Download: https://data.caltech.edu/records/s0vdx-0k302
- Usage:
    dataset = CalMS21Dataset('calms21_task1_train.json', seq_len=35, fit_scaler=True)

---------------------------------------------------------------
FOR RatSI USERS:
---------------------------------------------------------------
- Input: Excel files (.xlsx), one per observation video
- Keypoints: 12D pose features (6 keypoints x 2 coordinates)
    Features: center_m_x, center_m_y, nose_m_x, nose_m_y, tail_m_x, tail_m_y,
              center_w_x, center_w_y, nose_w_x, nose_w_y, tail_w_x, tail_w_y
- Labels: Approaching, Following, Moving Away, Social Nose Contact, Solitary
- Download: https://mlorbach.gitlab.io/datasets/
- No separate dataset loader needed — load directly with pandas:
    import pandas as pd
    df = pd.read_excel('Observation01.xlsx')
    X = df[features].values
    y = label_encoder.fit_transform(df['action'].values)
- Use the same model with input_dim=12, num_classes=5:
    model = MSGLTransformer(input_dim=12, num_classes=5, seq_len=35)
---------------------------------------------------------------
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class CalMS21Dataset(Dataset):
    def __init__(self, json_path, seq_len=35, stride=1, scaler=None, fit_scaler=False):
        """
        Args:
            json_path:   Path to calms21_task1_train.json or calms21_task1_test.json
            seq_len:     Sliding window length (default 35, same as RatSI)
            stride:      Sliding window stride (default 1)
            scaler:      Pre-fitted StandardScaler (for test set)
            fit_scaler:  Whether to fit a new scaler (True for train set)
        """
        self.seq_len = seq_len
        self.stride  = stride

        print(f"Loading CalMS21 data from: {json_path}")
        with open(json_path, 'r') as f:
            data = json.load(f)

        all_keypoints       = []
        all_labels          = []
        sequence_boundaries = []

        label_map = {'attack': 0, 'investigation': 1, 'mount': 2, 'other': 3}

        for seq_id, seq_data in data['sequences'].items():
            keypoints = np.array(seq_data['keypoints'])  # (T, 2, 2, 7)
            labels    = seq_data['annotations']

            T = keypoints.shape[0]
            keypoints_flat = keypoints.reshape(T, -1)  # (T, 28)
            int_labels = [label_map.get(l, 3) for l in labels]

            start_idx = len(all_keypoints)
            all_keypoints.extend(keypoints_flat)
            all_labels.extend(int_labels)
            end_idx = len(all_keypoints)
            sequence_boundaries.append((start_idx, end_idx))

        self.all_keypoints = np.array(all_keypoints, dtype=np.float32)
        self.all_labels    = np.array(all_labels,    dtype=np.int64)
        print(f"Total frames: {len(self.all_keypoints)}")

        # Normalize
        if fit_scaler:
            self.scaler = StandardScaler()
            self.all_keypoints = self.scaler.fit_transform(self.all_keypoints)
        elif scaler is not None:
            self.scaler = scaler
            self.all_keypoints = self.scaler.transform(self.all_keypoints)
        else:
            self.scaler = None

        # Create sliding window samples
        self.samples = []
        for start_idx, end_idx in sequence_boundaries:
            num_frames = end_idx - start_idx
            for i in range(0, num_frames - seq_len, stride):
                global_start = start_idx + i
                label = self.all_labels[global_start + seq_len - 1]
                self.samples.append((global_start, label))

        print(f"Total samples: {len(self.samples)}")

        # Class distribution
        labels_array = np.array([s[1] for s in self.samples])
        unique, counts = np.unique(labels_array, return_counts=True)
        label_names = {0: 'attack', 1: 'investigation', 2: 'mount', 3: 'other'}
        print("Class distribution:")
        for u, c in zip(unique, counts):
            print(f"  {label_names.get(u, f'class_{u}')} ({u}): {c} ({100*c/len(self.samples):.1f}%)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        start, label = self.samples[idx]
        sequence = torch.FloatTensor(self.all_keypoints[start:start + self.seq_len])
        label    = torch.LongTensor([label]).squeeze()
        return sequence, label

    def get_class_weights(self):
        """Compute inverse frequency class weights for handling class imbalance."""
        labels        = np.array([s[1] for s in self.samples])
        unique, counts = np.unique(labels, return_counts=True)
        total         = len(labels)
        weights       = total / (len(unique) * counts)
        weight_tensor = torch.zeros(max(unique) + 1)
        for u, w in zip(unique, weights):
            weight_tensor[u] = w
        return weight_tensor
