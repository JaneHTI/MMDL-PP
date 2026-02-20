import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



class T1SepClassifier(nn.Module):
    def __init__(self, num_classes=1, topk_ratio=0.1, sub_vol_dim=16):
        super().__init__()

        self.thick_dim = 68
        self.area_dim = 68
        self.vol_dim = 68
        self.sub_vol_dim = sub_vol_dim

        self.thick_topk = int(np.ceil(self.thick_dim * topk_ratio))
        self.area_topk = int(np.ceil(self.area_dim * topk_ratio))
        self.vol_topk = int(np.ceil(self.vol_dim * topk_ratio))
        self.sub_vol_topk = int(np.ceil(self.sub_vol_dim * topk_ratio))

        self.thick_attention = nn.Sequential(
            nn.Linear(self.thick_dim, 32),
            nn.ReLU(),
            nn.Linear(32, self.thick_dim)
        )
        self.area_attention = nn.Sequential(
            nn.Linear(self.area_dim, 32),
            nn.ReLU(),
            nn.Linear(32, self.area_dim)
        )
        self.vol_attention = nn.Sequential(
            nn.Linear(self.vol_dim, 32),
            nn.ReLU(),
            nn.Linear(32, self.vol_dim)
        )
        self.sub_vol_attention = nn.Sequential(
            nn.Linear(self.sub_vol_dim, 8),
            nn.ReLU(),
            nn.Linear(8, self.sub_vol_dim)
        )

        total_topk = self.thick_topk + self.area_topk + self.vol_topk + self.sub_vol_topk
        self.fc1 = nn.Linear(total_topk, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, thick, area, vol, sub_vol):
        # thick
        thick_scores = self.thick_attention(thick)
        _, thick_topk_idx = torch.topk(thick_scores, k=self.thick_topk, dim=1)
        thick_selected = torch.gather(thick, 1, thick_topk_idx)
        thick_weights = F.softmax(torch.gather(thick_scores, 1, thick_topk_idx), dim=1)
        thick_weighted = thick_selected * thick_weights

        # area
        area_scores = self.area_attention(area)
        _, area_topk_idx = torch.topk(area_scores, k=self.area_topk, dim=1)
        area_selected = torch.gather(area, 1, area_topk_idx)
        area_weights = F.softmax(torch.gather(area_scores, 1, area_topk_idx), dim=1)
        area_weighted = area_selected * area_weights

        # vol
        vol_scores = self.vol_attention(vol)
        _, vol_topk_idx = torch.topk(vol_scores, k=self.vol_topk, dim=1)
        vol_selected = torch.gather(vol, 1, vol_topk_idx)
        vol_weights = F.softmax(torch.gather(vol_scores, 1, vol_topk_idx), dim=1)
        vol_weighted = vol_selected * vol_weights

        # sub_vol
        sub_vol_scores = self.sub_vol_attention(sub_vol)
        _, sub_vol_topk_idx = torch.topk(sub_vol_scores, k=self.sub_vol_topk, dim=1)
        sub_vol_selected = torch.gather(sub_vol, 1, sub_vol_topk_idx)
        sub_vol_weights = F.softmax(torch.gather(sub_vol_scores, 1, sub_vol_topk_idx), dim=1)
        sub_vol_weighted = sub_vol_selected * sub_vol_weights

        # concat
        features = torch.cat([thick_weighted, area_weighted, vol_weighted, sub_vol_weighted], dim=1)
        x1 = self.fc1(features)
        x = self.relu(x1)
        out = self.fc2(x)

        results = {
            'out': out,
            'embed': x1,
            'ct_topk_idx': thick_topk_idx,
            'ct_weights': thick_weights,
            'ca_topk_idx': area_topk_idx,
            'ca_weights': area_weights,
            'cv_topk_idx': vol_topk_idx,
            'cv_weights': vol_weights,
            'sv_topk_idx': sub_vol_topk_idx,
            'sv_weights': sub_vol_weights
        }
        return results