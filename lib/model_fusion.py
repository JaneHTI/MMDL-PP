import os
import torch
import torch.nn as nn
from .model_clinic import ClinicClassifier
from .model_t1 import T1SepClassifier
from .model_conn import GATClassifier

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class FusionMRINetSep1(nn.Module):
    def __init__(self, t1_topk_ratio=0.1, t1_sv_dim=16, conn_dim=268, conn_hidden_dim=64, conn_heads=2, fusion_mode='all'):
        super().__init__()
        self.fusion_mode = fusion_mode

        # 根据模式确定要使用的模态
        self.use_clinic = fusion_mode in ['all']
        self.use_t1 = fusion_mode in ['all', 'mri_only', 't1_only']
        self.use_sc = fusion_mode in ['all', 'mri_only', 'sc_only']
        self.use_fc = fusion_mode in ['all', 'mri_only', 'fc_only']

        # clinical variables
        if self.use_clinic:
            self.model_clinic = ClinicClassifier()
            self.proj_clinic = nn.Linear(256, 64)

        # MRI features
        if self.use_t1:
            ## t1 - thick 68, area 68, volume 68, sub_volume 16
            self.model_t1 = T1SepClassifier(topk_ratio=t1_topk_ratio, sub_vol_dim=t1_sv_dim)
            self.proj_t1 = nn.Linear(256, 64)

        if self.use_sc:
            ## SC
            self.model_sc = GATClassifier(input_dim=conn_dim,
                                             hidden_dim=conn_hidden_dim,
                                             heads=conn_heads,
                                             output_dim=1)
            self.proj_sc = nn.Linear(conn_hidden_dim * conn_heads, 64)

        if self.use_fc:
            ## FC
            self.model_fc = GATClassifier(input_dim=conn_dim,
                                             hidden_dim=conn_hidden_dim,
                                             heads=conn_heads,
                                             output_dim=1)
            self.proj_fc = nn.Linear(conn_hidden_dim * conn_heads, 64)

        # 动态计算分类器的输入维度
        input_dim = 0
        if self.use_clinic:
            input_dim += 64
        if self.use_t1:
            input_dim += 64
        if self.use_sc:
            input_dim += 64
        if self.use_fc:
            input_dim += 64

        if input_dim > 0:
            self.classifier = nn.Sequential(
                nn.ReLU(),
                nn.Linear(input_dim, 1)
            )
        else:
            raise ValueError("At least one modality!")

    def forward(self, clinic, ct, ca, cv, sv, sc_graph, fc_graph, args):
        device = args.device
        data_name = args.data_name
        sub_name = args.sub_name

        features = []
        results = {}

        # STEP1: feature embeddings
        ## clinic variables
        if self.use_clinic:
            clinic_out, clinic_embed = self.model_clinic(clinic, data_name, sub_name, device)  # [batch, 1], [batch, 256]
            clinic_embed2 = self.proj_clinic(clinic_embed)  # [batch, 64]
            features.append(clinic_embed2)

        ## MRI-t1
        if self.use_t1:
            t1_results = self.model_t1(ct, ca, cv, sv)
            t1_embed = t1_results['embed']  # [batch, 1], [batch, 256]
            t1_embed2 = self.proj_t1(t1_embed)  # [batch, 64]
            features.append(t1_embed2)

        if self.use_sc:
            ## MRI-sc
            sc_out, sc_embed = self.model_sc(x=sc_graph.x,
                                             edge_index=sc_graph.edge_index,
                                             edge_weight=sc_graph.edge_attr,
                                             batch=sc_graph.batch)  # [batch, 1], [batch, 64*heads]
            sc_embed2 = self.proj_sc(sc_embed)  # [batch, 64]
            features.append(sc_embed2)

        if self.use_fc:
            ## MRI-fc
            fc_out, fc_embed = self.model_fc(x=fc_graph.x,
                                             edge_index=fc_graph.edge_index,
                                             edge_weight=fc_graph.edge_attr,
                                             batch=fc_graph.batch)  # [batch, 1], [batch, 64*heads]
            fc_embed2 = self.proj_fc(fc_embed)  # [batch, 64]
            features.append(fc_embed2)

        if features:
            fused_embed = torch.cat(features, dim=1)
            output = self.classifier(fused_embed)
        else:
            raise RuntimeError("Error")

        if self.use_t1:
            results = {
                'output': output,
                'fused_embed': fused_embed,
                'ct_topk_indices': t1_results['ct_topk_idx'],
                'ct_topk_weights': t1_results['ct_weights'],
                'ca_topk_indices': t1_results['ca_topk_idx'],
                'ca_topk_weights': t1_results['ca_weights'],
                'cv_topk_indices': t1_results['cv_topk_idx'],
                'cv_topk_weights': t1_results['cv_weights'],
                'sv_topk_indices': t1_results['sv_topk_idx'],
                'sv_topk_weights': t1_results['sv_weights']
            }
        else:
            results = {
                'output': output,
                'fused_embed': fused_embed
            }
        return results