import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GATConv, GATv2Conv, global_mean_pool


class CustomGATConv(GATConv):
    def __init__(self, *args, **kwargs):
        super(CustomGATConv, self).__init__(*args, **kwargs)
        self.attention_weights = None

    def forward(self, x, edge_index, return_attention_weights=False):
        x, self.attention_weights = super(CustomGATConv, self).forward(x, edge_index, return_attention_weights=True)
        return x

class GATClassifier(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads=4):
        super(GATClassifier, self).__init__()
        self.conv1 = CustomGATConv(input_dim, hidden_dim, heads=heads, concat=True, add_self_loops=True)
        self.conv2 = CustomGATConv(hidden_dim * heads, hidden_dim, heads=heads, concat=True, add_self_loops=True)
        self.pool = global_mean_pool
        # self.bn1 = torch.nn.BatchNorm1d(hidden_dim * heads)
        # self.bn2 = torch.nn.BatchNorm1d(hidden_dim * heads)
        # self.dropout = torch.nn.Dropout(0.1)
        self.fc = torch.nn.Linear(hidden_dim * heads, output_dim)

    def forward(self, x, edge_index, edge_weight, batch):
        x = self.conv1(x, edge_index).relu()
        # x = self.bn1(x).relu()
        # x = self.dropout(x)

        x = self.conv2(x, edge_index).relu()
        # x = self.bn2(x).relu()
        # x = self.dropout(x)

        self.feature_maps = x
        x = self.pool(x, batch)
        output = self.fc(x)
        return output, x

    def get_attention_weights(self):
        return self.conv1.attention_weights, self.conv2.attention_weights

class CustomGATv2Conv(GATv2Conv):
    def __init__(self, *args, **kwargs):
        super(CustomGATv2Conv, self).__init__(*args, **kwargs)
        self.attention_weights = None

    def forward(self, x, edge_index, edge_weight, return_attention_weights=True):
        out, alpha = super(CustomGATv2Conv, self).forward(
            x, edge_index, edge_attr=edge_weight, return_attention_weights=True
        )
        self.attention_weights = (edge_index, alpha)
        return out

class GATv2Classifier(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads=4):
        super(GATv2Classifier, self).__init__()
        self.conv1 = CustomGATv2Conv(input_dim, hidden_dim, heads=heads, edge_dim=1, concat=True, add_self_loops=True)
        self.conv2 = CustomGATv2Conv(hidden_dim * heads, hidden_dim, heads=heads, edge_dim=1, concat=True, add_self_loops=True)
        self.pool = global_mean_pool
        # self.bn1 = torch.nn.BatchNorm1d(hidden_dim * heads)
        # self.bn2 = torch.nn.BatchNorm1d(hidden_dim * heads)
        # self.dropout = torch.nn.Dropout(0.1)
        self.fc = torch.nn.Linear(hidden_dim * heads, output_dim)

    def forward(self, x, edge_index, edge_weight, batch):
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        # x = self.bn1(x).relu()
        # x = self.dropout(x)

        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        # x = self.bn2(x).relu()
        # x = self.dropout(x)

        self.feature_maps = x
        x = self.pool(x, batch)
        output = self.fc(x)
        return output, x

    def get_attention_weights(self):
        return self.conv1.attention_weights, self.conv2.attention_weights