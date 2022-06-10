import torch
import torch.nn as nn
import torch.nn.functional as t_func
import torch_geometric as tg

class ValueNet(nn.Module):
    def __init__(self, num_node_features, num_hidden_features=32, is_reg=False):
        super(ValueNet, self).__init__()
        self.num_node_features = num_node_features
        self.num_hidden_features = num_hidden_features
        self.is_reg = is_reg
        # Define layers for embedding
        self.conv1 = tg.nn.GCNConv(self.num_node_features, self.num_hidden_features)
        self.conv2 = tg.nn.GCNConv(self.num_hidden_features, self.num_hidden_features*2)
        # Define global attention
        self.global_attention_pool = tg.nn.GlobalAttention(torch.nn.Linear(self.num_hidden_features*2, 1),
                                                           torch.nn.Linear(self.num_hidden_features*2, self.num_hidden_features*2))
        # Define layer for regression or classification
        self.outer = torch.nn.Linear(self.num_hidden_features*2, 1 if self.is_reg else 2)

    def forward(self, data):
        # Get input from data received
        if isinstance(data, tg.data.Data):
            x, edge_index, batch = data.x.float(), data.edge_index, data.batch
        else:
            x, edge_index, batch = data
        # Apply Graph convolutions
        middle_out = t_func.relu(self.conv1(x, edge_index))
        middle_out = self.conv2(middle_out, edge_index)
        # Global pooling
        middle_out = self.global_attention_pool(middle_out, batch)
        # middle_out = tg.nn.global_mean_pool(middle_out, batch)
        middle_out = t_func.dropout(middle_out, p=0.2, training=self.training)
        # Get output
        out = self.outer(middle_out)
        if self.is_reg:
            # Squeeze and clamp value
            out = out.squeeze().clamp(min=0.0, max=1.0)
        # Return output
        return out
