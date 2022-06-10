import torch
import torch.nn as nn
import torch.nn.functional as t_func
import torch_geometric as tg


class DiscriminatorNet(nn.Module):
    def __init__(self, num_node_features, num_hidden_features=32):
        super(DiscriminatorNet, self).__init__()
        self.num_node_features = num_node_features
        self.num_hidden_features = num_hidden_features
        # Define layers for embedding
        self.conv1 = tg.nn.GCNConv(self.num_node_features, self.num_hidden_features)
        self.conv2 = tg.nn.GCNConv(self.num_hidden_features, self.num_hidden_features)
        # Define layer for classification
        self.lin_class = torch.nn.Linear(self.num_hidden_features, 2)

    def forward(self, data):
        if isinstance(data, tg.data.Data):
            x, edge_index, batch = data.x.float(), data.edge_index, data.batch
        else:
            x, edge_index, batch = data
        # print('x is: {}'.format(x))
        # print('edge_index is: {}'.format(edge_index))
        # print('batch is: {}'.format(batch))
        # print('Finish discriminator!\n')
        # Apply Graph convolutions
        middle_out = t_func.relu(self.conv1(x, edge_index))
        middle_out = self.conv2(middle_out, edge_index)
        middle_out = tg.nn.global_mean_pool(middle_out, batch)
        middle_out = t_func.dropout(middle_out, p=0.2, training=self.training)
        # Classification
        out_class = self.lin_class(middle_out)
        # Return output
        return out_class
