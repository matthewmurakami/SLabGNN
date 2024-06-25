# import torch
# import torch.nn.functional as F
# from torch_geometric.nn import Sequential, GraphConv, GATv2Conv, global_mean_pool
# from torch.nn import Linear, ReLU,CrossEntropyLoss


"""
Original except the hidden_chanels was 1

class GCN(torch.nn.Module):
    def __init__(self, num_node_features, output_dim):
        super().__init__()
        self.num_node_features = num_node_features
        self.output_dim = output_dim
        self.hidden_channels = num_node_features
        self.num_heads = 1

        self.conv_first = GATv2Conv(self.num_node_features, self.hidden_channels, heads=self.num_heads, concat=False, edge_dim=1)

        self.conv1 = GATv2Conv(self.hidden_channels * self.num_heads, self.hidden_channels, heads=self.num_heads, concat=False, edge_dim=1)

        self.conv_last = GATv2Conv(self.hidden_channels * self.num_heads, output_dim, heads=1, concat=False, edge_dim=1)
        
        self.linear_layer = Linear(self.output_dim, self.output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv_first(x, edge_index)
        x = F.leaky_relu(x)
        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x)
        x = self.conv_last(x, edge_index)
        x = global_mean_pool(x, batch)
        
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.linear_layer(x)
        
        return x
"""


"""
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch.nn import Linear, BatchNorm1d, Dropout

class GCN(torch.nn.Module):
    def __init__(self, num_node_features, output_dim, hidden_channels=64, num_heads=4):
        super().__init__()
        self.num_node_features = num_node_features
        self.output_dim = output_dim
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads

        self.conv_first = GATv2Conv(self.num_node_features, self.hidden_channels, heads=self.num_heads, concat=False, edge_dim=1)
        self.norm1 = BatchNorm1d(self.hidden_channels)
        
        self.conv1 = GATv2Conv(self.hidden_channels, self.hidden_channels, heads=self.num_heads, concat=False, edge_dim=1)
        self.norm2 = BatchNorm1d(self.hidden_channels)
        
        self.conv_last = GATv2Conv(self.hidden_channels, self.output_dim, heads=1, concat=False, edge_dim=1)
        self.norm3 = BatchNorm1d(self.output_dim)
        
        self.dropout = Dropout(p=0.5)
        self.linear_layer = Linear(self.output_dim, self.output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv_first(x, edge_index)
        x = self.norm1(x)
        x = F.leaky_relu(x)

        x = self.conv1(x, edge_index)
        x = self.norm2(x)
        x = F.leaky_relu(x)

        x = self.conv_last(x, edge_index)
        x = self.norm3(x)
        
        x = global_mean_pool(x, batch)
        
        x = self.dropout(x)
        x = self.linear_layer(x)
        
        return x
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, BatchNorm, JumpingKnowledge
from torch.nn import Linear, Dropout

class GCN(torch.nn.Module):
    def __init__(self, num_node_features, output_dim, hidden_channels=128, num_heads=8):
        super().__init__()
        self.num_node_features = num_node_features
        self.output_dim = output_dim
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads

        self.conv1 = GATv2Conv(self.num_node_features, self.hidden_channels, heads=self.num_heads, concat=True, edge_dim=1)
        self.norm1 = BatchNorm(self.hidden_channels * self.num_heads)

        self.conv2 = GATv2Conv(self.hidden_channels * self.num_heads, self.hidden_channels, heads=self.num_heads, concat=True, edge_dim=1)
        self.norm2 = BatchNorm(self.hidden_channels * self.num_heads)
        
        self.conv3 = GATv2Conv(self.hidden_channels * self.num_heads, self.output_dim, heads=1, concat=False, edge_dim=1)
        self.norm3 = BatchNorm(self.output_dim)
        
        self.dropout = Dropout(p=0.5)

        self.linear1 = Linear(self.output_dim, self.hidden_channels)
        self.linear2 = Linear(self.hidden_channels, self.output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = self.norm1(x)
        x = F.leaky_relu(x)

        x = self.conv2(x, edge_index)
        x = self.norm2(x)
        x = F.leaky_relu(x)

        x = self.conv3(x, edge_index)
        x = self.norm3(x)
        x = F.leaky_relu(x)

        x = global_mean_pool(x, batch)

        x = F.leaky_relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)

        return x


"""
MIDDLE
class GCN(torch.nn.Module):
    def __init__(self, num_node_features, output_dim, hidden_channels=128, num_heads=8):
        super().__init__()
        self.num_node_features = num_node_features
        self.output_dim = output_dim
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads

        self.conv1 = GATv2Conv(self.num_node_features, self.hidden_channels, heads=self.num_heads, concat=True, edge_dim=1)
        self.norm1 = BatchNorm(self.hidden_channels * self.num_heads)

        self.conv2 = GATv2Conv(self.hidden_channels * self.num_heads, self.hidden_channels, heads=self.num_heads, concat=True, edge_dim=1)
        self.norm2 = BatchNorm(self.hidden_channels * self.num_heads)
        
        self.conv3 = GATv2Conv(self.hidden_channels * self.num_heads, self.output_dim, heads=1, concat=False, edge_dim=1)
        self.norm3 = BatchNorm(self.output_dim)
        
        self.dropout = Dropout(p=0.5)

        self.linear1 = Linear(self.output_dim, self.hidden_channels)
        self.linear2 = Linear(self.hidden_channels, self.output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = self.norm1(x)
        x = F.leaky_relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = self.norm2(x)
        x = F.leaky_relu(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index)
        x = self.norm3(x)
        x = F.leaky_relu(x)
        x = self.dropout(x)

        x = global_mean_pool(x, batch)

        x = F.leaky_relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)

        return x
"""

"""
SIMPLER
class GCN(torch.nn.Module):
    def __init__(self, num_node_features, output_dim, hidden_channels=128, num_heads=8):
        super().__init__()
        self.num_node_features = num_node_features
        self.output_dim = output_dim
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads

        self.conv = GATv2Conv(self.num_node_features, self.hidden_channels, heads=self.num_heads, concat=True, edge_dim=1)
        self.norm = BatchNorm(self.hidden_channels * self.num_heads)
        
        self.dropout = Dropout(p=0.5)

        self.linear1 = Linear(self.hidden_channels * self.num_heads, self.hidden_channels)
        self.linear2 = Linear(self.hidden_channels, self.output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv(x, edge_index)
        x = self.norm(x)
        x = F.leaky_relu(x)
        x = self.dropout(x)

        x = global_mean_pool(x, batch)
        
        x = F.leaky_relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)

        return x"""

"""
OP MODEL
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, BatchNorm, JumpingKnowledge
from torch.nn import Linear, Dropout


class GCN(torch.nn.Module):
    def __init__(self, num_node_features, output_dim, hidden_channels=128, num_heads=8, num_layers=5):
        super().__init__()
        self.num_node_features = num_node_features
        self.output_dim = output_dim
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()

        # Input layer
        self.convs.append(GATv2Conv(self.num_node_features, self.hidden_channels, heads=self.num_heads, concat=True, edge_dim=1))
        self.norms.append(BatchNorm(self.hidden_channels * self.num_heads))

        # Hidden layers
        for _ in range(self.num_layers - 2):
            self.convs.append(GATv2Conv(self.hidden_channels * self.num_heads, self.hidden_channels, heads=self.num_heads, concat=True, edge_dim=1))
            self.norms.append(BatchNorm(self.hidden_channels * self.num_heads))

        # Output layer
        self.convs.append(GATv2Conv(self.hidden_channels * self.num_heads, self.output_dim, heads=1, concat=False, edge_dim=1))
        self.norms.append(BatchNorm(self.output_dim))

        self.jk = JumpingKnowledge("cat")

        # Adjust the input size of the first linear layer
        self.linear1 = Linear(self.hidden_channels * self.num_heads * (self.num_layers - 1) + self.output_dim, self.hidden_channels)
        self.linear2 = Linear(self.hidden_channels, self.output_dim)

        self.dropout = Dropout(p=0.5)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        out = []
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.norms[i](x)
            x = F.leaky_relu(x)
            x = self.dropout(x)
            out.append(x)

        x = self.jk(out)

        x = global_mean_pool(x, batch)
        
        x = F.leaky_relu(self.linear1(x))
        
        x = self.dropout(x)
        x = self.linear2(x)

        return x
"""