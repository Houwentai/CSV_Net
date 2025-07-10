import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class GCNModel(nn.Module):
    def __init__(self, num_node_features, num_classes, num_clinical_features, graph_hidden_dim, attention_hidden_dim):
        super(GCNModel, self).__init__()
        self.num_clinical_features = num_clinical_features
        
        # 图卷积层
        self.gcn1 = SAGEConv(num_node_features * 2, graph_hidden_dim)
        self.gcn2 = SAGEConv(graph_hidden_dim, 1)
        
        # 独立映射层
        self.clinical_linears = nn.ModuleList([
            nn.Linear(1, attention_hidden_dim) for _ in range(num_clinical_features)
        ])
        
        self.x_attn_linear = nn.Linear(1, attention_hidden_dim)
        
        # 独立注意力计算层
        self.score_linears = nn.ModuleList([
            nn.Linear(attention_hidden_dim * 2, 1) for _ in range(num_clinical_features)
        ])
        
        # 预测层
        self.fc1 = nn.Sequential(
            nn.Linear(num_clinical_features + graph_hidden_dim, num_classes),
            nn.Sigmoid()
        )

    def forward(self, data, clinical_features):
        # GCN前向传播
        x, edge_index = torch.cat([data.phenotypes, torch.abs(data.phenotypes-data.prototypes)], dim=1), data.edge_index
        num_nodes = x.shape[0]
        x = F.relu(self.gcn1(x, edge_index))
        
        # 节点注意力特征
        x_attn = self.gcn2(x, edge_index)  # [num_nodes, 1]
        x_attn = self.x_attn_linear(x_attn)  # [num_nodes, attention_hidden_dim]
        x_attn = x_attn.unsqueeze(1)  # [num_nodes, 1, attention_hidden_dim]
        
        # 临床特征处理
        clinical_attn = clinical_features.view(1, -1, 1).expand(num_nodes, -1, -1)  # [num_nodes, num_clinical_features, 1]
        
        # 独立注意力计算
        attention_weights_list = []
        for i in range(self.num_clinical_features):
            clinic_feat = clinical_attn[:, i:i+1, :]  # [num_nodes, 1, 1]
            clinic_attn = self.clinical_linears[i](clinic_feat)  # [num_nodes, 1, attention_hidden_dim]
            
            # 确保维度匹配
            current_x_attn = x_attn  # [num_nodes, 1, attention_hidden_dim]
            
            # 拼接特征
            combined = torch.cat([current_x_attn, clinic_attn], dim=-1)  # [num_nodes, 1, attention_hidden_dim*2]
            
            # 计算注意力分数
            score = self.score_linears[i](combined)  # [num_nodes, 1, 1]
            attention_weight = torch.sigmoid(score)
            attention_weights_list.append(attention_weight)
        
        # 合并注意力权重
        attention_weights_clinic = torch.cat(attention_weights_list, dim=1)  # [num_nodes, num_clinical_features, 1]
        
        # 全局注意力权重
        attention_weights_sum = F.softmax(
            attention_weights_clinic.sum(dim=1),  # [num_nodes, 1]
            dim=0
        ).transpose(0, 1)  # [num_nodes, 1, 1]
        
        
        # 池化
        x_pool = torch.matmul(attention_weights_sum, x)  # [1, graph_hidden_dim]
        

        # 将图特征与临床特征拼接
        combined_features = torch.cat((x_pool, clinical_features), dim=1)
        
        
        # 全连接层进行最终预测
        output = self.fc1(combined_features)
        
        return output, attention_weights_clinic
