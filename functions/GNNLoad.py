from typing import List, Dict
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv

class PpiKkoTwistGNN(nn.Module):  # 삐삐꼬는 GNN
    def __init__(self, metadata, user_input_dim, travel_input_dim, hidden_dim=128, num_layers=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Input Projections
        self.input_proj = nn.ModuleDict({
            'user': nn.Linear(user_input_dim, hidden_dim),
            'travel': nn.Linear(travel_input_dim, hidden_dim),
            'visit_area': nn.Identity()
        })

        # Deep HeteroConv Layers
        self.convs = nn.ModuleList([
            HeteroConv(
                {etype: SAGEConv((-1, -1), hidden_dim) for etype in metadata[1]},
                aggr='sum'
            ) for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.ModuleDict({ntype: nn.LayerNorm(hidden_dim) for ntype in metadata[0]})
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(0.35)

        # Multi-Expert System (location / preference / category)
        self.expert_location = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.expert_preference = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.expert_category = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Multihead Attention Gating: attend across experts
        self.attn_gate = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        self.attn_query = nn.Parameter(torch.randn(1, hidden_dim))

        self.final_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x_dict, edge_index_dict, feedback_mask=None):
        # 1. Input projection
        x_dict = {k: self.input_proj[k](v) if k in self.input_proj else v for k, v in x_dict.items()}

        # 2. Deep GNN layers with residuals
        for i in range(self.num_layers):
            h_dict = self.convs[i](x_dict, edge_index_dict)
            h_dict = {
                k: self.dropout(F.relu(self.norms[i][k](v))) + x_dict[k]
                for k, v in h_dict.items() if k in x_dict
            }
            x_dict = h_dict

        h_visit = x_dict['visit_area']  # [num_nodes, hidden_dim]

        # 3. Expert Predictions
        loc = self.expert_location(h_visit)         # [N, 1]
        pref = self.expert_preference(h_visit)      # [N, 1]
        cat = self.expert_category(h_visit)         # [N, 1]
        experts = torch.cat([loc, pref, cat], dim=1).unsqueeze(1)  # [N, 1, 3]

        # 4. Multi-head Attention Gating
        q = self.attn_query.expand(h_visit.size(0), -1).unsqueeze(1)  # [N, 1, H]
        attn_out, _ = self.attn_gate(q, h_visit.unsqueeze(1), h_visit.unsqueeze(1))  # [N, 1, H]
        final_score = self.final_proj(attn_out.squeeze(1)).squeeze(-1)  # [N]

        if feedback_mask is not None:
            final_score = final_score + feedback_mask

        return final_score

# 1. 모델 로더 함수
def load_model(model_path: str, metadata, input_dims=(25, 29)) -> nn.Module:
    model = PpiKkoTwistGNN(
        metadata=metadata,
        user_input_dim=input_dims[0],
        travel_input_dim=input_dims[1],
        hidden_dim=128,
        num_layers=8
    )
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

# 2. 추천 추론 함수
def recommend_from_input(model: nn.Module,
                         user_input: Dict[str, float],
                         travel_input: Dict[str, float],
                         base_data,
                         visit_area_id_map: Dict[str, int],
                         topk: int = 5) -> List[Dict[str, float]]:
    # 입력 변환
    user_tensor = torch.tensor([list(user_input.values())], dtype=torch.float)
    travel_tensor = torch.tensor([list(travel_input.values())], dtype=torch.float)

    # HeteroData 복사 및 입력 추가
    data = base_data.clone()
    data['user'].x = torch.cat([data['user'].x, user_tensor], dim=0)
    data['travel'].x = torch.cat([data['travel'].x, travel_tensor], dim=0)
    uid, tid = data['user'].x.size(0) - 1, data['travel'].x.size(0) - 1

    # 엣지 연결 (user ↔ travel)
    data[('user', 'traveled', 'travel')].edge_index = torch.cat([
        data[('user', 'traveled', 'travel')].edge_index,
        torch.tensor([[uid], [tid]], dtype=torch.long)
    ], dim=1)
    data[('travel', 'traveled_by', 'user')].edge_index = torch.cat([
        data[('travel', 'traveled_by', 'user')].edge_index,
        torch.tensor([[tid], [uid]], dtype=torch.long)
    ], dim=1)

    # 추론
    with torch.no_grad():
        scores = model(data.x_dict, data.edge_index_dict)
        k = min(topk, scores.size(0))
        topk_result = torch.topk(scores, k)
        indices = topk_result.indices.tolist()
        values = topk_result.values.tolist()

    # index → visit_area_id 변환
    index_to_id = {v: k for k, v in visit_area_id_map.items()}
    return [{"visit_area_id": index_to_id[i], "score": round(v, 4)} for i, v in zip(indices, values)]
