
import torch
import pandas as pd
from torch_geometric.data import HeteroData

@torch.no_grad()
def recommend_and_decode(
    model,
    user_feat: list,
    travel_feat: list,
    visit_area_ids: list,
    index_to_visit_area_id: dict,
    visit_move_edge_index: torch.Tensor,
    visit_df: pd.DataFrame,
    hidden_dim: int = 64,
    top_k: int = 5
):
    model.eval()

    # 1. HeteroData 구성
    data = HeteroData()
    data['user'].x = torch.tensor(user_feat, dtype=torch.float).unsqueeze(0)
    data['travel'].x = torch.tensor(travel_feat, dtype=torch.float).unsqueeze(0)
    data['visit_area'].x = torch.zeros((len(visit_area_ids), hidden_dim), dtype=torch.float)
    data[('user', 'traveled', 'travel')].edge_index = torch.tensor([[0], [0]], dtype=torch.long)
    data[('visit_area', 'move_1', 'visit_area')].edge_index = visit_move_edge_index

    # 2. 모델 추론
    score = model(data.x_dict, data.edge_index_dict)  # [num_visit_area]
    topk = torch.topk(score, k=top_k)
    top_indices = topk.indices.tolist()
    top_scores = topk.values.tolist()

    # 3. ID 디코딩
    recommend_ids = [index_to_visit_area_id[i] for i in top_indices]
    score_df = pd.DataFrame({"VISIT_AREA_ID": recommend_ids, "SCORE": top_scores})

    # 4. 장소 정보와 병합
    decoded = visit_df[visit_df["VISIT_AREA_ID"].isin(recommend_ids)].copy()
    cols = ['VISIT_AREA_ID', 'VISIT_AREA_NM']
    cols += [c for c in ['LC_LCL_NM', 'LC_MCLS_NM', 'THEMA_NM'] if c in decoded.columns]
    decoded = decoded[cols].drop_duplicates("VISIT_AREA_ID")

    result = decoded.merge(score_df, on="VISIT_AREA_ID")
    return result.sort_values("SCORE", ascending=False).reset_index(drop=True)
