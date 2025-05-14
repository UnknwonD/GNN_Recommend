
import pandas as pd
import numpy as np
import torch
from datetime import datetime
import ast
from torch_geometric.data import HeteroData

def make_single_heterodata_sample(user_feat, travel_feat, label_vec, visit_area_count, visit_move_edge_index, hidden_dim=64):
    data = HeteroData()
    data['user'].x = torch.tensor(user_feat, dtype=torch.float).unsqueeze(0)
    data['travel'].x = torch.tensor(travel_feat, dtype=torch.float).unsqueeze(0)
    data['visit_area'].x = torch.zeros((visit_area_count, hidden_dim), dtype=torch.float)
    data[('user', 'traveled', 'travel')].edge_index = torch.tensor([[0], [0]], dtype=torch.long)
    data[('visit_area', 'move_1', 'visit_area')].edge_index = visit_move_edge_index
    data['visit_area'].y = torch.tensor(label_vec, dtype=torch.float)
    return data

def build_dataset():
    data_path = '../data/VL_csv/'
    user_df = pd.read_csv(data_path + "tn_traveller_master_여행객 Master_E_preprocessed.csv")
    travel_df = pd.read_csv(data_path + "tn_travel_여행_E_COST_cleaned_gnn.csv")
    visit_df = pd.read_csv(data_path + "tn_visit_area_info_방문지정보_Cleaned_E.csv")
    move_df = pd.read_csv(data_path + "tn_move_his_이동내역_E.csv")

    travel_feature_cols = travel_feature_cols = [
    'LODGOUT_COST', 'ACTIVITY_COST',
       'TOTAL_COST', 'DURATION', 'PURPOSE_1', 'PURPOSE_10', 'PURPOSE_11',
       'PURPOSE_12', 'PURPOSE_13', 'PURPOSE_2', 'PURPOSE_21', 'PURPOSE_22',
       'PURPOSE_23', 'PURPOSE_24', 'PURPOSE_25', 'PURPOSE_26', 'PURPOSE_27',
       'PURPOSE_28', 'PURPOSE_3', 'PURPOSE_4', 'PURPOSE_5', 'PURPOSE_6',
       'PURPOSE_7', 'PURPOSE_8', 'PURPOSE_9', 'MVMN_NM_ENC', 'age_ENC',
       'whowith_ENC', 'mission_ENC'
]
    travel_df[travel_feature_cols] = travel_df[travel_feature_cols].fillna(0).astype(np.float32)

    visit_area_ids = sorted(visit_df["VISIT_AREA_ID"].dropna().unique().astype(int))
    visit_area_id_to_index = {vid: i for i, vid in enumerate(visit_area_ids)}
    travel_to_visits = visit_df.groupby("TRAVEL_ID")["VISIT_AREA_ID"].apply(list).to_dict()

    # 이동내역 기반 visit_area index edge 구성
    # 이거 지금 모든 이동경로가 연결되어있음
    edge_set = set()
    for travel_id, group in move_df.groupby("TRAVEL_ID"):
        path = []
        for _, row in group.iterrows():
            sid = row["START_VISIT_AREA_ID"]
            eid = row["END_VISIT_AREA_ID"]
            if pd.notna(sid):
                path = [int(sid)]
            if pd.notna(eid):
                path.append(int(eid))
        for a, b in zip(path[:-1], path[1:]):
            if a in visit_area_id_to_index and b in visit_area_id_to_index:
                a_idx = visit_area_id_to_index[a]
                b_idx = visit_area_id_to_index[b]
                edge_set.add((a_idx, b_idx))
                edge_set.add((b_idx, a_idx))
                
    
    edge_visit_move_index = torch.tensor(list(edge_set), dtype=torch.long).T


    user_feature_cols = [
        'GENDER', 'EDU_NM', 'EDU_FNSH_SE', 'MARR_STTS', 'JOB_NM', 'HOUSE_INCOME',
        'TRAVEL_TERM', 'TRAVEL_LIKE_SIDO_1', 'TRAVEL_LIKE_SIDO_2', 'TRAVEL_LIKE_SIDO_3',
        'AGE_GRP', 'FAMILY_MEMB', 'TRAVEL_NUM', 'TRAVEL_COMPANIONS_NUM',
        'TRAVEL_STYL_1', 'TRAVEL_STYL_2', 'TRAVEL_STYL_3', 'TRAVEL_STYL_4',
        'TRAVEL_STYL_5', 'TRAVEL_STYL_6', 'TRAVEL_STYL_7', 'TRAVEL_STYL_8',
        'TRAVEL_MOTIVE_1', 'TRAVEL_MOTIVE_2', 'INCOME'
    ]

    merged_df = travel_df.merge(user_df, on="TRAVELER_ID", how="inner")
    user_features, travel_features, travel_labels, valid_travel_ids = [], [], [], []

    for _, row in merged_df.iterrows():
        travel_id = row["TRAVEL_ID"]
        if travel_id not in travel_to_visits:
            continue
        visit_ids = travel_to_visits[travel_id]
        visit_indices = [visit_area_id_to_index[vid] for vid in visit_ids if vid in visit_area_id_to_index]
        if not visit_indices:
            continue
        u_feat = row[user_feature_cols].values.astype(np.float32)
        t_feat = row[travel_feature_cols].values.astype(np.float32)
        label = np.zeros(len(visit_area_ids), dtype=np.float32)
        label[visit_indices] = 1.0
        user_features.append(u_feat)
        travel_features.append(t_feat)
        travel_labels.append(label)

    hetero_list = []
    for u, t, y in zip(user_features, travel_features, travel_labels):
        h = make_single_heterodata_sample(u, t, y, len(visit_area_ids), edge_visit_move_index)
        hetero_list.append(h)

    return hetero_list, visit_area_id_to_index, edge_visit_move_index


import torch
from torch_geometric.data import HeteroData

def build_travel_subgraph(travel_id: str,
                           travel_df: pd.DataFrame,
                           visit_df: pd.DataFrame,
                           move_df: pd.DataFrame,
                           user_id_map: dict,
                           travel_id_map: dict,
                           visit_id_map: dict,
                           user_features: torch.Tensor,
                           travel_features: torch.Tensor,
                           visit_area_dim: int,
                           travel_label_vectors: dict):
    # travel index
    if travel_id not in travel_id_map:
        return None
    t_idx = travel_id_map[travel_id]

    # 해당 travel의 row
    row = travel_df[travel_df["TRAVEL_ID"] == travel_id].iloc[0]
    traveler_id = row["TRAVELER_ID"]
    if traveler_id not in user_id_map:
        return None
    u_idx = user_id_map[traveler_id]

    # 1. 포함된 visit_area 추출
    visits = visit_df[visit_df["TRAVEL_ID"] == travel_id]["VISIT_AREA_ID"].tolist()
    visits = [vid for vid in visits if vid in visit_id_map]
    v_indices = [visit_id_map[vid] for vid in visits]

    # 2. 이동 edge (해당 travel_id만)
    move_group = move_df[move_df["TRAVEL_ID"] == travel_id]
    move_edges = [[], []]
    path = []
    for _, r in move_group.iterrows():
        sid = r["START_VISIT_AREA_ID"]
        eid = r["END_VISIT_AREA_ID"]
        if pd.notna(sid):
            path = [int(float(sid))]
        if pd.notna(eid):
            path.append(int(float(eid)))
    for a, b in zip(path[:-1], path[1:]):
        if a in visit_id_map and b in visit_id_map:
            if visit_id_map[a] in v_indices and visit_id_map[b] in v_indices:
                move_edges[0].append(v_indices.index(visit_id_map[a]))
                move_edges[1].append(v_indices.index(visit_id_map[b]))
    move_edge_index = torch.tensor(move_edges, dtype=torch.long) if move_edges[0] else torch.empty((2, 0), dtype=torch.long)

    # 3. HeteroData 구성
    data = HeteroData()

    # 노드 x
    data['user'].x = user_features[u_idx].unsqueeze(0)               # [1, user_dim]
    data['travel'].x = travel_features[t_idx].unsqueeze(0)           # [1, travel_dim]
    data['visit_area'].x = torch.zeros((len(v_indices), visit_area_dim))  # dummy

    # 엣지
    data[('user', 'traveled', 'travel')].edge_index = torch.tensor([[0], [0]], dtype=torch.long)
    data[('travel', 'contains', 'visit_area')].edge_index = torch.stack([
        torch.zeros(len(v_indices), dtype=torch.long),
        torch.arange(len(v_indices))
    ])

    data[('visit_area', 'move_1', 'visit_area')].edge_index = move_edge_index

    # 라벨 (multi-hot)
    full_label = travel_label_vectors[travel_id]
    label_mask = torch.tensor(v_indices)
    visit_label = full_label[label_mask]
    data['visit_area'].y = visit_label  # [num_local_visit]

    return data
