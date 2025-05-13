
import pandas as pd
import numpy as np
import torch
from datetime import datetime
import ast
from torch_geometric.data import HeteroData

def parse_travel_purpose(purpose_str):
    try:
        values = ast.literal_eval(purpose_str)
        return float(values[0]) if values else 0.0
    except:
        return 0.0

def compute_days(start_str, end_str):
    try:
        start = datetime.strptime(str(start_str), "%Y-%m-%d")
        end = datetime.strptime(str(end_str), "%Y-%m-%d")
        return (end - start).days
    except:
        return 0

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
    travel_df = pd.read_csv(data_path + "tn_travel_여행_E_COST_cleaned.csv")
    visit_df = pd.read_csv(data_path + "tn_visit_area_info_방문지정보_Cleaned_E.csv")
    move_df = pd.read_csv(data_path + "tn_move_his_이동내역_E.csv")

    travel_df["TRAVEL_PURPOSE"] = travel_df["TRAVEL_PURPOSE"].apply(parse_travel_purpose)
    travel_df["TRAVEL_DAYS"] = travel_df.apply(
        lambda row: compute_days(row["TRAVEL_START_YMD"], row["TRAVEL_END_YMD"]), axis=1
    )
    travel_feature_cols = ["TRAVEL_PURPOSE", "TRAVEL_DAYS", "LODGOUT_COST", "ACTIVITY_COST", "TOTAL_COST"]
    travel_df[travel_feature_cols] = travel_df[travel_feature_cols].fillna(0).astype(np.float32)

    visit_area_ids = sorted(visit_df["VISIT_AREA_ID"].dropna().unique().astype(int))
    visit_area_id_to_index = {vid: i for i, vid in enumerate(visit_area_ids)}
    travel_to_visits = visit_df.groupby("TRAVEL_ID")["VISIT_AREA_ID"].apply(list).to_dict()

    # 이동내역 기반 visit_area index edge 구성
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
