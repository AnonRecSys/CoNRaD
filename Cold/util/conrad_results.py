import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import copy
import os
import pandas as pd
import sys
import pickle

from util.cold_deb_databuilder import ColdDebDataBuilder
from util.evaluator import ranking_evaluation
from util.operator import find_k_largest
from util.loader import DataLoader
from util.utils import sparse_mx_to_torch_tensor, sparse_mx_to_torch_sparse_tensor


col_titles = [
    "ID Recall",
    "ID NDCG",
    "Deb Recall",
    "Deb NDCG",
    "Overall Recall",
    "Overall NDCG",
    "Cold Recall",
    "Cold NDCG",
]

topk = 20


def get_data(dataset):
    feat_filenames = sorted(os.listdir(f"./data/{dataset}/cold_deb/feats"))
    feat_files = [f"./data/{dataset}/cold_deb/feats/{f}" for f in feat_filenames]
    item_content = [torch.from_numpy(np.load(f).astype(np.float32)) for f in feat_files]
    training_data = DataLoader.load_data_set(
        f"./data/{dataset}/cold_deb/warm_train.csv"
    )
    all_valid_data = DataLoader.load_data_set(
        f"./data/{dataset}/cold_deb/overall_val.csv"
    )
    standard_valid_data = DataLoader.load_data_set(
        f"./data/{dataset}/cold_deb/standard_val.csv"
    )
    deb_valid_data = DataLoader.load_data_set(f"./data/{dataset}/cold_deb/deb_val.csv")
    cold_valid_data = DataLoader.load_data_set(
        f"./data/{dataset}/cold_deb/cold_val.csv"
    )
    all_test_data = DataLoader.load_data_set(
        f"./data/{dataset}/cold_deb/overall_test.csv"
    )
    standard_test_data = DataLoader.load_data_set(
        f"./data/{dataset}/cold_deb/standard_test.csv"
    )
    deb_test_data = DataLoader.load_data_set(f"./data/{dataset}/cold_deb/deb_test.csv")
    cold_test_data = DataLoader.load_data_set(
        f"./data/{dataset}/cold_deb/cold_test.csv"
    )

    cold_items = set([p[1] for p in cold_valid_data]).union(
        set([p[1] for p in cold_test_data])
    )
    user_idx = sorted(set([p[0] for p in training_data]))
    item_idx = sorted(set([p[1] for p in training_data]).union(cold_items))
    user_num = len(user_idx)
    item_num = len(item_idx)
    data = ColdDebDataBuilder(
        training_data,
        standard_valid_data,
        deb_valid_data,
        all_valid_data,
        cold_valid_data,
        standard_test_data,
        deb_test_data,
        all_test_data,
        cold_test_data,
        user_num,
        item_num,
        user_idx,
        item_idx,
        None,
        item_content,
    )
    return data, item_content


def get_rec_list(u, top_inds, top_values_final, item_inds):
    u_mapped = u
    items = top_inds[u_mapped]
    items_mapped = [item_inds[iid] for iid in items]
    return list(zip(items_mapped, top_values_final[u_mapped]))


def get_rec_list_full(test_set, top_inds, top_values_final, item_inds):
    rec_list = {}
    for i, user in enumerate(test_set):
        rec_list[user] = get_rec_list(user, top_inds, top_values_final, item_inds)
    return rec_list


@torch.no_grad()
def eval_test(data, user_emb, item_emb, batch_size=5000, topk=20, cold=False):
    warm_item_emb = item_emb[data.warm_item_list]
    if cold:
        cold_item_emb = item_emb[data.cold_item_list]

    num_batches = 1 + data.user_num // batch_size
    warm_out = []
    cold_out = []
    for b in range(num_batches):
        user_emb_batch = user_emb[b * batch_size : (b + 1) * batch_size]
        warm_scores = user_emb_batch @ warm_item_emb.T
        mask = sparse_mx_to_torch_sparse_tensor(
            data.interaction_mat[b * batch_size : (b + 1) * batch_size][
                :, data.warm_item_list
            ]
        ).cuda()
        warm_scores += -1e10 * mask
        warm_out.append(torch.topk(warm_scores, topk))
        if cold:
            cold_scores = user_emb_batch @ cold_item_emb.T
            cold_out.append(torch.topk(cold_scores, topk))

    warm_vals, warm_inds = [torch.cat(x) for x in list(zip(*warm_out))]
    warm_rec_list = get_rec_list_full(
        data.overall_test_set, warm_inds, warm_vals, data.warm_item_list
    )

    overall_out = ranking_evaluation(data.overall_test_set, warm_rec_list, [topk])[1][0]
    deb_out = ranking_evaluation(
        data.deb_test_set,
        {u: v for u, v in warm_rec_list.items() if u in data.deb_test_set},
        [topk],
    )[1][0]
    standard_out = ranking_evaluation(
        data.standard_test_set,
        {u: v for u, v in warm_rec_list.items() if u in data.standard_test_set},
        [topk],
    )[1][0]

    if cold:
        cold_vals, cold_inds = [torch.cat(x) for x in list(zip(*cold_out))]
        cold_rec_list = get_rec_list_full(
            data.cold_test_set, cold_inds, cold_vals, data.cold_item_list
        )
        cold_out = ranking_evaluation(data.cold_test_set, cold_rec_list, [topk])[1][0]
        return pd.DataFrame(
            [standard_out + deb_out + overall_out + cold_out], columns=col_titles
        )
    return pd.DataFrame([standard_out + deb_out + overall_out], columns=col_titles[:-2])


beta_dict = {
    "baby": {
        "MF_raw": 2.5,
        "LightGCN_raw": 9,
        "MF_384_192_50": 3.5,
        "LightGCN_384_192_50": 8,
    },
    "clothing": {
        "MF_raw": 4.5,
        "LightGCN_raw": 8,
        "MF_384_192_50": 6.5,
        "LightGCN_384_192_50": 5.5,
    },
    "electronics": {
        "MF_raw": 6,
        "LightGCN_raw": 52,
        "MF_384_192_50": 3,
        "LightGCN_384_192_50": 12,
    },
    "sports": {
        "MF_raw": 3.5,
        "LightGCN_raw": 17.5,
        "MF_384_192_50": 3.5,
        "LightGCN_384_192_50": 5.5,
    },
}

for dataset in ["baby", "clothing", "electronics", "sports"]:
    data, _ = get_data(dataset)
    overall_metrics = {}
    for teacher in ["raw", "enc"]:
        teacher_metrics = {}
        for backbone in ["MF", "LightGCN"]:
            print(teacher, backbone)
            conrad_outs = []
            w = beta_dict[dataset]["%s_%s" % (backbone, teacher)]
            for seed in [111, 333, 555, 777, 999]:
                conrad_u_emb, conrad_i_emb, feats_out = torch.load(
                    "./saved_embs_conrad/%s_cold_deb/CoNRaD_%s_%s_%d.pt"
                    % (dataset, backbone, teacher, seed)
                )

                final_i_emb = conrad_i_emb + w * feats_out[data.warm_item_list]
                final_i_emb = torch.cat([final_i_emb, feats_out[data.cold_item_list]])
                conrad_outs.append(
                    eval_test(data, conrad_u_emb.cuda(), final_i_emb.cuda(), cold=True)
                )
                print(seed)
            conrad_results = pd.concat(conrad_outs)
            teacher_metrics[backbone] = conrad_results.mean(axis=0)
        overall_metrics[teacher] = teacher_metrics
    with open("../results/%s_conrad_results.pkl", "wb") as f:
        pickle.dump(overall_metrics, f)
