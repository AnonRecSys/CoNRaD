# coding: utf-8
# @email: enoche.chow@gmail.com

import numpy as np
import os
import torch
import torch.nn as nn
import random
import time

from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss
from common.init import xavier_normal_initialization
import torch.nn.functional as F
from utils.conrad_utils import (
    masked_gather_padded_vectorized,
    filter_and_remap_topk,
    l2_reg_loss,
    bpr_loss,
)
from collections import defaultdict
import scipy.sparse as sp


class CoNRaD(GeneralRecommender):
    def __init__(self, config, dataloader):
        super(CoNRaD, self).__init__(config, dataloader)
        # load parameters info
        self.emb_size = config["embedding_size"]
        self.alpha = config["alpha"]
        self.top_k = config["teacher_top_k"]
        self.distil_weight = config["lambda"]
        self.temp = config["tau"]
        self.feat_weight = config["training_beta"]
        self.inference_betas = config["inference_betas"]
        self.t_weight = config["t_weight"]
        self.v_weight = 1 - self.t_weight
        self.cf_backbone = config["cf_backbone"]
        self.n_gcn_layers = config["n_gcn_layers"]
        self.total_item_num = config["total_item_num"]
        self.emb_reg = config["emb_reg"]

        self.best_inference_beta = self.inference_betas[0]

        self.dataloader = dataloader
        self.n_nodes = self.n_users + self.n_items
        self.interaction_matrix = dataloader.inter_matrix(form="coo").astype(np.float32)
        self.interaction_matrix_csr = dataloader.inter_matrix(form="csr").astype(
            np.float32
        )

        self.top_per_user = config["top_per_user"]
        self.pos_per_user = config["pos_per_user"]
        dataset_path = os.path.abspath(config["data_path"] + config["dataset"])

        teacher_file = os.path.join(dataset_path, "%s.pt" % config["teacher_file"])

        v_feat_file_path_full = os.path.join(
            dataset_path, config["vision_feature_file"].replace("mapped", "mapped_full")
        )
        t_feat_file_path_full = os.path.join(
            dataset_path, config["text_feature_file"].replace("mapped", "mapped_full")
        )
        if os.path.isfile(v_feat_file_path_full):
            self.v_feat_full = (
                torch.from_numpy(np.load(v_feat_file_path_full, allow_pickle=True))
                .type(torch.FloatTensor)
                .to(self.device)
            )
        if os.path.isfile(t_feat_file_path_full):
            self.t_feat_full = (
                torch.from_numpy(np.load(t_feat_file_path_full, allow_pickle=True))
                .type(torch.FloatTensor)
                .to(self.device)
            )

        self.sparse_norm_adj = self.get_norm_adj_mat().to(self.device)
        self.teacher = torch.load(teacher_file).to(self.device)

        self.v_layer = nn.Sequential(
            nn.Linear(4096, 192),
            nn.BatchNorm1d(192, track_running_stats=False),
            nn.ReLU(),
        )

        self.t_layer = nn.Sequential(
            nn.Linear(384, 192),
            nn.BatchNorm1d(192, track_running_stats=False),
            nn.ReLU(),
        )
        self.final_layer = nn.Linear(192, self.emb_size)

        # define layers and loss
        initializer = nn.init.xavier_uniform_
        self.embedding_dict = nn.ParameterDict(
            {
                "user_emb": nn.Parameter(
                    initializer(torch.empty(self.n_users, self.emb_size))
                ),
                "item_emb": nn.Parameter(
                    initializer(torch.empty(self.n_items, self.emb_size))
                ),
            }
        )
        if self.v_feat is not None:
            self.v_feat = F.normalize(self.v_feat)
            self.v_feat_full = F.normalize(self.v_feat_full)
        if self.t_feat is not None:
            self.t_feat = F.normalize(self.t_feat)
            self.t_feat_full = F.normalize(self.t_feat_full)

        self.bpr_loss = BPRLoss()

    def content_encoder(self, v_feat, t_feat):
        t_hidden = self.t_layer(t_feat)
        v_hidden = self.v_layer(v_feat)
        hidden_fused = self.v_weight * v_hidden + self.t_weight * t_hidden
        feats_out = self.final_layer(hidden_fused)
        return F.normalize(feats_out)

    def cf_forward(self):
        user_vecs = self.embedding_dict["user_emb"]
        item_vecs = self.embedding_dict["item_emb"]
        if self.cf_backbone == "MF":
            return user_vecs, item_vecs
        elif self.cf_backbone == "LightGCN":
            ego_embeddings = torch.cat([user_vecs, item_vecs], 0)
            all_embeddings = [ego_embeddings]
            for k in range(self.n_gcn_layers):
                ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
                all_embeddings += [ego_embeddings]
            all_embeddings = torch.stack(all_embeddings, dim=1)
            all_embeddings = torch.mean(all_embeddings, dim=1)
            return all_embeddings[: self.n_users], all_embeddings[self.n_users :]
        else:
            raise Exception("CF backbone must be either MF or LightGCN")

    def forward(self):
        user_vecs, item_vecs = self.cf_forward()
        feats_out = self.content_encoder(self.v_feat, self.t_feat)
        return user_vecs, item_vecs, feats_out

    def rank_loss(self, batch_full_mat, pos_items, teacher_batch, batch_user_mask):
        # Loss adapted from HetComp: https://github.com/SeongKu-Kang/HetComp_WWW23
        S_pos = masked_gather_padded_vectorized(
            batch_full_mat,
            pos_items,
            torch.tensor(sorted(self.selected_items)),
            pad_value=-1000,
        )
        pos_mask = S_pos > -1000

        S_top = masked_gather_padded_vectorized(
            batch_full_mat,
            teacher_batch,
            torch.tensor(sorted(self.selected_items)),
            pad_value=-1000,
        )
        top_mask = S_top > -1000
        below2 = (batch_full_mat.exp() * (1 - batch_user_mask)).sum(
            1, keepdims=True
        ) - S_top.exp().sum(1, keepdims=True)

        above_pos = (S_pos * pos_mask).sum(1, keepdims=True)
        below_pos = S_pos.flip(-1).exp().cumsum(1)
        below_pos = (
            (torch.clamp((below_pos + below2) * pos_mask.flip(-1), 1e-5))
            .log()
            .sum(1, keepdims=True)
        )
        pos_KD_loss = -(above_pos - below_pos).mean()

        above_top = (S_top * top_mask).sum(1, keepdims=True)
        below_top = S_top.flip(-1).exp().cumsum(1)
        below_top = (
            (torch.clamp((below_top + below2) * top_mask.flip(-1), 1e-5))
            .log()
            .sum(1, keepdims=True)
        )

        top_KD_loss = -(
            above_top - below_top
        ).mean()  # - (above_top_sub - below_top_sub).sum()

        return self.alpha * pos_KD_loss + (1 - self.alpha) * top_KD_loss

    def get_positive_examples(self, user_idx, num_pos):
        l = self.dataloader.history_items_per_u[user_idx]
        if len(l) >= num_pos:
            return random.sample(l, num_pos)
        else:
            new_l = []
            while len(new_l) < num_pos:
                new_l += l
            return random.sample(new_l, num_pos)

    def calculate_loss(self, interaction):
        """
        loss on one batch
        :param interaction:
            batch data format: tensor(3, batch_size)
            [0]: user list; [1]: positive items; [2]: negative items
        :return:
        """
        user_idx_full, pos_idx, neg_idx = interaction
        user_idx = list(set(user_idx_full.cpu().numpy()))

        # Get teacher items
        teacher_batch = self.teacher[user_idx]
        teacher_batch_lst = teacher_batch[:, : self.top_k].tolist()
        teacher_sample = [
            random.sample(x[: self.top_k], self.top_per_user) for x in teacher_batch_lst
        ]
        teacher_sample = set([i for l in teacher_sample for i in l])

        # Get positive examples for L_O
        pos_items_lst = [
            self.get_positive_examples(u, self.pos_per_user) for u in user_idx
        ]
        pos_sample = set([i for l in pos_items_lst for i in l])

        # Get full set of batch items
        self.selected_items = pos_sample.union(teacher_sample)
        num_extra = min([self.total_item_num, self.n_items]) - len(self.selected_items)
        unused = set(list(range(self.n_items))).difference(self.selected_items)
        self.selected_items = self.selected_items.union(
            random.sample(list(unused), num_extra)
        )

        # Model outputs
        user_vecs_full, item_vecs, feats_full = self.forward()
        feats = feats_full[sorted(self.selected_items)]
        feats_fused = item_vecs + self.feat_weight * feats_full

        # BPR loss
        args = (
            user_vecs_full[user_idx_full],
            feats_fused[pos_idx],
            feats_fused[neg_idx],
        )
        bpr = bpr_loss(*args)
        emb_reg_loss = l2_reg_loss(self.emb_reg, *args)

        user_vecs = user_vecs_full[user_idx]

        # Ranking distillation loss
        self.selected_ind_map = {
            v: i for i, v in enumerate(sorted(self.selected_items))
        }
        pos_items = torch.from_numpy(np.array(pos_items_lst)).cuda()
        batch_interaction_mat = torch.from_numpy(
            self.interaction_matrix_csr[user_idx][
                :, sorted(self.selected_items)
            ].todense()
        ).cuda()
        batch_full_mat = torch.clamp((user_vecs @ feats.T) / self.temp, min=-40, max=40)
        rank_loss = self.rank_loss(
            batch_full_mat, pos_items, teacher_batch, batch_interaction_mat
        )

        total_loss = self.distil_weight * rank_loss + bpr + emb_reg_loss
        return total_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_embeddings, item_embeddings, feats_out = self.forward()
        user_e = user_embeddings[user, :]
        all_item_e = item_embeddings + self.beta * feats_out
        score = torch.matmul(user_e, all_item_e.transpose(0, 1))
        return score

    @torch.no_grad()
    def forward_pred(self):
        feats_out = self.content_encoder(self.v_feat_full, self.t_feat_full)
        user_embeddings, item_embeddings = self.cf_forward()
        return [x.cpu() for x in [user_embeddings, item_embeddings, feats_out]]

    def get_norm_adj_mat(self):
        A = sp.dok_matrix(
            (self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32
        )
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(
            zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz)
        )
        data_dict.update(
            dict(
                zip(
                    zip(inter_M_t.row + self.n_users, inter_M_t.col),
                    [1] * inter_M_t.nnz,
                )
            )
        )
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid Devide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)

        return torch.sparse.FloatTensor(
            i, data, torch.Size((self.n_nodes, self.n_nodes))
        )
