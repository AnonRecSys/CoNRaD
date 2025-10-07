# coding: utf-8
# @email: enoche.chow@gmail.com

import numpy as np
import os
import torch
import torch.nn as nn
import random
import time

from common.abstract_recommender import GeneralRecommender
import torch.nn.functional as F
from utils.conrad_utils import (
    sparse_mx_to_torch_sparse_tensor,
    slice_sparse_tensor_rows,
    sparse_mx_to_torch_tensor,
)
from utils.utils import build_sim, build_knn_normalized_graph
import scipy.sparse as sp


class CoNRaD_teacher(GeneralRecommender):
    def __init__(self, config, dataloader):
        super(CoNRaD_teacher, self).__init__(config, dataloader)

        # load parameters info
        self.emb_size = config["embedding_size"]
        self.hidden_size = config["hidden_size"]
        self.temp = config["tau"]
        self.t_weight = config["t_weight"]
        self.v_weight = 1 - self.t_weight

        self.dataloader = dataloader
        self.n_nodes = self.n_users + self.n_items
        self.interaction_matrix = dataloader.inter_matrix(form="csr").astype(np.float32)

        self.ui_adj_tensor = sparse_mx_to_torch_sparse_tensor(
            self.interaction_matrix
        ).cuda()

        self.v_layer = nn.Sequential(
            nn.Linear(4096, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size, track_running_stats=False),
            nn.ReLU(),
        )

        self.t_layer = nn.Sequential(
            nn.Linear(384, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size, track_running_stats=False),
            nn.ReLU(),
        )
        self.final_layer = nn.Linear(self.hidden_size, self.emb_size)

        # define layers and loss
        initializer = nn.init.xavier_uniform_
        if self.v_feat is not None:
            self.v_feat = F.normalize(self.v_feat)
        if self.t_feat is not None:
            self.t_feat = F.normalize(self.t_feat)

    def content_encoder(self, v_feat, t_feat):
        t_hidden = self.t_layer(t_feat)
        v_hidden = self.v_layer(v_feat)
        hidden_fused = self.v_weight * v_hidden + self.t_weight * t_hidden
        feats_out = self.final_layer(hidden_fused)
        return feats_out

    def forward(self, user_idx=None, item_idx=None):
        inter_mat_batch = self.interaction_matrix.copy()
        v_feat_in = self.v_feat
        t_feat_in = self.t_feat
        if user_idx is not None:
            inter_mat_batch = inter_mat_batch[user_idx.cpu().numpy()]

        inter_mat_batch_tensor = sparse_mx_to_torch_sparse_tensor(
            inter_mat_batch
        ).cuda()
        feats_out = self.content_encoder(v_feat_in, t_feat_in)
        user_vecs = torch.sparse.mm(inter_mat_batch_tensor, feats_out)
        return F.normalize(user_vecs), F.normalize(feats_out)

    def calculate_loss(self, interaction):
        """
        loss on one batch
        :param interaction:
            batch data format: tensor(3, batch_size)
            [0]: user list; [1]: positive items; [2]: negative items
        :return:
        """
        user_idx, pos_idx, neg_idx = interaction
        user_vecs, feats = self.forward(user_idx, pos_idx)
        cl_loss = self.InfoNCE(user_vecs, feats[pos_idx], self.temp)
        return cl_loss

    def InfoNCE(self, view1, view2, temperature: float, b_cos: bool = True):
        """
        Args:
            view1: (torch.Tensor - N x D)
            view2: (torch.Tensor - N x D)
            temperature: float
            b_cos (bool)

        Return: Average InfoNCE Loss
        """
        if b_cos:
            view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)

        pos_score = (view1 @ view2.T) / temperature
        score = torch.diag(F.log_softmax(pos_score, dim=1))
        return -score.mean()

    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_vecs, feats_out = self.forward()
        user_e = user_vecs[user, :]
        score = torch.matmul(user_e, feats_out.transpose(0, 1))
        return score

    def get_sparse_scores_knn(self, item_emb, batch_size, knn_k=50):
        num_batches = int(item_emb.shape[0] // batch_size) + 1
        inter_matrix_input = self.interaction_matrix
        knn_outs = []
        for i in range(num_batches):
            item_vec = item_emb[i * batch_size : (i + 1) * batch_size]
            item_vec_sim = item_vec @ item_emb.T
            item_vec_sim[:, i * batch_size :][
                np.diag_indices(item_vec_sim.shape[0])
            ] = 0
            knn_outs.append(torch.topk(item_vec_sim, knn_k, dim=-1))

        final_inds = torch.Tensor().cuda()
        final_vals = torch.Tensor().cuda()

        for i, (knn_val, knn_ind) in enumerate(knn_outs):
            batch_num = knn_val.shape[0]
            knn_sim = (
                torch.zeros((batch_num, item_emb.shape[0]))
                .cuda()
                .scatter_(-1, knn_ind[:, :knn_k], knn_val[:, :knn_k])
                .to_sparse_coo()
            )
            knn_sim.indices()[0] += i * batch_size
            final_inds = torch.cat([final_inds, knn_sim.indices()], dim=1)
            final_vals = torch.cat([final_vals, knn_sim.values()])

        final_sim = torch.sparse_coo_tensor(
            final_inds, final_vals, size=(item_emb.shape[0], item_emb.shape[0])
        )

        mm_out = torch.sparse.mm(self.ui_adj_tensor, final_sim.T)
        inter_matrix_batch_zero = sparse_mx_to_torch_sparse_tensor(
            inter_matrix_input * -1e8
        )
        mm_out_masked = (mm_out.cpu() + inter_matrix_batch_zero).coalesce()

        num_user_batches = int(mm_out_masked.shape[0] // batch_size) + 1
        top_out = []
        for j in range(num_user_batches):
            user_batch = slice_sparse_tensor_rows(
                mm_out_masked,
                j * batch_size,
                min((j + 1) * batch_size, mm_out_masked.shape[0]),
            ).to_dense()
            top_val, top_inds = torch.topk(user_batch, 50)
            top_out.append((top_val, top_inds))
        top_inds_knn, top_values_knn = [
            torch.cat(x).numpy() for x in reversed(list(zip(*top_out)))
        ]

        return top_inds_knn

    @torch.no_grad()
    def forward_pred(self):
        return [x.cpu() for x in self.forward()]

    @torch.no_grad()
    def create_teacher(self):
        start = time.time()
        _, feats_out = self.forward()
        teacher_inds = self.get_sparse_scores_knn(feats_out, batch_size=5000)
        print("teacher time", time.time() - start)
