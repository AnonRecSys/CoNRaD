import numpy as np
from collections import defaultdict
import scipy.sparse as sp
import torch


class ColdDebDataBuilder(object):
    def __init__(
        self,
        training_data,
        standard_valid_data,
        deb_valid_data,
        overall_valid_data,
        cold_valid_data,
        standard_test_data,
        deb_test_data,
        overall_test_data,
        cold_test_data,
        user_num,
        item_num,
        user_idx,
        item_idx,
        user_content=None,
        item_content=None,
    ):
        super(ColdDebDataBuilder, self).__init__()
        self.training_data = training_data
        self.standard_valid_data = standard_valid_data
        self.standard_test_data = standard_test_data
        self.deb_valid_data = deb_valid_data
        self.deb_test_data = deb_test_data
        self.overall_valid_data = overall_valid_data
        self.overall_test_data = overall_test_data
        self.cold_valid_data = cold_valid_data
        self.cold_test_data = cold_test_data

        self.user = {u: u for u in range(user_num)}
        self.item = {u: u for u in range(item_num)}
        self.training_set_u = defaultdict(dict)
        self.training_set_i = defaultdict(dict)
        self.standard_valid_set = defaultdict(dict)
        self.standard_valid_set_item = set()
        self.deb_valid_set = defaultdict(dict)
        self.deb_valid_set_item = set()
        self.overall_valid_set = defaultdict(dict)
        self.overall_valid_set_item = set()
        self.cold_valid_set = defaultdict(dict)
        self.cold_valid_set_item = set()
        self.standard_test_set = defaultdict(dict)
        self.standard_test_set_item = set()
        self.deb_test_set = defaultdict(dict)
        self.deb_test_set_item = set()
        self.overall_test_set = defaultdict(dict)
        self.overall_test_set_item = set()
        self.cold_test_set = defaultdict(dict)
        self.cold_test_set_item = set()
        if user_content is not None:
            self.source_user_content = user_content
            self.mapped_user_content = np.empty(
                (user_content.shape[0], user_content.shape[1])
            )
            self.user_content_dim = user_content.shape[-1]
        if item_content is not None:
            self.item_content = item_content
            # self.item_content_dim = item_content.shape[-1]

        self.warm_item_list = set()

        self.generate_set()

        self.cold_item_list = sorted(
            self.cold_test_set_item.union(self.cold_valid_set_item)
        )
        self.warm_item_list = sorted(self.warm_item_list)

        self.user_num = user_num
        self.item_num = item_num
        # print(self.item_num, len(self.item.keys()))
        # raise Exception("debugging...")
        # PLEASE NOTE: the original and mapped index are different!
        self.user_idx = user_idx
        self.item_idx = item_idx
        # print(self.source_deb_item_idx, self.mapped_deb_item_idx)
        # raise Exception("debugging...")
        self.ui_adj = self.create_sparse_complete_bipartite_adjacency()
        self.norm_adj = self.normalize_graph_mat(self.ui_adj)
        self.interaction_mat = self.create_sparse_interaction_matrix()

    def generate_set(self):
        # training set building
        for entry in self.training_data:
            user, item, rating = entry
            self.training_set_u[user][item] = rating
            self.training_set_i[item][user] = rating
            self.warm_item_list.add(item)

        # standard validation set building
        for entry in self.standard_valid_data:
            user, item, rating = entry
            self.standard_valid_set[user][item] = rating
            self.standard_valid_set_item.add(item)

        # standard testing set building
        for entry in self.standard_test_data:
            user, item, rating = entry
            self.standard_test_set[user][item] = rating
            self.standard_test_set_item.add(item)

        for entry in self.deb_valid_data:
            user, item, rating = entry
            self.deb_valid_set[user][item] = rating
            self.deb_valid_set_item.add(item)

        for entry in self.deb_test_data:
            user, item, rating = entry
            self.deb_test_set[user][item] = rating
            self.deb_test_set_item.add(item)

        for entry in self.overall_valid_data:
            user, item, rating = entry
            self.overall_valid_set[user][item] = rating
            self.overall_valid_set_item.add(item)

        for entry in self.overall_test_data:
            user, item, rating = entry
            self.overall_test_set[user][item] = rating
            self.overall_test_set_item.add(item)

        for entry in self.cold_valid_data:
            user, item, rating = entry
            self.cold_valid_set[user][item] = rating
            self.cold_valid_set_item.add(item)

        for entry in self.cold_test_data:
            user, item, rating = entry
            self.cold_test_set[user][item] = rating
            self.cold_test_set_item.add(item)

        # raise Exception("now debugging...")

    def create_sparse_complete_bipartite_adjacency(self, self_connection=False):
        """
        return a sparse adjacency matrix with the shape (|u| + |i|, |u| + |i|)
        """
        n_nodes = self.user_num + self.item_num
        row_idx = [self.user[pair[0]] for pair in self.training_data]
        col_idx = [self.item[pair[1]] for pair in self.training_data]
        user_np = np.array(row_idx)
        item_np = np.array(col_idx)
        ratings = np.ones_like(user_np, dtype=np.float32)
        tmp_adj = sp.csr_matrix(
            (ratings, (user_np, item_np + self.user_num)),
            shape=(n_nodes, n_nodes),
            dtype=np.float32,
        )
        adj_mat = tmp_adj + tmp_adj.T
        if self_connection:
            adj_mat += sp.eye(n_nodes)
        return adj_mat

    def normalize_graph_mat(self, adj_mat):
        """
        :param adj_mat: the sparse adjacency matrix
        :return: normalized adjacency matrix
        """
        shape = adj_mat.get_shape()
        rowsum = np.array(adj_mat.sum(1))
        if shape[0] == shape[1]:
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.0
            d_mat_inv = sp.diags(d_inv)
            norm_adj_tmp = d_mat_inv.dot(adj_mat)
            norm_adj_mat = norm_adj_tmp.dot(d_mat_inv)
        else:
            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.0
            d_mat_inv = sp.diags(d_inv)
            norm_adj_mat = d_mat_inv.dot(adj_mat)
        return norm_adj_mat

    def convert_to_laplacian_mat(self, adj_mat):
        adj_shape = adj_mat.get_shape()
        n_nodes = adj_shape[0] + adj_shape[1]
        (user_np_keep, item_np_keep) = adj_mat.nonzero()
        ratings_keep = adj_mat.data
        tmp_adj = sp.csr_matrix(
            (ratings_keep, (user_np_keep, item_np_keep + adj_shape[0])),
            shape=(n_nodes, n_nodes),
            dtype=np.float32,
        )
        tmp_adj = tmp_adj + tmp_adj.T
        return self.normalize_graph_mat(tmp_adj)

    def create_sparse_interaction_matrix(self):
        """
        return a sparse adjacency matrix with the shape (user number, item number)
        """
        row, col, entries = [], [], []
        for pair in self.training_data:
            row += [self.user[pair[0]]]
            col += [self.item[pair[1]]]
            entries += [1.0]
        interaction_mat = sp.csr_matrix(
            (entries, (row, col)),
            shape=(self.user_num, self.item_num),
            dtype=np.float32,
        )
        return interaction_mat

    def get_user_id(self, u):
        if u in self.user:
            return self.user[u]
        else:
            raise Exception(f"user {u} not in current id table")

    def get_item_id(self, i):
        if i in self.item:
            return self.item[i]
        else:
            raise Exception(f"item {i} not in current id table")

    def training_size(self):
        return len(self.user), len(self.item), len(self.training_data)

    def standard_valid_size(self):
        return (
            len(self.standard_valid_set),
            len(self.standard_valid_set_item),
            len(self.standard_valid_data),
        )

    def standard_test_size(self):
        return (
            len(self.standard_test_set),
            len(self.standard_test_set_item),
            len(self.standard_test_data),
        )

    def deb_valid_size(self):
        return (
            len(self.deb_valid_set),
            len(self.deb_valid_set_item),
            len(self.deb_valid_data),
        )

    def deb_test_size(self):
        return (
            len(self.deb_test_set),
            len(self.deb_test_set_item),
            len(self.deb_test_data),
        )

    def overall_valid_size(self):
        return (
            len(self.overall_valid_set),
            len(self.overall_valid_set_item),
            len(self.overall_valid_data),
        )

    def overall_test_size(self):
        return (
            len(self.overall_test_set),
            len(self.overall_test_set_item),
            len(self.overall_test_data),
        )

    def contain(self, u, i):
        "whether user u rated item i"
        if u in self.user and i in self.training_set_u[u]:
            return True
        else:
            return False

    def contain_user(self, u):
        "whether user is in training set"
        if u in self.user:
            return True
        else:
            return False

    def contain_item(self, i):
        """whether item is in training set"""
        if i in self.item:
            return True
        else:
            return False

    def user_rated(self, u):
        return list(self.training_set_u[u].keys()), list(
            self.training_set_u[u].values()
        )

    def item_rated(self, i):
        return list(self.training_set_i[i].keys()), list(
            self.training_set_i[i].values()
        )
