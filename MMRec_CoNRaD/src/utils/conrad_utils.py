import torch
import numpy as np


def slice_sparse_tensor_rows(t, min_row, max_row):
    row_idxs = t.indices()[0]
    index_mask = (min_row <= row_idxs) & (row_idxs < max_row)

    num_rows = max_row - min_row
    num_cols = t.shape[1]

    idxs = t.indices()[:, index_mask]
    vals = t.values()[index_mask]
    idxs[0] -= min_row
    return torch.sparse_coo_tensor(idxs, vals, size=(num_rows, num_cols)).coalesce()


def filter_and_remap_topk(topk_indices, relevant_indices):
    """
    Filters and remaps a top-k index matrix to a reduced index space.

    Args:
        topk_indices (LongTensor): (B, K) top-k indices
        relevant_indices (LongTensor): (R,) allowed indices

    Returns:
        remapped (LongTensor): (B, K) where only relevant indices are remapped, others set to -1
        mask (BoolTensor): (B, K) mask indicating which entries were retained
    """
    device = topk_indices.device
    B, K = topk_indices.shape

    # Build remap lookup: index -> position
    max_index = torch.max(topk_indices.max(), relevant_indices.max()).item() + 1
    remap_lut = torch.full((max_index,), -1, dtype=torch.long, device=device)
    remap_lut[relevant_indices] = torch.arange(len(relevant_indices), device=device)

    # Remap and mask
    remapped = remap_lut[
        topk_indices
    ]  # (B, K), values will be -1 if not in relevant_indices
    mask = remapped != -1

    return remapped, mask


def masked_gather_padded_vectorized(src, topk_indices, relevant_indices, pad_value=0):
    """
    Fully vectorized version: gathers from `src` using `topk_indices` where `mask` is True,
    and returns a (B, max_valid_per_row) padded tensor.

    Args:
        src: (B, D) tensor to gather from
        topk_indices: (B, K) LongTensor of indices into dim=1 of `src`
        mask: (B, K) BoolTensor, True where index is valid
        pad_value: scalar to pad output with

    Returns:
        (B, max_valid_per_row) tensor
    """
    B, K = topk_indices.shape
    device = src.device
    topk_indices, mask = filter_and_remap_topk(topk_indices, relevant_indices)

    # Get number of valid entries per row
    valid_counts = mask.sum(dim=1)  # (B,)
    max_valid = valid_counts.max().item()

    # Compute gather indices and values
    flat_src_indices = topk_indices[mask]  # (N,)
    flat_batch_indices = (
        torch.arange(B, device=device).unsqueeze(1).expand(B, K)[mask]
    )  # (N,)
    flat_values = src[flat_batch_indices, flat_src_indices]  # (N,)

    # Compute per-row insertion positions (0, 1, 2, ...) for each row
    insert_pos = torch.cumsum(mask.to(torch.long), dim=1) - 1  # (B, K)
    insert_pos = insert_pos[mask]  # (N,)

    # Build output tensor
    output = torch.full((B, max_valid), pad_value, device=device, dtype=src.dtype)
    output[flat_batch_indices, insert_pos] = flat_values
    return output


def l2_reg_loss(reg, *args):
    emb_loss = 0
    for emb in args:
        emb_loss += torch.norm(emb, p=2) / emb.shape[0]
    return emb_loss * reg


def bpr_loss(user_emb, pos_item_emb, neg_item_emb):
    pos_score = torch.mul(user_emb, pos_item_emb).sum(dim=1)
    neg_score = torch.mul(user_emb, neg_item_emb).sum(dim=1)
    loss = -torch.log(10e-6 + torch.sigmoid(pos_score - neg_score))
    return torch.mean(loss)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def sparse_mx_to_torch_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    zeros = torch.zeros(sparse_mx.shape, dtype=torch.float32)
    values = torch.from_numpy(sparse_mx.data)
    zeros[sparse_mx.row, sparse_mx.col] = values
    return zeros
