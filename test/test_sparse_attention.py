# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import time
import torch
import random
from SparTA.OPs import *


def compute_distance_to_token(ATTN_H, ATTN_W, FRAME_W):
    distance = []
    x_distance = []
    y_distance = []
    token_h = ATTN_H // 2
    token_w = ATTN_W // 2
    for h in range(ATTN_H):
        for w in range(ATTN_W):
            d = (h - token_h) * FRAME_W + (w - token_w)
            distance.append(d)
            x_distance.append(w - token_w)
            y_distance.append(h - token_h)
    # remove some due to causal
    #attn_size_per_frame = ATTN_H * ATTN_W
    #causal_remove_num = attn_size_per_frame // 2 + 1
    #distance = distance[:(-1 * causal_remove_num)]
    return distance, x_distance, y_distance


def nuwa_sparse_pattern(ATTN_T, ATTN_H, ATTN_W, FRAME_T, FRAME_H, FRAME_W):
    """
    Returns
    -------
    List[List]
        a sparse matrix, where 0 means pruned, 1 means kept
    """
    M = K = FRAME_T * FRAME_H * FRAME_W
    # init matrix
    matrix = []
    for _ in range(M):
        matrix.append([])
        for _ in range(K):
            matrix[-1].append(0)
    # prepare distance
    distance, x_distance, y_distance = compute_distance_to_token(
        ATTN_H, ATTN_W, FRAME_W)
    # print(distance)
    # make the places of attention to be 1
    for i, token_attn in enumerate(matrix):
        frame_seq = i // (FRAME_H*FRAME_W)
        intra_frame_loc = i % (FRAME_H*FRAME_W)
        intra_frame_h = intra_frame_loc // FRAME_W
        intra_frame_w = intra_frame_loc % FRAME_W
        # deal with the previous ATTN - 1 frames
        for curr_frame_seq in range(max(0, frame_seq - ATTN_T), frame_seq):
            for dis, dis_x, dis_y in zip(distance, x_distance, y_distance):
                if 0 <= intra_frame_loc + dis < FRAME_H * FRAME_W and \
                        0 <= intra_frame_h + dis_y < FRAME_H and \
                        0 <= intra_frame_w + dis_x < FRAME_W:
                    token_attn[curr_frame_seq *
                               (FRAME_H*FRAME_W) + intra_frame_loc + dis] = 1
        # deal with the current frame
        for dis, dis_x, dis_y in zip(distance[:-1*((ATTN_H*ATTN_W)//2)], x_distance[:-1*((ATTN_H*ATTN_W)//2)], y_distance[:-1*((ATTN_H*ATTN_W)//2)]):
            if 0 <= intra_frame_loc + dis < FRAME_H * FRAME_W and \
                    0 <= intra_frame_h + dis_y < FRAME_H and \
                    0 <= intra_frame_w + dis_x < FRAME_W:
                token_attn[frame_seq * (FRAME_H*FRAME_W) +
                           intra_frame_loc + dis] = 1
    return matrix


def random_sparse_pattern(seq_len, sparsity):
    pattern = torch.zeros(seq_len, seq_len, dtype=torch.int32)
    nnz = int(seq_len * seq_len * sparsity)
    print("NNZ: ", nnz)
    for _ in range(nnz):
        i, j = random.randint(0, seq_len-1), random.randint(0, seq_len-1)
        pattern[i][j] = 1
    return pattern


def test_speed(sparse_attention, head_num, seq_len, hidden_n, device):
    # warmup
    q = torch.rand(batch_size, head_num, seq_len, hidden_n,
                   dtype=torch.float32, device=device)
    k = torch.rand(batch_size, head_num, seq_len, hidden_n,
                   dtype=torch.float32, device=device)
    v = torch.rand(batch_size, head_num, seq_len, hidden_n,
                   dtype=torch.float32, device=device)
    out = sparse_attention(q, k, v)
    out_grad = torch.rand_like(out)

    torch.cuda.synchronize()
    st = time.time()
    for _ in range(50):
        q = torch.rand(batch_size, head_num, seq_len, hidden_n,
                       dtype=torch.float32, device=device)
        k = torch.rand(batch_size, head_num, seq_len, hidden_n,
                       dtype=torch.float32, device=device)
        v = torch.rand(batch_size, head_num, seq_len, hidden_n,
                       dtype=torch.float32, device=device)
        sparse_attention(q, k, v)
    torch.cuda.synchronize()
    end = time.time()
    print('Sparse Forward Implementation', end-st)

    torch.cuda.synchronize()
    st = time.time()
    for _ in range(50):
        q = torch.rand(batch_size, head_num, seq_len, hidden_n,
                       dtype=torch.float32, device=device, requires_grad=True)
        k = torch.rand(batch_size, head_num, seq_len, hidden_n,
                       dtype=torch.float32, device=device, requires_grad=True)
        v = torch.rand(batch_size, head_num, seq_len, hidden_n,
                       dtype=torch.float32, device=device, requires_grad=True)
        out = sparse_attention(q, k, v)
        out.backward(out_grad)
    torch.cuda.synchronize()
    end = time.time()
    print('Sparse Forward+Backward Implementation', end-st)


def dense_speed(sparse_attention, head_num, seq_len, hidden_n, device):
    # warmup
    q = torch.rand(batch_size, head_num, seq_len, hidden_n,
                   dtype=torch.float32, device=device)
    k = torch.rand(batch_size, head_num, seq_len, hidden_n,
                   dtype=torch.float32, device=device)
    v = torch.rand(batch_size, head_num, seq_len, hidden_n,
                   dtype=torch.float32, device=device)
    out = sparse_attention.reference_forward(q, k, v)
    out_grad = torch.rand_like(out)

    torch.cuda.synchronize()
    st = time.time()
    for _ in range(50):
        q = torch.rand(batch_size, head_num, seq_len, hidden_n,
                       dtype=torch.float32, device=device)
        k = torch.rand(batch_size, head_num, seq_len, hidden_n,
                       dtype=torch.float32, device=device)
        v = torch.rand(batch_size, head_num, seq_len, hidden_n,
                       dtype=torch.float32, device=device)
        out = sparse_attention.reference_forward(q, k, v)
    torch.cuda.synchronize()
    end = time.time()
    print('Dense Forward Implementation', end-st)

    torch.cuda.synchronize()
    st = time.time()
    for _ in range(50):
        q = torch.rand(batch_size, head_num, seq_len, hidden_n,
                       dtype=torch.float32, device=device, requires_grad=True)
        k = torch.rand(batch_size, head_num, seq_len, hidden_n,
                       dtype=torch.float32, device=device, requires_grad=True)
        v = torch.rand(batch_size, head_num, seq_len, hidden_n,
                       dtype=torch.float32, device=device, requires_grad=True)
        out = sparse_attention.reference_forward(q, k, v)
        out.backward(out_grad)
    torch.cuda.synchronize()
    end = time.time()
    print('Dense Forward+Backward Implementation', end-st)


def test_correctness(sparse_attention, HEAD_NUM, seq_len, hidden_n, device):
    q, k, v = torch.randn(batch_size, HEAD_NUM, seq_len, hidden_n, dtype=torch.float32, device=device), torch.randn(batch_size, HEAD_NUM, seq_len,
                                                                                                                    hidden_n, dtype=torch.float32, device=device), torch.randn(batch_size, HEAD_NUM, seq_len, hidden_n, dtype=torch.float32, device=device)

    # test the correctness of the backward function
    q1 = q.clone().detach()
    q2 = q.clone().detach()
    k1 = k.clone().detach()
    k2 = k.clone().detach()
    v1 = v.clone().detach()
    v2 = v.clone().detach()
    q1.requires_grad_()
    q2.requires_grad_()
    k1.requires_grad_()
    k2.requires_grad_()
    v1.requires_grad_()
    v2.requires_grad_()
    out_2 = sparse_attention.reference_forward(q2, k2, v2)
    in_grad = torch.rand_like(out_2)
    out = sparse_attention(q1, k1, v1)
    # tmp_attn.retain_grad()
    out_2.backward(in_grad)
    out.backward(in_grad)
    if not (torch.allclose(out, out_2, rtol=1e-08, atol=1e-04) and torch.allclose(q1.grad.data, q2.grad.data, rtol=1e-08, atol=1e-04) and torch.allclose(k1.grad.data, k2.grad.data, rtol=1e-08, atol=1e-04) and torch.allclose(v1.grad.data, v2.grad.data, rtol=1e-08, atol=1e-04)):
        import pdb
        pdb.set_trace()
    assert torch.allclose(out, out_2, rtol=1e-08, atol=1e-04)
    assert torch.allclose(q1.grad.data, q2.grad.data, rtol=1e-08, atol=1e-04)
    assert torch.allclose(k1.grad.data, k2.grad.data, rtol=1e-08, atol=1e-04)
    assert torch.allclose(v1.grad.data, v2.grad.data, rtol=1e-08, atol=1e-04)
    print('Correctness test passed')


def test_nuwa():
    HEAD_NUM = 20
    attn_t = 1  # 4
    attn_h = 5
    attn_w = 5
    frame_t = 1  # 10
    frame_h = 32  # 16
    frame_w = 32  # 16
    device = torch.device('cuda:2')
    sp_matrix = nuwa_sparse_pattern(
        attn_t, attn_h, attn_w, frame_t, frame_h, frame_w)
    out_mask = torch.tensor(sp_matrix)
    M, N = out_mask.size()
    K = 64
    spa = SparseAttention(out_mask, HEAD_NUM, M, K)
    # test the speed
    test_speed(spa, HEAD_NUM, M, K, device)
    dense_speed(spa, HEAD_NUM, M, K, device)
    test_correctness(spa, HEAD_NUM, M, K, device)


def test_random(HEAD_NUM, seq_len, hidden_dim, sparsity):
    print(HEAD_NUM, seq_len, hidden_dim, sparsity)
    sp_pattern = random_sparse_pattern(seq_len, sparsity)
    M, N = sp_pattern.size()
    K = hidden_dim
    device = torch.device('cuda:2')

    spa = SparseAttention(sp_pattern, HEAD_NUM, M, K)
    # test the speed
    test_speed(spa, HEAD_NUM, M, K, device)
    test_correctness(spa, HEAD_NUM, M, K, device)


if __name__ == '__main__':
    batch_size = 1
    test_nuwa()
    # test_random(20, 1024, 64, 0.00001)
