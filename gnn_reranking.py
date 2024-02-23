import torch
import build_adjacency_matrix
import gnn_propagate

def gnn_reranking(X_q, k1, k2):
    ##query_num, gallery_num = X_q.shape[0], X_g.shape[0]

    #X_u = torch.cat((X_q, X_g), axis = 0)
    original_score = torch.mm(X_q, X_q.t())
    #print(original_score)

    del X_q

    # initial ranking list
    S, initial_rank = original_score.topk(k=k1, dim=-1, largest=True, sorted=True)
    #print(S)
    #print(initial_rank)
    # stage 1
    A = build_adjacency_matrix.forward(initial_rank.float())   
    S = S * S

    # stage 2
    if k2 != 1:      
        for i in range(2):
            A = A + A.T
            A = gnn_propagate.forward(A, initial_rank[:, :k2].contiguous().float(), S[:, :k2].contiguous().float())
            A_norm = torch.norm(A, p=2, dim=1, keepdim=True)
            A = A.div(A_norm.expand_as(A))                     
    
    cosine_similarity = torch.mm(A, A.t())
    del A, S
    
    X, L = torch.sort(cosine_similarity, dim = 1, descending=True)
    L = L.data.cpu()
    X = X.data.cpu()
    return  X, L


def gnn_reranking_v2(X_q, X_g, k1, k2):
    query_num, gallery_num = X_q.shape[0], X_g.shape[0]

    X_u = torch.cat((X_q, X_g), axis = 0)
    original_score = torch.mm(X_q, X_q.t())
    
    del X_q

    # initial ranking list
    S, initial_rank = original_score.topk(k=k1, dim=-1, largest=True, sorted=True)
    
    # stage 1
    A = build_adjacency_matrix.forward(initial_rank.float())   
    S = S * S

    # stage 2
    if k2 != 1:      
        for i in range(2):
            A = A + A.T
            A = gnn_propagate.forward(A, initial_rank[:, :k2].contiguous().float(), S[:, :k2].contiguous().float())
            A_norm = torch.norm(A, p=2, dim=1, keepdim=True)
            A = A.div(A_norm.expand_as(A))                     
    
    cosine_similarity = torch.mm(A[:query_num,], A[query_num:, ].t())
    del A, S
    
    return cosine_similarity