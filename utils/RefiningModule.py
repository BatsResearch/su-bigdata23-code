import torch
import numpy as np
import os
import torch


def get_graph(args, LFs_removed):
    total_lf = args['total_lf']-len(LFs_removed)
    dataset_path = args['dataset_path']
    data = args['data']
    embedding_path = os.path.join(dataset_path, data, 'embedding.pt')
    emb = torch.load(embedding_path).detach()
    simi_matrix = emb @ emb.t()
    simi_matrix  = np.array(simi_matrix) 
    
    
    simi_matrix = np.delete(simi_matrix, LFs_removed, axis=1)
    simi_matrix = np.delete(simi_matrix, LFs_removed, axis=0)
    
    dependency_graph = []
    dependency_rate = args['threshold_structure']
    
    simi_matrix_lower = np.tril(simi_matrix, -1)
    simi_matrix_lower[np.where(simi_matrix_lower==0)] = 1
    # make at least four indepedendt LFs.
    idx = np.argmin(simi_matrix_lower)
    lf1 = idx // total_lf 
    lf2 = idx % total_lf 
    
    simi_matrix_lower[np.where(simi_matrix_lower== 1)] = 0
    simi_matrix_lower[lf1, :] = 0
    simi_matrix_lower[:, lf1] = 0
    simi_matrix_lower[lf2, :] = 0
    simi_matrix_lower[:, lf2] = 0
    if dependency_rate >=1:
        dependency_num = int(dependency_rate)
    else: 
        dependency_num = int(np.sum(simi_matrix_lower>0) * dependency_rate)

    
    for i in range(dependency_num):
        idx = np.argmax(simi_matrix_lower)
        lf1 = idx // total_lf 
        lf2 = idx % total_lf 
        
        dependency_graph.append((lf1,lf2))
        simi_matrix_lower[lf1, lf2] = 0
        
    return dependency_graph


def get_removing_list(args):
    total_lf = args['total_lf']
    dataset_path = args['dataset_path']
    data = args['data']
    embedding_path = os.path.join(dataset_path, data, 'embedding.pt')
    emb = torch.load(embedding_path).detach()
    simi_matrix = emb @ emb.t()
    simi_matrix  = np.array(simi_matrix) 
    simi_matrix[(np.array(range(len(simi_matrix))), np.array(range(len(simi_matrix))))] = simi_matrix.min()
    Max_similarity = simi_matrix.max(0)


    
    if args['threshold_removing']>=1:
        removing_num = int(args['threshold_removing'])
    else:
        removing_num = int(total_lf * args['threshold_removing'])
    
    if removing_num >= len(Max_similarity) -2:
        raise Exception(f"Quantile has to be < =  {(len(Max_similarity) -3)/total_lf}")

    LFs_removed = []
    for i in range(removing_num):
        idx = np.argmax(simi_matrix)
        lf1 = idx // total_lf 
        lf2 = idx % total_lf 
        lf_num_to_remove = np.max([lf1, lf2])
        LFs_removed.append(lf_num_to_remove)
        simi_matrix[lf_num_to_remove,:] = 0
        simi_matrix[:,lf_num_to_remove] = 0
    return LFs_removed