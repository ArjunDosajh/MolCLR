import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from map_mhr_mrr import calculate_metric
import faiss
from prettytable import PrettyTable
from collections import defaultdict
from models.gcn_molclr import GCN
from dataset.dataset_contrastive import USPTO50_contrastive
from torch_geometric.loader import DataLoader
# import metrics_map, metrics_mhr, metrics_mrr from torchmetrics
from torchmetrics import RetrievalMAP, RetrievalMRR, RetrievalHitRate

def get_GCN_model(path, device):
    gcn_model = GCN(feat_dim=512)
    if path != "random_weights":
        try:
            gcn_model.load_state_dict(torch.load(path))
        except:
            gcn_model.load_state_dict(torch.load(path, map_location='cuda:0'))
    else:
        print("Random weights are used")
    gcn_model.eval()
    gcn_model.to(device)
    
    return gcn_model

def update_exclude_indices(df, old_to_new_index):
    df['exclude_indices'] = df['exclude_indices'].apply(lambda x: [old_to_new_index.get(i, i) for i in x])
    return df

def get_test_dataset(dataset):
    old_to_new_index = {old_index: new_index for new_index, old_index in enumerate(dataset[dataset['set'] == 'test'].index)}

    dataset_filtered = dataset[dataset['set'] == 'test'].reset_index(drop=True)
    dataset_filtered = update_exclude_indices(dataset_filtered, old_to_new_index)
    return dataset_filtered

def get_embeddings(model, dataclass, device):
    uspto_graph_retrieval_dataloader = DataLoader(dataclass, batch_size=32, shuffle=False, num_workers=36, pin_memory=True)
    reactants_embedding = []
    products_embedding = []

    with torch.no_grad():
        print("Getting embeddings")
        for bn, (anchor, positive, negative, index) in enumerate(tqdm(uspto_graph_retrieval_dataloader)):
            anchor = anchor.pin_memory().to(device, non_blocking=True)
            positive = positive.pin_memory().to(device, non_blocking=True)

            anchor_embedding = model(anchor)
            positive_embedding = model(positive)

            reactants_embedding.extend(positive_embedding.cpu().detach().numpy())
            products_embedding.extend(anchor_embedding.cpu().detach().numpy())

    return np.array(reactants_embedding), np.array(products_embedding)

def get_metrics(model_path, dataset_path):
    uspto_triplets_dataset = pd.read_pickle(dataset_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    uspto_filtered_dataset = get_test_dataset(uspto_triplets_dataset)
    USPTO_triplets_dataclass = USPTO50_contrastive(uspto_filtered_dataset, return_index=True, split='all')

    gcn_model = get_GCN_model(model_path, device)

    reactants_embedding, products_embedding = get_embeddings(gcn_model, USPTO_triplets_dataclass, device)
    reactants_embedding = reactants_embedding / np.linalg.norm(reactants_embedding, axis=1, keepdims=True)
    products_embedding = products_embedding / np.linalg.norm(products_embedding, axis=1, keepdims=True)
    reactants_embedding, products_embedding = torch.tensor(reactants_embedding), torch.tensor(products_embedding)

    exclude_indices = uspto_filtered_dataset['exclude_indices'].tolist()

    d = reactants_embedding.shape[1]

    index = faiss.IndexFlatIP(d)
    index = faiss.index_cpu_to_all_gpus(index)
    index.add(reactants_embedding)

    k = 5
    k_retrieved_indices = []
    print("Using faiss to retrieve indices")
    for i in tqdm(range(products_embedding.shape[0])):
        D, I = index.search(products_embedding[i][np.newaxis, ...], k)
        k_retrieved_indices.append(I[0])

    skip_indices = []
    # targets = [] # for map_mhr_mrr.py

    metrics_queries = torch.tensor([], dtype=torch.long)
    metrics_targets = torch.tensor([], dtype=torch.bool)
    metrics_predictions = torch.tensor([], dtype=torch.float)

    # filling the predictions array with any numbers in decreasing order, since faiss by default returns the indices in decreasing order of similarity
    for i in range(5004):
        metrics_predictions = torch.cat((metrics_predictions, torch.tensor([k-i for i in range(k)], dtype=torch.long)))
    query_count = 0

    print("Calculating metrics")
    for idx, row in tqdm(uspto_filtered_dataset.iterrows(), total=len(uspto_filtered_dataset)):
        if idx in skip_indices:
            continue

        true_reactants_indices = exclude_indices[idx]
        retrieved_reactants_indices = k_retrieved_indices[idx]

        skip_indices.extend(exclude_indices[idx])

        targets_idx = [False for _ in range(len(retrieved_reactants_indices))]

        for idx, retrieved_idx in enumerate(retrieved_reactants_indices):
            if retrieved_idx in true_reactants_indices:
                targets_idx[idx] = True
        
        metrics_targets = torch.cat((metrics_targets, torch.tensor(targets_idx, dtype=torch.bool)))
        # targets.append(targets_idx) # for map_mhr_mrr.py
        metrics_queries = torch.cat((metrics_queries, torch.tensor([query_count for i in range(k)], dtype=torch.long)))

        query_count += 1
    
    # targets = np.array(targets)
    # map, mhr, mrr = calculate_metric(targets)

    map_metric = RetrievalMAP()
    mrr_metric = RetrievalMRR()
    mhr_metric = RetrievalHitRate()

    # Compute the metrics
    map_value = map_metric(metrics_predictions, metrics_targets, indexes=metrics_queries)
    mrr_value = mrr_metric(metrics_predictions, metrics_targets, indexes=metrics_queries)
    mhr_value = mhr_metric(metrics_predictions, metrics_targets, indexes=metrics_queries)

    return map_value, mhr_value, mrr_value
    # return map, mhr, mrr
