import warnings
warnings.filterwarnings("ignore")

import os
from tqdm import tqdm
import numpy as np
import torch
import faiss
from prettytable import PrettyTable

def calculate_metric(targets):
    mAP, mHR, mRR = [], [], []

    for _, target in enumerate(targets):
        if target.sum() == 0:
            mHR.append(0)
            continue

        pos = 0
        found_hit = False
        for i, t in enumerate(target):
            if t:
                pos += 1
                mAP.append(pos/(i+1))
                if not found_hit: mRR.append(1/(i+1))
                found_hit = True
        mHR.append(int(found_hit))

    if len(mAP) == 0:
        return 0., 0., 0.

    return np.mean(mAP), np.mean(mHR), np.mean(mRR)


def compute_metrics(config, model_criteria, all_emb, all_labels, all_label_names, top_k=10, dist_metric='cosine', is_save=True):

    # build the index using faiss
    d = all_emb.shape[1]
    if dist_metric == 'cosine':
        faiss_retriever = faiss.IndexFlatIP(d)
        all_emb = all_emb/np.linalg.norm(all_emb, axis=1, keepdims=True)
    else:
        faiss_retriever = faiss.IndexFlatL2(d)
    faiss_retriever.add(all_emb)
    print(f'\nFaiss trained {faiss_retriever.is_trained} on {faiss_retriever.ntotal} vectors of size {faiss_retriever.d}')

    # targets contains the retrieved labels checked against ground truth labels
    # predictions contain the distances of the retrieved labels
    all_targets_for_metric = {lbl_name: [] for lbl_name in all_label_names}
    all_predicted_for_metric = {lbl_name: [] for lbl_name in all_label_names}

    # perform retrieval and save the input required for the metrics
    for _, (emb, query_labels) in enumerate(tqdm(zip(all_emb, all_labels), total=len(all_emb), desc=f'Retreiving top-{top_k} {dist_metric}...')):

        # expand dimension
        emb = emb[np.newaxis, ...]
        query_labels = query_labels[np.newaxis, ...]

        # perform retrieval
        D, I = faiss_retriever.search(emb, top_k+1)

        # find the corresponding labels from the retrieved indices
        # ignore the first one as it the query itself
        labels = all_labels[I[:, 1:]]
        distances = torch.softmax(torch.tensor(D[:, 1:]), dim=1)

        # we only care about query labels that are present
        target = torch.tensor(labels == 1)
        predicted = 1/(distances+1)

        # class wise metrics
        for i, label_name in enumerate(all_label_names):

            # works with batched retrieval as well
            consider_batches = query_labels[:, i] == 1
            if consider_batches.sum() == 0:
                continue
            # extract only the relevant batches
            temp_target = target[consider_batches]
            temp_predicted = predicted[consider_batches]

            # save necessary values
            all_targets_for_metric[label_name].append(temp_target[:, :, i])
            all_predicted_for_metric[label_name].append(temp_predicted)

    # convert to tensors
    all_targets_for_metric = {k: torch.cat(v) for k, v in all_targets_for_metric.items()}
    all_predicted_for_metric = {k: torch.cat(v) for k, v in all_predicted_for_metric.items()}
    all_indexes_for_metric = {k: torch.tensor([[i for _ in range(v.shape[1])] for i in range(v.shape[0])]) for k, v in all_targets_for_metric.items()}

    # for pretty tables
    t = PrettyTable(['Label Name', 'mAP', 'mHR', 'mRR'])
    print()

    # compute class wise metrics
    avg_values = []
    for i, label_name in enumerate(tqdm(all_label_names, desc='Computing metrics...')):

        new_map, new_mhr, new_mrr = calculate_metric(all_targets_for_metric[label_name])
        avg_values.append([new_map, new_mhr, new_mrr])

        # add the row to the table
        t.add_row([label_name,  np.round(new_map, 3), np.round(new_mhr, 3), np.round(new_mrr, 3)])

    avg_map, avg_mhr, avg_mrr = np.mean(avg_values, axis=0)
    t.add_row(['Class Average', np.round(avg_map, 3), np.round(avg_mhr, 3), np.round(avg_mrr, 3)])

    # add the average row to the table and write to file
    weights = np.load('data/mimic_cxr_jpg/test_weights.npy')
    avg_map, avg_mhr, avg_mrr = np.average(avg_values, axis=0, weights=weights)
    t.add_row(['Class Weighted Average', np.round(avg_map, 3), np.round(avg_mhr, 3), np.round(avg_mrr, 3)])

    print(t)
    if not is_save:
        return t, avg_map, avg_mhr, avg_mrr

    # create directory for the run
    dir_name = f'output/{config["task"]}/{config["run"]}'
    os.makedirs(dir_name, exist_ok=True)

    # save the table to file
    file_name = f'{dir_name}/{len(all_label_names)}_classes_{model_criteria}{"_concat_global_feat" if config["concat_global_feature"] else ""}_top_{top_k}_{dist_metric}.txt'
    with open(file_name, 'w') as f:
        f.write(str(t))


def compute_occluded_metrics(config, model_criteria, gt_emb, gt_labels, anatomy_embs, all_label_names, anatomy_names, top_k=10, dist_metric='cosine'):

    # build the index using faiss
    d = gt_emb.shape[1]
    if dist_metric == 'cosine':
        faiss_retriever = faiss.IndexFlatIP(d)
        gt_emb = gt_emb/np.linalg.norm(gt_emb, axis=1, keepdims=True)
        anatomy_embs = [anatomy_emb/np.linalg.norm(anatomy_emb, axis=1, keepdims=True) for anatomy_emb in anatomy_embs]
    else:
        faiss_retriever = faiss.IndexFlatL2(d)
    faiss_retriever.add(gt_emb)
    print(f'\nFaiss trained {faiss_retriever.is_trained} on {faiss_retriever.ntotal} vectors of size {faiss_retriever.d}')

    # create directory for the run
    dir_name = f'output/{config["task"]}/{config["run"]}'
    os.makedirs(dir_name, exist_ok=True)

    # save the table to file
    file_name = f'{dir_name}/Aanlysis_{len(all_label_names)}_classes_{model_criteria}{"_concat_global_feat" if config["concat_global_feature"] else ""}_top_{top_k}_{dist_metric}.txt'

    with open(file_name, 'w') as f:
        t, _, _, _ = compute_metrics(
            config, model_criteria,
            gt_emb, gt_labels, all_label_names,
            top_k=top_k, dist_metric=dist_metric, is_save=False)
        f.write(f'Results without occlusion:\n{str(t)}\n\n')
        print(t)

        for anatomy_occluded_emb, anatomy_name in zip(anatomy_embs, anatomy_names):
            t = compute_metrics(
                config, model_criteria,
                anatomy_occluded_emb, gt_labels, all_label_names,
                top_k=top_k, dist_metric=dist_metric, is_save=False)
            f.write(f'Results occluding {anatomy_name} node:\n{str(t)}\n\n')
            print(t)