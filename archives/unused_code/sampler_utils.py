from dotenv import load_dotenv
import time
import openai
import numpy as np
from scipy.stats import rankdata
import subprocess
import logging
import os
from pydantic import BaseModel

# Disable httpx logging from OpenAI client

import torch
# from src.modules.freebase_interface import freebase_interface

import random
from functools import wraps


# def convert_mid_to_name(mid: str) -> str:
#     mid = freebase_interface._remove_ns(mid)

#     if mid[1] == '/':
#         mid = mid.replace('/', '.')
#     if mid[:2] in ['m.', '@.', 'g.']:
#         return freebase_interface.convert_id_to_name(mid)
#     else:
#         return freebase_interface.convert_name_to_id(mid)




def extract_numbers(t):
    """Recursively extract all numbers from nested tuples into a flat list."""
    numbers = []
    if isinstance(t, int):
        numbers.append(t)
    elif isinstance(t, (tuple, list)):
        for item in t:
            numbers.extend(extract_numbers(item))
    return numbers


def extract_strings(t):
    """Recursively extract all strings from nested tuples into a flat list."""
    strings = []
    if isinstance(t, str):
        strings.append(t)
    elif isinstance(t, (tuple, list)):
        for item in t:
            strings.extend(extract_strings(item))
    return strings


def extract_notations(t):
    """Recursively extract all notations (entities and relations)
    from nested tuples into a flat list."""
    notations = []
    t = str(t)
    for c in t:
        if c.isalpha():
            notations.append(c)
    return notations


def extract_answer(text: str) -> str:
    start_index = text.find("{")
    end_index = text.find("}")
    if start_index != -1 and end_index != -1:
        return text[start_index + 1: end_index].strip()
    else:
        return ""


def flatten_nested_tuple(input: tuple) -> list:
    """Recursively flatten a nested tuple into a flat list."""
    flat_list = []
    if isinstance(input, (tuple, list)):
        for item in input:
            flat_list.extend(flatten_nested_tuple(item))
    else:
        flat_list.append(input)
    return flat_list


def normalize_mid(mid: str):
    if mid.startswith("/"):
        mid = mid[1:]
    return mid.replace("/", ".")


class SubobjectiveOutput(BaseModel):
    res: list[str]


def get_last_token_embedding(
    text: str,
    model: str = "meta-llama/Llama-3.1-8B-Instruct",
) -> torch.Tensor:
    """Return the last-token embedding of *text* from the remote Llama model.

    The /v1/embeddings endpoint of a decoder model returns the hidden state at
    the final (last) token position, which acts as a summary representation of
    the full input sequence.

    Returns a 1-D float32 numpy array of shape (hidden_dim,).
    """
    # dummy code:
    return torch.randn(4096)

    # while True:
    #     try:
    #         response = llama_client.embeddings.create(model=model, input=text)
    #         return torch.tensor(response.data[0].embedding, dtype=torch.float32)
    #     except Exception as e:
    #         print(f"embedding error, retry: {e}")
    #         time.sleep(2)


def get_subgraph_relation_embeddings(
    subgraph_edges: torch.Tensor,
    relation_description_dict: dict[int, str],
    model: str = "meta-llama/Llama-3.1-8B-Instruct",
) -> dict[int, torch.Tensor]:
    """Extract all unique relations from a subgraph and return their embeddings.

    Args:
        subgraph_edges: LongTensor of shape (E, 3) where column 1 is relation ID.
        model:          Llama model identifier served at the remote endpoint.

    Returns:
        dict mapping relation_id (int) -> np.ndarray embedding (float32, 1-D).
    """
    # from data_utils import id2rel
    # DONE, hope so

    unique_rel_ids = torch.unique(subgraph_edges[:, 1]).tolist()

    embeddings = {}
    for rel_id in unique_rel_ids:
        # rel_name = id2rel.get(rel_id)
        relation_description = relation_description_dict.get(rel_id, None)
        if relation_description is None:
            continue
        embeddings[rel_id] = get_last_token_embedding(relation_description, model=model)
    return embeddings


def cal_ranks(scores, labels, filters):
    scores = scores - np.min(scores, axis=1, keepdims=True) + 1e-8
    full_rank = rankdata(-scores, method='average', axis=1)
    filter_scores = scores * filters
    filter_rank = rankdata(-filter_scores, method='min', axis=1)
    ranks = (full_rank - filter_rank + 1) * labels      # get the ranks of multiple answering entities simultaneously
    ranks = ranks[np.nonzero(ranks)]
    return list(ranks)


def cal_ranks_mean(scores, labels, filters):
    scores = scores - np.min(scores, axis=1, keepdims=True) + 1e-8
    full_rank = rankdata(-scores, method='average', axis=1)
    filter_scores = scores * filters
    filter_rank = rankdata(-filter_scores, method='min', axis=1)
    ranks = (full_rank - filter_rank + 1) * labels      # get the ranks of multiple answering entities simultaneously
    batch_idx, ent_idx = np.nonzero(ranks)
    mean_rank = [[] for i in range(int(ranks.shape[0]))]
    for i in range(len(batch_idx)):
        x, y = batch_idx[i], ent_idx[i]
        rank = ranks[x, y]
        mean_rank[x].append(rank)
    return mean_rank


def checkPath(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return


def cal_performance(ranks):
    mrr = (1. / ranks).sum() / len(ranks)
    h_1 = sum(ranks <= 1) * 1.0 / len(ranks)
    h_10 = sum(ranks <= 10) * 1.0 / len(ranks)
    return mrr, h_1, h_10


def select_gpu():
    nvidia_info = subprocess.run('nvidia-smi', stdout=subprocess.PIPE)
    gpu_info = False
    gpu_info_line = 0
    proc_info = False
    gpu_mem = []
    gpu_occupied = set()
    i = 0
    for line in nvidia_info.stdout.split(b'\n'):
        line = line.decode().strip()
        if gpu_info:
            gpu_info_line += 1
            if line == '':
                gpu_info = False
                continue
            if gpu_info_line % 3 == 2:
                mem_info = line.split('|')[2]
                used_mem_mb = int(mem_info.strip().split()[0][:-3])
                gpu_mem.append(used_mem_mb)
        if proc_info:
            if line == '|  No running processes found                                                 |':
                continue
            if line == '+-----------------------------------------------------------------------------+':
                proc_info = False
                continue
            proc_gpu = int(line.split()[1])
            # proc_type = line.split()[3]
            gpu_occupied.add(proc_gpu)
        if line == '|===============================+======================+======================|':
            gpu_info = True
        if line == '|=============================================================================|':
            proc_info = True
        i += 1
    for i in range(0, len(gpu_mem)):
        if i not in gpu_occupied:
            logging.info('Automatically selected GPU %d because it is vacant.', i)
            return i
    for i in range(0, len(gpu_mem)):
        if gpu_mem[i] == min(gpu_mem):
            logging.info('All GPUs are occupied. Automatically selected GPU %d because it has the most free memory.', i)
            return i


def calculate_statistics(data):
    """
    Calculates descriptive statistics for a given array-like object.

    Args:
        data (array-like): A list or NumPy array of numerical data.

    Returns:
        dict: A dictionary containing the statistics,
                or a dictionary of Nones if the input is empty.
    """
    # Ensure data is a numpy array
    data_arr = np.asarray(data)

    # Handle empty array
    if data_arr.size == 0:
        return {
            'count': 0,
            'mean': None,
            'median': None,
            'std_dev': None,
            'min': None,
            'max': None,
            '25th_percentile': None,
            '75th_percentile': None
        }

    stats = {
        'count': data_arr.size,
        'mean': np.mean(data_arr),
        'median': np.median(data_arr),
        'std_dev': np.std(data_arr),
        'min': np.min(data_arr),
        'max': np.max(data_arr),
        '25th_percentile': np.percentile(data_arr, 25),
        '75th_percentile': np.percentile(data_arr, 75)
    }
    return stats


def fuse_mean(scores: torch.Tensor) -> torch.Tensor:
    ls = scores.unbind(dim=0)
    res = torch.zeros(1, scores.shape[1])
    n = scores.shape[0]
    for l in ls:
        res += l / n
    return res


def fuse_rrf(scores: torch.Tensor, k=5) -> torch.Tensor:
    ls = scores.unbind(dim=0)
    dict_scores = {}
    for l in ls:
        _, indices = torch.sort(l, descending=True)
        for i in range(scores.shape[1]):
            if indices[i].item() not in dict_scores:
                dict_scores[indices[i].item()] = []
            dict_scores[indices[i].item()].append(i + 1)

    for key in dict_scores:
        dict_scores[key] = sum([1 / (k + rank) for rank in dict_scores[key]])
    scores = torch.tensor([list(dict_scores.values())])
    return scores


def getBatchSubgraph(subgraph_list: list):
    # print("Getting batch subgraph...")
    batchsize = len(subgraph_list)
    ent_delta_values = [0]
    batch_sampled_edges = []
    batch_idxs, abs_idxs = [], []
    query_sub_idxs = []
    edge_batch_idxs = []

    for batch_idx in range(batchsize):
        sub, topk_nodes, node_index, sampled_edges = subgraph_list[batch_idx]
        num_nodes = len(topk_nodes)
        ent_delta = sum(ent_delta_values)  # tính offset.

        # Adding ent_delta to make node indices unique in the batch
        sampled_edges[:, 0] = node_index[sampled_edges[:, 0]] + ent_delta
        sampled_edges[:, 2] = node_index[sampled_edges[:, 2]] + ent_delta
        batch_sampled_edges.append(sampled_edges)
        edge_batch_idxs += [batch_idx] * int(sampled_edges.shape[0])

        ent_delta_values.append(num_nodes)
        batch_idxs += [batch_idx] * num_nodes
        abs_idxs += topk_nodes.tolist()
        query_sub_idxs.append(int(node_index[sub]) + ent_delta)

    # [n_batch_ent]
    batch_idxs = torch.LongTensor(batch_idxs)
    # [n_batch_ent]
    abs_idxs = torch.LongTensor(abs_idxs)
    # [n_batch_edges, 3]
    batch_sampled_edges = torch.cat(batch_sampled_edges, dim=0)
    # [n_batch_edges]
    edge_batch_idxs = torch.LongTensor(edge_batch_idxs)
    # [n_batch]
    query_sub_idxs = torch.LongTensor(query_sub_idxs)

    return batch_idxs, abs_idxs, query_sub_idxs, edge_batch_idxs, batch_sampled_edges


def retry_on_exception(retry_limit: int = 3):
    if retry_limit < 1:
        raise ValueError("retry_limit must be >= 1")

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(1, retry_limit + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    last_exc = exc
                    if attempt < retry_limit:
                        time.sleep(random.choice((0.5, 1.0)))
                    else:
                        break
            raise last_exc

        return wrapper

    return decorator
