import argparse
import os
import time

import numpy as np
import torch

from load_data import DataLoader
from base_model import BaseModel
from PPR_sampler import pprSampler


def _infer_dataset_name(data_path: str) -> str:
    parts = data_path.replace('\\', '/').rstrip('/').split('/')
    return parts[-1] if parts else 'dataset'


def _read_triples_file(path: str, loader: DataLoader | None = None):
    """Read triples from a txt file.

    Supports either integer ids: "h r t" (ints)
    or string ids (if loader is provided): "entity relation entity".
    """
    triples = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            h, r, t = line.split()
            try:
                triples.append((int(h), int(r), int(t)))
            except ValueError:
                if loader is None:
                    raise ValueError(
                        f"Non-integer triple found in {path}, but no loader was provided for name->id mapping. "
                        f"Offending line: {line}"
                    )
                triples.append((loader.entity2id[h], loader.relation2id[r], loader.entity2id[t]))
    return triples


def main():
    parser = argparse.ArgumentParser(description="Demo: run top-k tail prediction for one (h, r) query")

    # Data / checkpoint
    parser.add_argument('--data_path', type=str, default='data/FB15k_237/')
    parser.add_argument('--weight', type=str, required=True, help='Path to a saved .pt checkpoint')

    # Inference graph (user-provided)
    parser.add_argument(
        '--infer_graph_path',
        type=str,
        required=True,
        help='Path to triples file that defines the graph used for inference subgraph extraction',
    )
    parser.add_argument(
        '--infer_cache_dir',
        type=str,
        default='./infer_cache/',
        help='Where to store/reuse PPR cache for the inference graph',
    )
    parser.add_argument(
        '--add_idd_edges',
        action='store_true',
        help='Append identity (self-loop) edges like training does',
    )

    # Query
    parser.add_argument('--entity_id', type=int, required=True, help='Head entity id (global id)')
    parser.add_argument('--relation_id', type=int, required=True, help='Relation id (global id)')
    parser.add_argument('--pred_topk', type=int, default=10, help='How many tail predictions to return')

    # Sampler settings (match training defaults)
    parser.add_argument('--topk', type=float, default=0.1, help='Subgraph sampling ratio (topk * n_ent nodes)')
    parser.add_argument('--topm', type=float, default=-1, help='Edge sampling ratio (topm * |facts| edges); -1 disables')
    parser.add_argument('--add_manual_edges', action='store_true')

    # Runtime
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--batchsize', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1234)

    # Data split behavior (only used to construct loader.fact_data etc.)
    parser.add_argument('--fact_ratio', type=float, default=0.85)
    parser.add_argument('--remove_1hop_edges', default=True)
    parser.add_argument('--not_shuffle_train', default=True)

    # Model architecture (MUST match the checkpoint)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--attn_dim', type=int, default=32)
    parser.add_argument('--n_layer', type=int, default=6)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--act', type=str, default='relu', choices=['relu', 'tanh', 'idd'])
    parser.add_argument('--initializer', type=str, default='binary', choices=['binary', 'relation'])
    parser.add_argument('--concatHidden', action='store_true')
    parser.add_argument('--shortcut', action='store_true')
    parser.add_argument('--readout', type=str, default='linear', choices=['linear', 'multiply'])

    # Optimizer placeholders (BaseModel requires them; not used for pure inference)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lamb', type=float, default=0.0)

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(8, args.cpu))
    torch.multiprocessing.set_sharing_strategy('file_system')

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)

    # Required by BaseModel
    args.n_batch = args.n_tbatch = int(args.batchsize)

    dataset_name = _infer_dataset_name(args.data_path)
    print(f"==> dataset: {dataset_name}")

    # Build loaders (entities/relations + filters + n_ent/n_rel)
    loader = DataLoader(args, mode='train')
    val_loader = DataLoader(args, mode='valid')
    test_loader = DataLoader(args, mode='test')
    args.n_ent = loader.n_ent
    args.n_rel = loader.n_rel

    # Build inference sampler from the USER-PROVIDED graph (can take time if PPR cache is missing)
    args.n_samp_ent = int(args.topk * loader.n_ent)
    args.n_samp_edge = int(args.topm * len(loader.fact_data)) if args.topm > 0 else -1
    print(f"==> #sampled entities: {args.n_samp_ent}, #sampled edges: {args.n_samp_edge}")

    infer_triples = _read_triples_file(args.infer_graph_path, loader=loader)
    infer_edge_index = np.array(infer_triples, dtype=np.int64)
    if args.add_idd_edges:
        infer_edge_index = np.concatenate([infer_edge_index, loader.idd_data], axis=0)

    infer_homo_edges = list(set([(int(h), int(t)) for (h, r, t) in infer_edge_index]))

    os.makedirs(args.infer_cache_dir, exist_ok=True)
    infer_sampler = pprSampler(
        loader.n_ent,
        loader.n_rel,
        args.n_samp_ent,
        args.n_samp_edge,
        infer_homo_edges,
        infer_edge_index,
        args.infer_cache_dir,
        split='infer',
        args=args,
    )

    # Attach the inference sampler to all loaders for consistency (optional)
    loader.addSampler(infer_sampler)
    val_loader.addSampler(infer_sampler)
    test_loader.addSampler(infer_sampler)

    # Build BaseModel + load checkpoint
    model = BaseModel(args, loaders=(loader, val_loader, test_loader), samplers=(infer_sampler, infer_sampler))
    model.loadModel(args.weight)

    # Run prediction
    t0 = time.time()
    scores, pred_ids = model.predict_topk(
        entity_id=args.entity_id,
        relation_id=args.relation_id,
        topk=args.pred_topk,
        sampler=infer_sampler,
    )
    dt = time.time() - t0

    print(f"==> query: (h={args.entity_id}, r={args.relation_id})")
    print(f"==> top-{args.pred_topk} tails (took {dt:.3f}s):")
    for rank, (tid, score) in enumerate(zip(pred_ids.tolist(), scores.tolist()), start=1):
        print(f"{rank:02d}. t={tid}  score={score:.6f}")


if __name__ == '__main__':
    main()
