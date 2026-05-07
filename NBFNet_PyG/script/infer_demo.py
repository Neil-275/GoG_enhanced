import os
import sys
import argparse

import torch
from torch_geometric.data import Data

# Make `import nbfnet` work when running as `python script/infer_demo.py`
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from nbfnet import util
from nbfnet.inference import predict_topk


def _parse_args():
    parser = argparse.ArgumentParser(description="Demo: run NBFNet-PyG top-k inference")
    parser.add_argument("-c", "--config", required=True, help="YAML config file")
    parser.add_argument("-s", "--seed", type=int, default=1024, help="Random seed")

    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint path override")

    parser.add_argument("--k", type=int, default=10, help="Top-k candidates to return")
    parser.add_argument(
        "--mode",
        choices=["both", "tail", "head"],
        default="both",
        help="Which query to run: (h,r,?) tail, (?,r,t) head, or both",
    )

    parser.add_argument("--head", type=int, default=None, help="Head entity id for (h,r,?)")
    parser.add_argument("--tail", type=int, default=None, help="Tail entity id for (?,r,t)")
    parser.add_argument("--relation", type=int, default=None, help="Relation id")

    parser.add_argument(
        "--filter_known",
        action="store_true",
        help="Mask out known true triples from the results (filtered setting)",
    )
    parser.add_argument(
        "--promote_known",
        action="store_true",
        help="When NOT filtering, boost known true triples to rank highest",
    )

    args, unparsed = parser.parse_known_args()

    # Parse dynamic variables referenced in the config (Jinja2 template variables)
    vars_in_cfg = util.detect_variables(args.config)
    if vars_in_cfg:
        parser_vars = argparse.ArgumentParser(add_help=False)
        for var in vars_in_cfg:
            parser_vars.add_argument(f"--{var}", required=True)
        cfg_vars = parser_vars.parse_known_args(unparsed)[0]
        context = {k: util.literal_eval(v) for k, v in cfg_vars._get_kwargs()}
    else:
        context = {}

    return args, context


def _build_filtered_data(dataset, train_data, device):
    """Build a Data(edge_index, edge_type) graph used for filtered ranking / known-triple masking."""
    full_graph = getattr(dataset, "_data", None)
    if full_graph is None:
        full_graph = dataset.data

    filtered_data = Data(
        edge_index=full_graph.target_edge_index,
        edge_type=full_graph.target_edge_type,
        num_nodes=train_data.num_nodes,
    )
    return filtered_data.to(device)


def main():
    args, context = _parse_args()

    cfg = util.load_config(args.config, context=context)
    if args.checkpoint is not None:
        cfg.checkpoint = args.checkpoint

    dataset_class = cfg.dataset.get("class")
    is_inductive = bool(dataset_class) and str(dataset_class).startswith("Ind")

    torch.manual_seed(args.seed + util.get_rank())

    # Build dataset + model
    dataset = util.build_dataset(cfg)
    cfg.model.num_relation = dataset.num_relations

    model = util.build_model(cfg)
    device = util.get_device(cfg)
    model = model.to(device)

    train_data, valid_data, test_data = dataset[0], dataset[1], dataset[2]
    train_data = train_data.to(device)
    valid_data = valid_data.to(device)
    test_data = test_data.to(device)

    filtered_data = None if is_inductive else _build_filtered_data(dataset, train_data, device)

    # Choose a demo triple if not provided
    if args.head is None or args.tail is None or args.relation is None:
        demo_triplet = torch.cat(
            [test_data.target_edge_index, test_data.target_edge_type.unsqueeze(0)], dim=0
        )[:, 0]
        demo_h, demo_t, demo_r = demo_triplet.tolist()
        head = demo_h if args.head is None else args.head
        tail = demo_t if args.tail is None else args.tail
        rel = demo_r if args.relation is None else args.relation
    else:
        head, tail, rel = args.head, args.tail, args.relation

    print(f"Device: {device}")
    print(f"Query triple IDs: (h={head}, r={rel}, t={tail})")
    if getattr(cfg, "checkpoint", None):
        print(f"Checkpoint: {cfg.checkpoint}")
    else:
        print("Checkpoint: <none specified in config/args>")

    # Use the graph for message passing; use filtered_data (if available) for known-triple masking.
    filter_graph = filtered_data if filtered_data is not None else test_data

    if args.mode in ("both", "tail"):
        topk_ent, topk_score = predict_topk(
            model,
            graph=test_data,
            entity=head,
            relation=rel,
            k=args.k,
            predict="tail",
            filter_known=args.filter_known,
            filter_graph=filter_graph,
            promote_known=args.promote_known,
        )
        print("\nTop-k tails for (h, r, ?):")
        for i, (eid, s) in enumerate(zip(topk_ent.tolist(), topk_score.tolist()), start=1):
            print(f"  {i:>2}. tail={eid:<8} score={s:.6f}")

    if args.mode in ("both", "head"):
        topk_ent, topk_score = predict_topk(
            model,
            graph=test_data,
            entity=tail,
            relation=rel,
            k=args.k,
            predict="head",
            filter_known=args.filter_known,
            filter_graph=filter_graph,
            promote_known=args.promote_known,
        )
        print("\nTop-k heads for (?, r, t):")
        for i, (eid, s) in enumerate(zip(topk_ent.tolist(), topk_score.tolist()), start=1):
            print(f"  {i:>2}. head={eid:<8} score={s:.6f}")


if __name__ == "__main__":
    main()
