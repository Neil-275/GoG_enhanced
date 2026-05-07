import os
import csv

import torch
from torch_geometric.data import Data, InMemoryDataset, download_url

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


class LocalKnowledgeGraph(InMemoryDataset):

    """Transductive knowledge graph dataset from local TSV files.

    Expected raw files under `${root}/raw/`:
      - train.txt
      - val.txt (or valid.txt)
      - test.txt

    Processed output will be cached to `${root}/processed/data.pt` and contains
    3 PyG `Data` objects in the same shape as `IndRelLinkPredDataset`:
      - `dataset[0]`: train
      - `dataset[1]`: valid
      - `dataset[2]`: test

    Each `Data` has:
      - `edge_index`, `edge_type`, `num_nodes` for message passing (fact graph)
      - `target_edge_index`, `target_edge_type` for link prediction targets

    For message passing, inverse edges are added (relation ids are offset by +R),
    so `num_relations` equals `2R`.
    """

    def __init__(
        self,
        root,
        train_file="train.txt",
        valid_file="valid.txt",
        test_file="test.txt",
        delimiter="\t",
        transform=None,
        pre_transform=None,
        verbose=1,
    ):
        self.train_file = train_file
        self.valid_file = valid_file
        self.test_file = test_file
        self.delimiter = delimiter
        self.verbose = verbose
        super().__init__(root, transform=transform, pre_transform=pre_transform)
        # PyTorch 2.6+ defaults `weights_only=True`, which can't unpickle PyG `Data` objects.
        # This is a local, trusted cache file.
        try:
            self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        except TypeError:  # older PyTorch
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def num_relations(self):
        data = getattr(self, "_data", None)
        if data is None:
            data = self.data
        return int(data.edge_type.max()) + 1

    @property
    def raw_dir(self):
        # Support both layouts:
        # 1) `${root}/raw/{train,val,test}.txt` (default PyG convention)
        # 2) `${root}/{train,val,test}.txt` (user already has a prepared folder)
        direct_train = os.path.join(self.root, self.train_file)
        direct_valid = os.path.join(self.root, self.valid_file)
        direct_test = os.path.join(self.root, self.test_file)
        if os.path.exists(direct_train) and (os.path.exists(direct_valid) or os.path.exists(os.path.join(self.root, "val.txt")) or os.path.exists(os.path.join(self.root, "valid.txt"))) and os.path.exists(direct_test):
            return self.root
        return self.root

    @property
    def processed_dir(self):
        return os.path.join(self.root, "processed")

    @property
    def processed_file_names(self):
        return "data.pt"

    def _resolve_valid_file(self):
        # Prefer the configured filename, then fall back between valid.txt <-> val.txt.
        configured = os.path.join(self.raw_dir, self.valid_file)
        if os.path.exists(configured):
            return self.valid_file
        for alt in ("val.txt", "valid.txt"):
            alt_path = os.path.join(self.raw_dir, alt)
            if os.path.exists(alt_path):
                return alt
        return self.valid_file

    @property
    def raw_file_names(self):
        valid_file = self._resolve_valid_file()
        return [self.train_file, valid_file, self.test_file]

    def download(self):
        # Local dataset: nothing to download.
        return

    def _read_triples(self, path):
        triples = []
        if self.verbose and self.verbose >= 2:
            full_path = os.path.abspath(path)
        with open(path, "r", newline="") as fin:
            reader = csv.reader(fin, delimiter=self.delimiter)
            if self.verbose and tqdm is not None:
                reader = tqdm(reader, desc=f"Loading {os.path.basename(path)}")
            for tokens in reader:
                if not tokens:
                    continue
                if len(tokens) != 3:
                    raise ValueError(
                        "Expect 3 columns (head, relation, tail) in `%s`, but got %d columns: %s"
                        % (path, len(tokens), tokens)
                    )
                h_token, r_token, t_token = tokens
                triples.append((h_token, r_token, t_token))
        return triples

    def process(self):
        train_path, valid_path, test_path = self.raw_paths

        train_triples = self._read_triples(train_path)
        valid_triples = self._read_triples(valid_path)
        test_triples = self._read_triples(test_path)

        inv_entity_vocab = {}
        inv_relation_vocab = {}

        def map_triples(triples):
            mapped = []
            for h_token, r_token, t_token in triples:
                if h_token not in inv_entity_vocab:
                    inv_entity_vocab[h_token] = len(inv_entity_vocab)
                if t_token not in inv_entity_vocab:
                    inv_entity_vocab[t_token] = len(inv_entity_vocab)
                if r_token not in inv_relation_vocab:
                    inv_relation_vocab[r_token] = len(inv_relation_vocab)
                h = inv_entity_vocab[h_token]
                t = inv_entity_vocab[t_token]
                r = inv_relation_vocab[r_token]
                mapped.append((h, t, r))
            return torch.tensor(mapped, dtype=torch.long)

        train_triplets = map_triples(train_triples)
        valid_triplets = map_triples(valid_triples)
        test_triplets = map_triples(test_triples)

        num_nodes = len(inv_entity_vocab)
        num_relations = len(inv_relation_vocab)
        
        # save mapping dictionaries:
        import pickle as pkl
        entity_vocab_path = os.path.join(self.processed_dir, "entity2id.pkl")
        relation_vocab_path = os.path.join(self.processed_dir, "relation2id.pkl")
        with open(entity_vocab_path, "wb") as fout:
            pkl.dump(inv_entity_vocab, fout)
        with open(relation_vocab_path, "wb") as fout:
            pkl.dump(inv_relation_vocab, fout)
        if self.verbose:
            print(f"Saved entity vocabulary to {entity_vocab_path}")
            print(f"Saved relation vocabulary to {relation_vocab_path}")

        def build_fact_graph(triplets):
            edge_index = triplets[:, :2].t().contiguous()
            edge_type = triplets[:, 2].contiguous()
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=-1)
            edge_type = torch.cat([edge_type, edge_type + num_relations], dim=-1)
            return edge_index, edge_type

        # Fact graphs for message passing.
        train_fact_index, train_fact_type = build_fact_graph(train_triplets)
        test_fact_triplets = torch.cat([train_triplets, valid_triplets], dim=0)
        test_fact_index, test_fact_type = build_fact_graph(test_fact_triplets)

        # Targets for scoring (keep original relation ids 0..R-1).
        train_target_index = train_triplets[:, :2].t().contiguous()
        train_target_type = train_triplets[:, 2].contiguous()
        valid_target_index = valid_triplets[:, :2].t().contiguous()
        valid_target_type = valid_triplets[:, 2].contiguous()
        test_target_index = test_triplets[:, :2].t().contiguous()
        test_target_type = test_triplets[:, 2].contiguous()

        train_data = Data(
            edge_index=train_fact_index,
            edge_type=train_fact_type,
            num_nodes=num_nodes,
            target_edge_index=train_target_index,
            target_edge_type=train_target_type,
        )
        valid_data = Data(
            edge_index=train_fact_index,
            edge_type=train_fact_type,
            num_nodes=num_nodes,
            target_edge_index=valid_target_index,
            target_edge_type=valid_target_type,
        )
        test_data = Data(
            edge_index=test_fact_index,
            edge_type=test_fact_type,
            num_nodes=num_nodes,
            target_edge_index=test_target_index,
            target_edge_type=test_target_type,
        )

        if self.pre_transform is not None:
            train_data = self.pre_transform(train_data)
            valid_data = self.pre_transform(valid_data)
            test_data = self.pre_transform(test_data)

        torch.save(self.collate([train_data, valid_data, test_data]), self.processed_paths[0])

    def __repr__(self):
        return "LocalKnowledgeGraph()"

    # def load_vocab(self):
    #     """Load entity and relation vocabularies from processed pickle files.
        
    #     Returns:
    #         tuple: (inv_entity_vocab, inv_relation_vocab) dictionaries
    #     """
    #     import pickle as pkl
    #     entity_vocab_path = os.path.join(self.processed_dir, "entity2id.pkl")
    #     relation_vocab_path = os.path.join(self.processed_dir, "relation2id.pkl")
        
    #     with open(entity_vocab_path, "rb") as fin:
    #         inv_entity_vocab = pkl.load(fin)
    #     with open(relation_vocab_path, "rb") as fin:
    #         inv_relation_vocab = pkl.load(fin)
        
    #     return inv_entity_vocab, inv_relation_vocab

class IndRelLinkPredDataset(InMemoryDataset):

    urls = {
        "FB15k-237": [
            "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s_ind/train.txt",
            "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s_ind/test.txt",
            "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s/train.txt",
            "https://raw.githubusercontent.com/kkteru/grail/master/data/fb237_%s/valid.txt"
        ],
        "WN18RR": [
            "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s_ind/train.txt",
            "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s_ind/test.txt",
            "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s/train.txt",
            "https://raw.githubusercontent.com/kkteru/grail/master/data/WN18RR_%s/valid.txt"
        ]
    }

    def __init__(self, root, name, version, transform=None, pre_transform=None):
        self.name = name
        self.version = version
        assert name in ["FB15k-237", "WN18RR"]
        assert version in ["v1", "v2", "v3", "v4"]
        super().__init__(root, transform, pre_transform)
        try:
            self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        except TypeError:  # older PyTorch
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def num_relations(self):
        data = getattr(self, "_data", None)
        if data is None:
            data = self.data
        return int(data.edge_type.max()) + 1

    @property
    def raw_dir(self):
        return os.path.join(self.root, self.name, self.version, "raw")

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.name, self.version, "processed")

    @property
    def processed_file_names(self):
        return "data.pt"

    @property
    def raw_file_names(self):
        return [
            "train_ind.txt", "test_ind.txt", "train.txt", "valid.txt"
        ]

    def download(self):
        for url, path in zip(self.urls[self.name], self.raw_paths):
            download_path = download_url(url % self.version, self.raw_dir)
            os.rename(download_path, path)

    def process(self):
        test_files = self.raw_paths[:2]
        train_files = self.raw_paths[2:]

        inv_train_entity_vocab = {}
        inv_test_entity_vocab = {}
        inv_relation_vocab = {}
        triplets = []
        num_samples = []

        for txt_file in train_files:
            with open(txt_file, "r") as fin:
                num_sample = 0
                for line in fin:
                    h_token, r_token, t_token = line.strip().split("\t")
                    if h_token not in inv_train_entity_vocab:
                        inv_train_entity_vocab[h_token] = len(inv_train_entity_vocab)
                    h = inv_train_entity_vocab[h_token]
                    if r_token not in inv_relation_vocab:
                        inv_relation_vocab[r_token] = len(inv_relation_vocab)
                    r = inv_relation_vocab[r_token]
                    if t_token not in inv_train_entity_vocab:
                        inv_train_entity_vocab[t_token] = len(inv_train_entity_vocab)
                    t = inv_train_entity_vocab[t_token]
                    triplets.append((h, t, r))
                    num_sample += 1
            num_samples.append(num_sample)

        for txt_file in test_files:
            with open(txt_file, "r") as fin:
                num_sample = 0
                for line in fin:
                    h_token, r_token, t_token = line.strip().split("\t")
                    if h_token not in inv_test_entity_vocab:
                        inv_test_entity_vocab[h_token] = len(inv_test_entity_vocab)
                    h = inv_test_entity_vocab[h_token]
                    assert r_token in inv_relation_vocab
                    r = inv_relation_vocab[r_token]
                    if t_token not in inv_test_entity_vocab:
                        inv_test_entity_vocab[t_token] = len(inv_test_entity_vocab)
                    t = inv_test_entity_vocab[t_token]
                    triplets.append((h, t, r))
                    num_sample += 1
            num_samples.append(num_sample)
        triplets = torch.tensor(triplets)

        edge_index = triplets[:, :2].t()
        edge_type = triplets[:, 2]
        num_relations = int(edge_type.max()) + 1

        train_fact_slice = slice(None, sum(num_samples[:1]))
        test_fact_slice = slice(sum(num_samples[:2]), sum(num_samples[:3]))
        train_fact_index = edge_index[:, train_fact_slice]
        train_fact_type = edge_type[train_fact_slice]
        test_fact_index = edge_index[:, test_fact_slice]
        test_fact_type = edge_type[test_fact_slice]
        # add flipped triplets for the fact graphs
        train_fact_index = torch.cat([train_fact_index, train_fact_index.flip(0)], dim=-1)
        train_fact_type = torch.cat([train_fact_type, train_fact_type + num_relations])
        test_fact_index = torch.cat([test_fact_index, test_fact_index.flip(0)], dim=-1)
        test_fact_type = torch.cat([test_fact_type, test_fact_type + num_relations])

        train_slice = slice(None, sum(num_samples[:1]))
        valid_slice = slice(sum(num_samples[:1]), sum(num_samples[:2]))
        test_slice = slice(sum(num_samples[:3]), sum(num_samples))
        train_data = Data(edge_index=train_fact_index, edge_type=train_fact_type, num_nodes=len(inv_train_entity_vocab),
                          target_edge_index=edge_index[:, train_slice], target_edge_type=edge_type[train_slice])
        valid_data = Data(edge_index=train_fact_index, edge_type=train_fact_type, num_nodes=len(inv_train_entity_vocab),
                          target_edge_index=edge_index[:, valid_slice], target_edge_type=edge_type[valid_slice])
        test_data = Data(edge_index=test_fact_index, edge_type=test_fact_type, num_nodes=len(inv_test_entity_vocab),
                         target_edge_index=edge_index[:, test_slice], target_edge_type=edge_type[test_slice])

        if self.pre_transform is not None:
            train_data = self.pre_transform(train_data)
            valid_data = self.pre_transform(valid_data)
            test_data = self.pre_transform(test_data)

        torch.save((self.collate([train_data, valid_data, test_data])), self.processed_paths[0])

    def __repr__(self):
        return "%s()" % self.name
