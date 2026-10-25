import copy
import torch
from torch import nn
from torch.utils.data import DataLoader

from types import SimpleNamespace
import torch
from collections import defaultdict

# in_feature = 2 * n_rel (outgoing + incoming counts), set per dataset
IN_FEATURE = {
    "family":     25,   # placeholder (family has 12 relations, so 24)
    "fb15k_237":  475,   # placeholder
}

def make_args(dataset: str , in_features: int) -> SimpleNamespace:
    args = {
        "dataset":    dataset,
        "in_feature": in_features,
        "n_layer":    3,   # placeholder
        "lr":         5e-4,   # placeholder
        "batchsize":  32,   # placeholder
        "n_epoch":    200,   # placeholder
        "val_cycle":  5,   # placeholder
        "patience":   3,   # placeholder: validation checks without improvement before stopping
        "min_delta":  1e-4,   # placeholder: smallest val-loss drop that counts as improvement
        "device":     "cuda" if torch.cuda.is_available() else "cpu",
    }
    return SimpleNamespace(**args)


def make_finetune_args(dataset: str, in_features: int) -> SimpleNamespace:
    """Pretraining args with a 10x lower lr and per-epoch validation (the finetune set is small)."""
    args = make_args(dataset, in_features)
    args.lr = 5e-5
    args.val_cycle = 1
    args.patience = 5
    return args



class RelationInferenceModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(args.in_feature, args.in_feature), nn.ReLU(), nn.Dropout(0.1))
            for _ in range(args.n_layer)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = x + block(x)
        return x
    
class RelationInferenceData(torch.utils.data.Dataset):
    def __init__(self, data, args):
        groups = defaultdict(set)
        for x, y in data:
            groups[x].add(y.index(1))

        xs = list(groups)
        self.X = torch.tensor(xs, dtype=torch.float32)
        self.y = torch.zeros(len(xs), args.in_feature)
        for i, x in enumerate(xs):
            self.y[i, list(groups[x])] = 1.0
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    

class RelationInferenceTrainer():
    def __init__(self, model: RelationInferenceModel, data: RelationInferenceData, args,
                 val_data: RelationInferenceData = None, test_data: RelationInferenceData = None):
        """Without `val_data`, `data` is split 80/10/10 into train/val/test. With it, all of
        `data` is trained on, `val_data` drives early stopping and `test_data` (optional) is
        held for the final score."""
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.optmizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        self.args = args
        self.model = model.to(self.args.device)
        if val_data is None:
            g = torch.Generator().manual_seed(42)
            train_set, val_set, test_set = torch.utils.data.random_split(data, [0.8, 0.1, 0.1], generator=g)
        else:
            train_set, val_set, test_set = data, val_data, test_data
        self.trainLoader = DataLoader(train_set, batch_size=args.batchsize, shuffle=True)
        self.valLoader = DataLoader(val_set, batch_size=args.batchsize)
        self.testLoader = DataLoader(test_set, batch_size=args.batchsize) if test_set is not None else None

    @staticmethod
    def _filtered_ranks(pred, y):
        valid = y.bool()
        s_invalid = pred.masked_fill(valid, float("-inf"))
        above = (s_invalid.unsqueeze(1) > pred.unsqueeze(2)).sum(dim=2)
        ranks =  (above[valid] + 1)
        return ranks.float()
    
    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        total_loss, n, all_ranks = 0.0, 0, []
        for X, y in loader:
            X, y = X.to(self.args.device), y.to(self.args.device)
            pred = self.model(X)
            total_loss += self.loss_func(pred, y / y.sum(dim=1, keepdim=True)).item() * X.size(0)
            n += X.size(0)
            all_ranks.append(self._filtered_ranks(pred, y))
        ranks = torch.cat(all_ranks)
        return total_loss / n, (1.0 / ranks).mean().item(), (ranks <= 3).float().mean().item()

    def train(self, keep_baseline: bool = False):
        """Train with early stopping on val MRR and restore the best checkpoint.

        `keep_baseline=True` (for finetuning) scores the model before any update and uses that
        as the MRR to beat, so a finetune that never improves returns the starting weights.
        """
        best_mrr = 0
        best_state = None
        bad_checks = 0
        if keep_baseline:
            _, best_mrr, hit3 = self.evaluate(self.valLoader)
            best_state = copy.deepcopy(self.model.state_dict())
            print(f"Baseline _____ MRR: {best_mrr:.4f}  Hit@3: {hit3:.4f}")
        for epoch in range(self.args.n_epoch):
            self.model.train()
            print("---------------\nEpoch: ", epoch)
            epoch_train_loss = 0.0
            n = 0
            for batch in self.trainLoader:
                X, y = batch
                X, y = X.to(self.args.device), y.to(self.args.device)
                pred = self.model(X)
                loss = self.loss_func(pred, y / y.sum(dim=1, keepdim=True))
                self.optmizer.zero_grad()
                loss.backward()
                self.optmizer.step()
                epoch_train_loss += loss.item() * X.size(0)
                n += X.size(0)

            print(f".......... train loss: ", epoch_train_loss/n)

            if (epoch + 1) % self.args.val_cycle == 0:
                val_loss, mrr, hit3 = self.evaluate(self.valLoader)
                print(f"_____ val loss: {val_loss:.4f}")
                print(f"_____ MRR: {mrr:.2f}")
                print(f"_____ Hit@3: {hit3:.2f}")


                if mrr > best_mrr + self.args.min_delta:
                    best_mrr = mrr
                    best_state = copy.deepcopy(self.model.state_dict())
                    bad_checks = 0
                else:
                    bad_checks += 1
                    if bad_checks >= self.args.patience:
                        print(f"Early stopping at epoch {epoch} (best mrr: {best_mrr:.4f})")
                        break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return best_mrr


def finetune(pretrained_path, finetune_data, val_data, args, test_data=None):
    """Continue training a saved model on `finetune_data`, early-stopping on `val_data`.

    `finetune_data`, `val_data` and `test_data` are raw [(X, y), ...] samples, e.g. the leak
    samples of the BRINK train / val / test questions. Returns the model (best checkpoint,
    never worse on `val_data` than the pretrained one) and its trainer.
    """
    model = RelationInferenceModel(args)
    model.load_state_dict(torch.load(pretrained_path, map_location="cpu"))
    trainer = RelationInferenceTrainer(
        model,
        RelationInferenceData(finetune_data, args),
        args,
        val_data=RelationInferenceData(val_data, args),
        test_data=RelationInferenceData(test_data, args) if test_data is not None else None,
    )
    trainer.train(keep_baseline=True)
    return trainer.model, trainer

