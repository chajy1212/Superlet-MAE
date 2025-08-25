import torch
import random
import numpy as np
import torch.nn as nn
from pathlib import Path
from argparse import ArgumentParser
from torch.utils.data import DataLoader

from models.model import mae_vit_base_patch16
from data.data_loader import SubjectEEGDataset

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score


random_seed = 777
np.random.seed(random_seed)
torch.manual_seed(random_seed)
random.seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ===================== Classifier =====================
class Classifier(nn.Module):
    """ Simple 2-layer MLP classifier for linear probing """
    def __init__(self, input_dim=256, hidden_dim=256, num_classes=5):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.fc(x)


# ===================== Feature Extraction =====================
def extract_latent_features(model, dataloader, device):
    """ Extract CLS token features from MAE encoder """
    latents, labels = [], []
    model.eval()

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            latent = model.forward_latent(x)
            latents.append(latent.cpu())
            labels.append(y.cpu())

    return torch.cat(latents, dim=0), torch.cat(labels, dim=0)


# ===================== Training =====================
def train_classifier(train_x, train_y, val_x, val_y, device, save_path=None, epochs=300, lr=0.001):
    model = Classifier(input_dim=train_x.shape[1]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(train_x.to(device))
        loss = criterion(outputs, train_y.to(device))
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        outputs = model(val_x.to(device))
        pred_y = outputs.argmax(dim=1).cpu()

    acc = accuracy_score(val_y, pred_y) * 100
    f1 = f1_score(val_y, pred_y, average='macro') * 100
    kappa = cohen_kappa_score(val_y, pred_y)

    if save_path:
        torch.save(model.state_dict(), save_path)
        print(f"[Saved Classifier] fold{fold_idx}_superlet_classifier.pth")

    return acc, f1, kappa


# ===================== Linear Probing =====================
def linear_probing(train_subjects, test_subjects, model_file, data_dir, device, fold_idx, classifier_save_path):
    """ Evaluate MAE encoder by training a shallow classifier on latent features """
    train_set = SubjectEEGDataset(train_subjects, data_dir)
    test_set = SubjectEEGDataset(test_subjects, data_dir)

    train_loader = DataLoader(train_set, batch_size=512, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=512, shuffle=False)

    # Load pretrained MAE model and freeze parameters
    model = mae_vit_base_patch16(img_size=(30, 100), patch_size=(5, 5), in_chans=1)
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.to(device)
    for param in model.parameters():
        param.requires_grad = False

    # Extract features
    train_x, train_y = extract_latent_features(model, train_loader, device)
    test_x, test_y = extract_latent_features(model, test_loader, device)

    # Train shallow classifier
    acc, f1, kappa = train_classifier(train_x, train_y, test_x, test_y, device, save_path=classifier_save_path)
    print(f"[Fold {fold_idx}] Accuracy: {acc:.2f}% | Macro F1: {f1:.2f}% | Kappa: {kappa:.4f}")

    return acc, f1, kappa


# ================ Main ================
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="your model path")
    parser.add_argument("--data_dir", type=str, default="your data path")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--n_subjects", type=int, default=122)
    parser.add_argument("--test_size", type=int, default=31)
    parser.add_argument("--mode", type=str, required=False, default="all",
                        choices=["superlet", "stft", "wavelet", "all"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = {"acc": [], "f1": [], "kappa": []}

    data_dir = Path(args.data_dir) / "superlet_transformed"
    model_dir = Path(args.model_dir)
    classifier_dir = model_dir

    all_subjects = sorted([f.stem for f in data_dir.glob("*.npz")])
    print(f"[INFO] Total subjects: {len(all_subjects)}")

    train_val_subjects = all_subjects[:args.n_subjects]
    test_subjects = all_subjects[args.n_subjects:args.n_subjects + args.test_size]

    kf = KFold(n_splits=args.folds, shuffle=False)

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(train_val_subjects)):
        train_subjects = [train_val_subjects[i] for i in train_idx]
        val_subjects = [train_val_subjects[i] for i in val_idx]

        print(f"[Fold {fold_idx}] Train: {len(train_subjects)} | Val: {len(val_subjects)} | Test: {len(test_subjects)}")

        model_file = model_dir / f"fold{fold_idx}_superlet.pth"
        if not model_file.exists():
            print(f"[Skip] {model_file.name} not found.")
            continue

        classifier_save_path = classifier_dir / f"fold{fold_idx}_superlet_classifier.pth"
        acc, f1, kappa = linear_probing(
            train_subjects, test_subjects,
            model_file, data_dir, device,
            fold_idx, classifier_save_path
        )

        results["acc"].append(acc)
        results["f1"].append(f1)
        results["kappa"].append(kappa)

    acc = np.array(results["acc"])
    f1 = np.array(results["f1"])
    kappa = np.array(results["kappa"])

    print(f"\n=== SUPERLET SUMMARY ===")
    print(f"Accuracy : {acc.mean():.3f} ± {acc.std(ddof=1):.3f}")
    print(f"Macro F1 : {f1.mean():.3f} ± {f1.std(ddof=1):.3f}")
    print(f"Kappa    : {kappa.mean():.3f} ± {kappa.std(ddof=1):.3f}")