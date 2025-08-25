import torch
import random
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from models.model import mae_vit_base_patch16
from data.data_loader import SubjectEEGDataset

from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score


random_seed = 777
np.random.seed(random_seed)
torch.manual_seed(random_seed)
random.seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


# ===================== Utils =====================
def print_model_size(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"[Model Size]")
    print(f"  Total Parameters     : {total_params:,}")
    print(f"  Trainable Parameters : {trainable_params:,}")
    print(f"  Model Memory Estimate: {total_params * 4 / (1024**2):.2f} MB (float32)\n")


def extract_latent(model, dataloader, device):
    latents, labels = [], []
    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            latent = model.forward_latent(x)
            latents.append(latent.cpu().numpy())
            labels.append(y.cpu().numpy())
    return np.concatenate(latents), np.concatenate(labels)


def reconstruction_visualization(imgs, recons, masks, epoch, fold, patch_size=5):
    img = imgs[0, 0].detach().cpu().numpy()             # (30, 100)
    recon = recons[0, 0].detach().cpu().numpy()         # (30, 100)
    mask = masks[0].detach().cpu().numpy()              # (H/P * W/P,) → flatten mask

    H, W = img.shape
    h, w = H // patch_size, W // patch_size
    mask = mask.reshape(h, w)
    mask = mask.repeat(patch_size, axis=0).repeat(patch_size, axis=1)  # (30, 100)

    cmap = plt.colormaps['viridis']
    img_norm = (img - img.min()) / (img.max() - img.min() + 1e-6)
    img_rgb = cmap(img_norm)[..., :3]     # (30, 100, 3), remove alpha channel
    img_rgb[mask == 1] = [0.5, 0.5, 0.5]  # gray for masked

    fig, axs = plt.subplots(3, 1, figsize=(10, 9))
    axs[0].imshow(img, cmap='viridis', aspect='auto')
    axs[0].set_title(f"Fold{fold} Epoch : {epoch} - Original")
    axs[1].imshow(img_rgb, aspect='auto')
    axs[1].set_title(f"Fold{fold} Epoch : {epoch} - Masked")
    axs[2].imshow(recon, cmap='viridis', aspect='auto')
    axs[2].set_title(f"Fold{fold} Epoch : {epoch} - Reconstructed")

    for ax in axs:
        ax.axis('off')

    plt.tight_layout()
    save_result_dir = Path("your path")
    plt.savefig(save_result_dir / f"fold{fold}_epoch{epoch}.eps", format='eps')
    plt.close()


# ===================== Training =====================
def run_fold(fold_idx, train_subjects, val_subjects, test_subjects, data_dir, device,
             num_epochs=300, batch_size=512, mask_ratio=0.75,
             img_size=(30, 100), patch_size=5, accumulation_steps=1):
    print(f"\n=== Fold {fold_idx} Training ===")

    train_set = SubjectEEGDataset(train_subjects, data_dir)
    val_set = SubjectEEGDataset(val_subjects, data_dir)
    test_set = SubjectEEGDataset(test_subjects, data_dir)

    print(f"\nTrain subjects: {len(train_subjects)} | Val subjects: {len(val_subjects)} | Test subjects: {len(test_subjects)}")
    print(f"Train Epochs: {len(train_set)} | Val Epochs: {len(val_set)} | Test Epochs: {len(test_set)}\n")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)

    model = mae_vit_base_patch16(
        img_size=img_size,
        patch_size=(patch_size, patch_size),
        in_chans=1,
        norm_pix_loss=False
    ).to(device)
    print_model_size(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0
        optimizer.zero_grad()

        for i, (x, _) in enumerate(train_loader):
            x = x.to(device)
            loss, pred, mask = model(x, mask_ratio=mask_ratio)
            loss = loss / accumulation_steps
            loss.backward()

            if i == 0:  # 첫 배치 시각화
                with torch.no_grad():
                    recon = model.unpatchify(pred)
                    reconstruction_visualization(x, recon, mask, epoch, fold_idx, patch_size)

            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * accumulation_steps

        avg_loss = total_loss / len(train_loader)

        train_x, train_y = extract_latent(model, val_loader, device)
        test_x, test_y = extract_latent(model, test_loader, device)
        knn = KNeighborsClassifier()
        knn.fit(train_x, train_y)
        pred_y = knn.predict(test_x)

        acc = accuracy_score(test_y, pred_y) * 100
        f1 = f1_score(test_y, pred_y, average="macro") * 100

        print(f"[Fold {fold_idx} | Epoch {epoch}] Loss: {avg_loss:.4f} | KNN Acc: {acc:.2f}% | Macro F1: {f1:.2f}%")

    save_model_dir = Path("your path")
    torch.save(model.state_dict(), save_model_dir, f"fold{fold_idx}_superlet.pth")
    print(f"\n[Save] Fold {fold_idx} model saved to {f'fold{fold_idx}_superlet.pth'}")


# ================ Main ================
if __name__ == "__main__":
    data_dir = Path("your path")

    subject_files = sorted(data_dir.glob("*.npz"))
    subjects = [f.stem for f in subject_files]

    train_val_subjects = subjects[:122]
    test_subjects = subjects[122:153]

    kf = KFold(n_splits=5)

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(train_val_subjects)):
        train_subjects = [train_val_subjects[i] for i in train_idx]
        val_subjects = [train_val_subjects[i] for i in val_idx]
        run_fold(fold_idx, train_subjects, val_subjects, test_subjects, data_dir, device, accumulation_steps=1)