import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
import torch.nn.functional as F
from torch.utils.data import DataLoader

from linear_probing import Classifier
from model import mae_vit_base_patch16
from data_loader import SubjectEEGDataset


# ===================== Config =====================
fold_idx = 0
data_dir = Path("your data path")
model_dir = Path("your model path")
classifier_dir = model_dir
save_path = Path("your save path")

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# ===================== Load Models =====================
encoder_path = model_dir / f"fold{fold_idx}_superlet.pth"
classifier_path = classifier_dir / f"fold{fold_idx}_superlet_classifier.pth"

encoder = mae_vit_base_patch16(img_size=(30, 100), patch_size=(5, 5), in_chans=1)
encoder.load_state_dict(torch.load(encoder_path, map_location=device))
encoder.to(device).eval()

classifier = Classifier()
classifier.load_state_dict(torch.load(classifier_path, map_location=device))
classifier.to(device).eval()


# ===================== Data =====================
all_subjects = sorted([f.stem for f in data_dir.glob("*.npz")])
test_subjects = all_subjects[122:122+31]


# ===================== Inference =====================
for subject in tqdm(test_subjects):
    dataset = SubjectEEGDataset([subject], data_dir)
    loader = DataLoader(dataset, batch_size=512, shuffle=False)

    all_softmax, all_labels = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            latent = encoder.forward_latent(x)
            logits = classifier(latent)
            probs = F.softmax(logits, dim=1)

            all_softmax.append(probs.cpu().numpy())
            all_labels.append(y.numpy())

    softmax = np.concatenate(all_softmax, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    out_path = save_path / f"{subject}_softmax.npz"
    np.savez_compressed(out_path, softmax=softmax, labels=labels)
    print(f"    [Saved] {out_path.name} | shape: {softmax.shape}")
