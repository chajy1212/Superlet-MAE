import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset

class SubjectEEGDataset(Dataset):
    """
        EEG dataset from .npz files. Each file contains {'x': (N, F, T), 'y': (N,)}.
        Output shape: x → (N, 1, F, T)
    """
    def __init__(self, subject_list, data_dir):
        self.data, self.labels = [], []
        data_dir = Path(data_dir)

        for subject in subject_list:
            npz_path = data_dir / f"{subject}.npz"
            data = np.load(npz_path)
            x = data['x']  # (N, F, T)
            y = data['y']  # (N,)

            x = torch.tensor(x, dtype=torch.float32).unsqueeze(1)  # (N, 1, F, T)
            y = torch.tensor(y, dtype=torch.long)

            self.data.append(x)
            self.labels.append(y)

        self.data = torch.cat(self.data, dim=0)
        self.labels = torch.cat(self.labels, dim=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]