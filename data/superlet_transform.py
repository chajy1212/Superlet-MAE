import torch
import librosa
import numpy as np
from tqdm import tqdm
from pathlib import Path
from jax import numpy as jnp
from superlets import adaptive_superlet_transform
from torchvision.transforms import Resize, Normalize


def normalize_mel(S, min_level_db=-100):
    """ Normalize spectrogram in dB scale to [0, 1] """
    return np.clip((S - min_level_db) / -min_level_db, 0, 1)


def convert_npz_to_superlet(npz_dir, save_dir):
    npz_dir, save_dir = Path(npz_dir), Path(save_dir)

    fs = 100
    freqs = jnp.linspace(1, 30, 30)
    base_cycle, min_order, max_order = 1, 5, 40

    resize_inst = Resize(size=(30, 100))
    normalize_inst = Normalize(mean=0.5, std=0.5)

    for npz_file in tqdm(sorted(npz_dir.glob("*.npz"))):
        try:
            data = np.load(npz_file)
            signals = data['x']  # (N, 3000)
            labels = data['y']   # (N,)

            subject_x, subject_y = [], []
            for signal, label in zip(signals, labels):
                spec = adaptive_superlet_transform(signal, freqs=freqs,
                                                   sampling_freq=fs,
                                                   base_cycle=base_cycle,
                                                   min_order=min_order,
                                                   max_order=max_order,
                                                   mode='add')
                spec = librosa.power_to_db(np.abs(spec))
                spec = normalize_mel(spec)

                # Torch transform pipeline: resize & normalize
                spec = torch.tensor(spec, dtype=torch.float32).unsqueeze(0)  # (1, F, T)
                spec = resize_inst(spec)                                     # (1, H, W)
                spec = normalize_inst(spec)

                subject_x.append(spec)
                subject_y.append(label)

            subject_x = torch.stack(subject_x).squeeze().numpy().astype(np.float32)
            subject_y = np.array(subject_y).astype(np.int64)

            out_path = save_dir / npz_file.name
            np.savez_compressed(out_path, x=subject_x, y=subject_y)

            print(f"    [Saved] {out_path.name} | x: {subject_x.shape}, y: {subject_y.shape}")

        except Exception as e:
            print(f"[Error] {npz_file.name} failed | {e}")


# ===================== main =====================
if __name__ == "__main__":
    convert_npz_to_superlet(
        npz_dir="your path",
        save_dir="your path"
    )