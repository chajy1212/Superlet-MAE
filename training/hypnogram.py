import numpy as np
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid")

# ===== Config =====
input_dir = Path("your softmax npz path")
output_dir = Path("your softmax result path")

class_names = ["Wake", "REM", "N1", "N2", "N3"]
colors = sns.color_palette("viridis", 5)


# ===== Plotting =====
for file in sorted(input_dir.glob("*.npz")):
    subject_id = file.stem.replace("_softmax", "")
    print(f"[Processing] {subject_id}")

    # --- Load Data ---
    data = np.load(file)
    softmax = data["softmax"]  # (N, 5)
    labels = data["labels"]    # (N,)
    preds = np.argmax(softmax, axis=1)
    time = np.arange(softmax.shape[0])

    # --- Create Figure with 3 stacked subplots ---
    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1, 1.4]})

    # (A) Ground Truth
    axes[0].plot(time, labels, color="deepskyblue", linewidth=1)
    axes[0].set_yticks(np.arange(len(class_names)))
    axes[0].set_yticklabels(class_names)
    axes[0].set_ylabel("Sleep Stage")
    axes[0].set_title(f"(A) Ground Truth — {subject_id}")

    # (B) Predicted
    axes[1].plot(time, preds, color="black", linewidth=1)
    axes[1].set_yticks(np.arange(len(class_names)))
    axes[1].set_yticklabels(class_names)
    axes[1].set_ylabel("Sleep Stage")
    axes[1].set_title(f"(B) Predicted — {subject_id}")

    # (C) Softmax Probabilities
    axes[2].stackplot(
        time,
        softmax.T,
        labels=class_names,
        colors=colors,
        alpha=1.0,
        baseline="zero"
    )
    axes[2].set_ylabel("Probability")
    axes[2].set_xlabel("Epoch Index")
    axes[2].set_title(f"(C) Softmax Probabilities — {subject_id}")
    axes[2].legend(loc="upper center", ncol=5, fontsize=8)

    # --- Save & Close ---
    plt.tight_layout()
    out_path = output_dir / f"{subject_id}_hypnogram.eps"
    plt.savefig(out_path, format="eps", bbox_inches="tight")
    plt.close(fig)

    print(f"[Saved] {out_path.name}")