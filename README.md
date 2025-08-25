# Superlet-MAE: Self-Supervised Masked Autoencoding for Sleep Staging Using Single-Channel EEG


## 🚀 Abstract
Sleep stage classification is critical for diagnosing sleep disorders, but traditional polysomnography (PSG) is time-consuming and labor-intensive and subject to inter-rater variability, limiting its reliability.
In this study, we propose a self-supervised learning (SSL) framework that combines the Superlet Transform (SLT), which enables high-resolution time-frequency analysis, with a Masked Autoencoder (MAE) architecture.
Single-channel EEG signals (i.e., Fpz-Cz) from the Sleep-EDF dataset were transformed into Superlet scalograms and used as MAE input.
During training, 75% of the input patches were randomly masked, and reconstruction loss was computed only over the masked regions.
The proposed model outperformed those using Short-Time Fourier Transform (STFT) and Continuous Wavelet Transform (CWT) inputs across all evaluation metrics—Accuracy (ACC): 75.87%, Macro F1 Score (MF1): 63.23%, and Cohen’s Kappa (Kappa): 0.64—and exceeded prior SSL methods by up to 6.6% in ACC.
Reconstruction visualizations further confirmed that key EEG patterns were effectively recovered despite high masking.
This study is the first to apply Superlet scalograms to MAE for EEG representation learning, demonstrating its potential for accurate sleep staging from unlabeled data.


## 📊 Key Results

### 🔹 Comparison of Time-Frequency Representation Methods
| Transform Type |      Accuracy      |   Macro F1 Score   |   Cohen’s Kappa   |
|:--------------:|:------------------:|:------------------:|:-----------------:|
| **_Superlet_** | **_75.87 ± 1.39_** | **_63.23 ± 1.74_** | **_0.64 ± 0.02_** |
|      STFT      |    74.81 ± 1.27    |    61.74 ± 1.76    |    0.62 ± 0.02    |
|      CWT       |    75.45 ± 1.42    |    62.84 ± 1.16    |    0.63 ± 0.02    |

Superlet achieved the **_highest average scores across all metrics_**, with relatively smaller standard deviations, suggesting more consistent and robust classification performance.

---

### 🔹 Comparison with Other Self-supervised Methods
| Model      |      Accuracy      |   Macro F1 Score   |   Cohen’s Kappa   |
|:----------:|:------------------:|:------------------:|:-----------------:|
| **_Ours_** | **_75.87 ± 1.39_** | **_63.23 ± 1.74_** | **_0.64 ± 0.02_** |
| MAEEG      |    72.29 ± 4.56    |    62.58 ± 6.02    |    0.62 ± 0.08    |
| TS-TCC     |    69.27 ± 8.15    |   54.09 ± 18.89    |    0.55 ± 0.15    |
| BENDR      |    70.73 ± 8.57    |    62.04 ± 8.03    |    0.60 ± 0.11    |

Superlet-MAE outperformed all baselines:  
- **+3.58%** higher ACC than **_MAEEG_**
- **+6.6%** higher ACC than **_TS-TCC_** 
- **+5.14%** higher ACC than **_BENDR_**


## 📈 Visualization

### MAE Reconstruction (Mask Ratio = 0.75)
<p align="center">
  <img src="./figures/reconstruction.jpg" width="900"/>
</p>
<p align="center">
  <em>Visualization of MAE reconstruction performance (mask ratio = 0.75).<br>
  (A) original scalogram, (B) masked scalogram, (C) reconstructed scalogram.</em>
</p>


## 📂 Repository Structure
```
Superlet-MAE/
│
├── data/
│   ├── data_loader.py          # EEG dataset loader
│   ├── morlet.py               # Morlet wavelet utilities
│   ├── preprocess.py           # Convert raw Sleep-EDF to .npz
│   ├── superlet_transform.py   # Apply Superlet Transform
│   └── superlets.py            # Superlet implementation
│
├── figures/
│   ├── hypnogram.jpg           # Example hypnogram plot
│   └── reconstruction.jpg      # Example reconstruction plot
│
├── models/
│   ├── model.py                # MAE (ViT backbone) implementation
│   └── pos_embed.py            # Positional embedding functions
│
├── training/
│   ├── linear_probing.py       # Linear probing evaluation
│   └── train.py                # MAE pretraining script
│
├── requirements.txt            # Dependencies
└── README.md                   # Project documentation
```


## ⚙️ Installation
```
$ git clone https://github.com/chajy1212/Superlet-MAE.git
$ cd Superlet-MAE
$ pip install -r requirements.txt
```


## 🏃 Usage

### 1. Download Sleep EDF Dataset  
Download the Sleep-EDF dataset (e.g., Fpz-Cz channel) from [PhysioNet](https://www.physionet.org/content/sleep-edfx/1.0.0/).

### 2. Preprocess raw EEG data  
Use the preprocessing script to convert raw EDF files into NPZ format:
```bash
python data/preprocess.py
```

### 3. Apply Superlet Transform
Transform the preprocessed EEG data into time–frequency representations:
```bash
python data/superlet_transform.py
```

### 4. Pretrain the MAE Model
Run self-supervised pretraining of the Masked Autoencoder.  
Trained models are saved under `./models/`.
```bash
python training/train.py
```

### 5. Evaluate using Linear Probing
Train a shallow classifier on the encoder’s latent features:
```bash
python training/linear_probing.py
```


## 📌 Acknowledgements

This work was supported by BK21 Four Institute of Precision Public Health.
