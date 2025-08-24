import mne
import numpy as np
from pathlib import Path


def sleep_physionet_converter(src_path, trg_path, duration=30, ch_name="EEG Fpz-Cz"):
    """ Convert Sleep-EDFx EDF files into npz format. """
    annotation_desc_2_event_id = {
        'Sleep stage W': 0,
        'Sleep stage 1': 1,
        'Sleep stage 2': 2,
        'Sleep stage 3': 3,
        'Sleep stage 4': 3,  # merged into N3
        'Sleep stage R': 4,
    }

    # Alternative mappings
    event_id_1 = {
        'Sleep stage W': 0,
        'Sleep stage 1': 1,
        'Sleep stage 2': 2,
        'Sleep stage 3/4': 3,
        'Sleep stage R': 4,
    }

    event_id_2 = {
        'Sleep stage W': 0,
        'Sleep stage 1': 1,
        'Sleep stage 2': 2,
        'Sleep stage R': 4,
    }

    src_path, trg_path = Path(src_path), Path(trg_path)

    psg_files = sorted(src_path.glob('*PSG.edf'))
    hyp_files = sorted(src_path.glob('*Hypnogram.edf'))

    assert len(psg_files) == len(hyp_files), "Mismatch between PSG and Hypnogram files"

    for psg_file, hyp_file in zip(psg_files, hyp_files):
        raw = mne.io.read_raw_edf(psg_file, preload=True)
        annot = mne.read_annotations(hyp_file)

        # keep 30-min wake before/after sleep
        annot.crop(annot[1]['onset'] - 30 * 60, annot[-2]['onset'] + 30 * 60)
        raw.set_annotations(annot, emit_warning=False)

        # Select EEG channel and filter
        raw = raw.copy().pick([ch_name]).filter(0, 40)

        try:
            events, _ = mne.events_from_annotations(
                raw, event_id=annotation_desc_2_event_id, chunk_duration=duration
            )
        except Exception:
            print(f"[SKIP] Failed to extract events: {psg_file.name}")
            continue

        tmax = duration - 1.0 / raw.info['sfreq']

        try:
            epochs = mne.Epochs(raw, events, event_id=event_id_1,
                                tmin=0.0, tmax=tmax, baseline=None)
        except ValueError:
            epochs = mne.Epochs(raw, events, event_id=event_id_2,
                                tmin=0.0, tmax=tmax, baseline=None)

        data = epochs.get_data().squeeze()  # (n_epochs, 3000)
        labels = epochs.events[:, -1]       # (n_epochs,)

        out_name = psg_file.stem.split('-')[0].lower()
        out_path = trg_path / f"{out_name}.npz"

        np.savez(out_path, x=data, y=labels)
        print(f"[SAVE] {out_name}.npz | x: {data.shape}, y: {labels.shape}")


if __name__ == '__main__':
    sleep_physionet_converter(
        src_path="your path",
        trg_path="your path"
    )