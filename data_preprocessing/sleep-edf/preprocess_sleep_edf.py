import os
import glob
import ntpath
import math
import numpy as np
import argparse
from datetime import datetime
from mne.io import concatenate_raws, read_raw_edf

# Label values
W = 0
N1 = 1
N2 = 2
N3 = 3
REM = 4
UNKNOWN = 5

stage_dict = {
    "W": W,
    "N1": N1,
    "N2": N2,
    "N3": N3,
    "REM": REM,
    "UNKNOWN": UNKNOWN
}

class_dict = {
    0: "W",
    1: "N1",
    2: "N2",
    3: "N3",
    4: "REM",
    5: "UNKNOWN"
}

ann2label = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
    "Sleep stage ?": 5,
    "Movement time": 5
}

EPOCH_SEC_SIZE = 30

def combine_to_subjects(root_dir, output_dir):
    sampling_rate = 100.
    files = os.listdir(root_dir)
    files = [os.path.join(root_dir, i) for i in files]

    files_dict = {}

    for i in files:
        subject_id = ntpath.basename(i).split("_")[0]
        if subject_id not in files_dict:
            files_dict[subject_id] = []
        files_dict[subject_id].append(i)

    os.makedirs(output_dir, exist_ok=True)
    for subject_id, subject_files in files_dict.items():
        X = []
        y = []
        for f in subject_files:
            data = np.load(f)
            X.append(data["x"])
            y.append(data["y"])
        X = np.concatenate(X, axis=0)
        y = np.concatenate(y, axis=0)
        save_dict = {
            "x": X,
            "y": y,
            "fs": sampling_rate,
        }
        np.savez(os.path.join(output_dir, f"{subject_id}.npz"), **save_dict)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/media/leo/Huy/Project/CA-TCC/data/sleep",
                        help="File path to the PSG and annotation files.")
    parser.add_argument("--output_dir", type=str, default="/media/leo/Huy/Project/CA-TCC/data_preprocessing/sleep-edf/sleepEDF20_fpzcz",
                        help="Directory where to save numpy files outputs.")
    parser.add_argument("--subjects_output_dir", type=str, default="/media/leo/Huy/Project/CA-TCC/data_preprocessing/sleep-edf/sleepEDF20_fpzcz_subjects",
                        help="Directory where to save numpy files outputs.")
    parser.add_argument("--select_ch", type=str, default="EEG Fpz-Cz",
                        help="The selected channel")
    args = parser.parse_args()

    # Output dir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    else:
        print(f"Output directory {args.output_dir} already exists.")

    # Select channel
    select_ch = args.select_ch

    # Read raw and annotation EDF files
    psg_fnames = glob.glob(os.path.join(args.data_dir, "*PSG.edf"))
    ann_fnames = glob.glob(os.path.join(args.data_dir, "*Hypnogram.edf"))
    psg_fnames.sort()
    ann_fnames.sort()
    psg_fnames = np.asarray(psg_fnames)
    ann_fnames = np.asarray(ann_fnames)

    print(f"Found {len(psg_fnames)} PSG files and {len(ann_fnames)} annotation files.")

    for i in range(len(psg_fnames)):
        print(f"Processing file {i+1}/{len(psg_fnames)}: {psg_fnames[i]}")
        raw = read_raw_edf(psg_fnames[i], preload=True, stim_channel=None)
        raw_ch_df = raw.to_data_frame(scaling_time=100.0)[select_ch]
        raw_ch = raw_ch_df.values

        ann = read_raw_edf(ann_fnames[i], preload=True, stim_channel=None)
        ann_df = ann.to_data_frame()
        labels = np.zeros((len(raw_ch) // int(EPOCH_SEC_SIZE * raw.info['sfreq'])), dtype=np.int32)

        for onset, duration, description in zip(ann_df['onset'], ann_df['duration'], ann_df['description']):
            label = ann2label[description[0]]
            start_idx = int(onset * raw.info['sfreq'] // EPOCH_SEC_SIZE)
            end_idx = start_idx + int(duration * raw.info['sfreq'] // EPOCH_SEC_SIZE)
            labels[start_idx:end_idx] = label

        select_idx = np.arange(len(raw_ch))
        extra_idx = np.where(labels == UNKNOWN)[0]
        if np.all(extra_idx > select_idx[-1]):
            n_label_trims = int(math.ceil(len(extra_idx) / (EPOCH_SEC_SIZE * raw.info['sfreq'])))
            if n_label_trims != 0:
                labels = labels[:-n_label_trims]
        print("after remove extra labels: {}, {}".format(select_idx.shape, labels.shape))

        raw_ch = raw_ch_df.values[select_idx]

        if len(raw_ch) % (EPOCH_SEC_SIZE * raw.info['sfreq']) != 0:
            raise Exception("Something wrong")
        n_epochs = len(raw_ch) / (EPOCH_SEC_SIZE * raw.info['sfreq'])

        x = np.asarray(np.split(raw_ch, n_epochs)).astype(np.float32)
        y = labels.astype(np.int32)

        assert len(x) == len(y)

        w_edge_mins = 30
        nw_idx = np.where(y != stage_dict["W"])[0]
        start_idx = nw_idx[0] - (w_edge_mins * 2)
        end_idx = nw_idx[-1] + (w_edge_mins * 2)
        if start_idx < 0: start_idx = 0
        if end_idx >= len(y): end_idx = len(y) - 1
        select_idx = np.arange(start_idx, end_idx + 1)
        print("Data before selection: {}, {}".format(x.shape, y.shape))
        x = x[select_idx]
        y = y[select_idx]
        print("Data after selection: {}, {}".format(x.shape, y.shape))

        filename = ntpath.basename(psg_fnames[i]).replace("-PSG.edf", ".npz")
        save_dict = {
            "x": x,
            "y": y,
            "fs": raw.info['sfreq'],
            "ch_label": select_ch,
            "header_raw": raw.info,
            "header_annotation": ann.info,
        }
        np.savez(os.path.join(args.output_dir, filename), **save_dict)
        print(f"Saved {filename} to {args.output_dir}")

        print("\n=======================================\n")

    combine_to_subjects(args.output_dir, args.subjects_output_dir)
    print(f"Combined subjects saved to {args.subjects_output_dir}")

if __name__ == "__main__":
    main()