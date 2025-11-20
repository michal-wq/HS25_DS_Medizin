from functions import filter_and_segment_data, save_to_csv, split_data_ecg
from pathlib import Path
import numpy as np
import wfdb
import scipy.stats as stats
import pandas as pd
from sklearn.model_selection import train_test_split
from functions import print_eta, _format_seconds
import time
import numpy as np

def main():
    path = 'data/physionet.org/files/ptb-xl/1.0.3/'
    sampling_rate = 500
    cols = ['scp_codes', 'strat_fold', 'filename_lr', 'filename_hr']

    # Save Data
    base_dir = Path('ready_data')
    X_path = base_dir / 'X'
    y_path = base_dir / 'y'

    print(f'Phase 1: Split data, sampling rate: {sampling_rate}')

    # Data split
    X_train_raw, X_test_raw, y_train, y_test = split_data_ecg(path, sampling_rate, cols)
    print(pd.Series(y_train).isna().mean()*100)
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_train_raw,
        y_train,
        test_size=0.2,
        random_state=42,
        stratify=y_train
    )

    print('='*100)
    print(f'Train raw shape: { X_train_raw.shape}')
    print(f'Val raw shape: { X_val_raw.shape}')
    print('=' * 100)

    # Filtern und Segmentierung
    print('=' * 100)
    print('filter und segmentierung von X_train')
    X_train_seg, keep_idx_train = filter_and_segment_data(X_train_raw, sampling_rate=sampling_rate, max_peaks=5)
    y_train = y_train[keep_idx_train]

    # Train
    print('=' * 100)
    save_to_csv(X_train_seg, X_path / 'train.csv')
    save_to_csv(y_train, y_path / 'train.csv')
    del X_train_raw
    del X_train_seg
    del y_train

    # Val
    print('=' * 100)
    print('filter und segmentierung von X_val')
    X_val_seg, keep_idx_val = filter_and_segment_data(X_val_raw, sampling_rate=sampling_rate, max_peaks=5)
    y_val = y_val[keep_idx_val]
    print('=' * 100)
    save_to_csv(X_val_seg, X_path / 'val.csv')
    save_to_csv(y_val, y_path / 'val.csv')
    del X_val_raw
    del X_val_seg
    del y_val

    # Test
    print('=' * 100)
    X_test_seg, keep_idx_val_test = filter_and_segment_data(X_test_raw, sampling_rate=sampling_rate, max_peaks=5)
    y_test = y_test[keep_idx_val_test]
    print('=' * 100)
    save_to_csv(X_test_seg, X_path / 'test.csv')
    save_to_csv(y_test, y_path / 'test.csv')

main()