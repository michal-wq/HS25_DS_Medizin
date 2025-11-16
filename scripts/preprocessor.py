from functions import target_encoder, load_raw_data, sizeof_gb_array, band_pass_filter, ecg_segmentation, enforce_min_distance
from biosppy.signals.ecg import ecg
import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def split_data_ecg(path, sampling_rate, cols=['scp_codes', 'strat_fold']):
    # Zielvariable einlesen
    y = pd.read_csv(path + 'ptbxl_database.csv', index_col='ecg_id')[cols]
    y['target'] = y['scp_codes'].apply(target_encoder)
    print('Die Zielvariable wurde erfolgreich eingelesen und encoded')

    # ECG Daten einlesen
    print('Die ECG Daten werden eingelesen')
    X, kept_index, missing_log, error_log = load_raw_data(y, sampling_rate, path)
    print(f'Fehlende Dateien: {len(missing_log)}, andere Fehler: {len(error_log)}')

    # y überschreiben
    y = y.reindex(kept_index)  # exakt gleiche Reihenfolge wie X
    assert len(y) == len(X), "X und y haben unterschiedliche Länge nach dem Reindexing."

    print('Die ECG Daten wurden erfolgreich eingelesen')

    # Split data into train and test
    print('Datensatz wird gesplittet')
    test_fold = 10
    X_train = X[np.where(y.strat_fold != test_fold)]
    X_test = X[np.where(y.strat_fold == test_fold)]
    y_train = y[y.strat_fold != test_fold]['target']
    y_test = y[y.strat_fold == test_fold]['target']

    print('Datensatz wurde erfolgreich gesplittet')
    return X_train, X_test, y_train.values, y_test.values

def filter_and_segment_data(A, batch_size, start = 0, sampling_rate = 500, max_peaks = 5):
    people = []
    index_error_counter = 0
    for i in range(start, start + batch_size):
        kanals = []
        for k in range(A.shape[2]):
            # es wird ite Matrix (Person) genommen und aus dem Matrix wird ganze kte ECG Signal gefiltert
            signal = A[i, ::, k]
            signal_ecg_data = ecg(signal=signal, sampling_rate=sampling_rate, show=False)
            kanals.append(signal_ecg_data[4][0:max_peaks])
        people.append(kanals)
    return np.array(people)



def main():
    path = 'data/physionet.org/files/ptb-xl/1.0.3/'
    sampling_rate = 500
    cols = ['scp_codes', 'strat_fold', 'filename_lr', 'filename_hr']
    print(f'Phase 1: Split data, sampling rate: {sampling_rate}')

    # Data split
    X_train, X_test, y_train, y_test = split_data_ecg(path, sampling_rate, cols)

    # Filter
    print(X_train.shape)
    X_train = filter_and_segment_data(X_train, 64)
    print(X_train.shape)



    """
    
    Hier geht es nachher weiter mit der Preprocessing pipeline
    
    """
main()