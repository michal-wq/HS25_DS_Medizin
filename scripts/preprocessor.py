from functions import target_encoder, load_raw_data, sizeof_gb_array
import numpy as np
import scipy.stats as stats
import pandas as pd

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


def main():
    path = 'data/physionet.org/files/ptb-xl/1.0.3/'
    sampling_rate = 500
    cols = ['scp_codes', 'strat_fold', 'filename_lr', 'filename_hr']
    print('Phase 1: Split data')
    X_train, X_test, y_train, y_test = split_data_ecg(path, sampling_rate, cols)
    gb = sizeof_gb_array(X_train) + sizeof_gb_array(X_test) + sizeof_gb_array(y_train) + sizeof_gb_array(y_test)
    print(f"In-Memory-Grösse von X: {gb:.3f} GB")
main()