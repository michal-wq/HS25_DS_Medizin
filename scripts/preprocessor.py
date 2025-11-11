from functions import target_encoder, load_raw_data
import numpy as np
import scipy.stats as stats
import pandas as pd

def split_data_ecg(path, sampling_rate, cols=['scp_codes', 'strat_fold']):
    # Zielvariable einlesen
    y = pd.read_csv(path + 'ptbxl_database.csv', index_col='ecg_id')[cols]
    y['target'] = y['scp_codes'].apply(target_encoder)

    # ECG Daten einlesen
    X = load_raw_data(y, sampling_rate, path)

    # Split data into train and test
    test_fold = 10
    X_train = X[np.where(y.strat_fold != test_fold)]
    X_test = X[np.where(y.strat_fold == test_fold)]
    y_train = y[y.strat_fold != test_fold]['target']
    y_test = y[y.strat_fold == test_fold]['target']

    return X_train, X_test, y_train.values, y_test.values


def main():


    return 0