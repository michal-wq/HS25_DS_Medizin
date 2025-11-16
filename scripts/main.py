from functions import target_encoder, load_raw_data, sizeof_gb_array, band_pass_filter, ecg_segmentation, enforce_min_distance
from biosppy.signals.ecg import ecg
import numpy as np
import scipy.stats as stats
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

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

def filter_and_segment_data(A, sampling_rate = 500, max_peaks = 5):
    people = []
    index_error_counter = 0
    for i in range(A.shape[0]):
        kanals = []
        for k in range(A.shape[2]):
            # es wird ite Matrix (Person) genommen und aus dem Matrix wird ganze kte ECG Signal gefiltert
            signal = A[i, ::, k]
            signal_ecg_data = ecg(signal=signal, sampling_rate=sampling_rate, show=False)
            kanals.append(signal_ecg_data[4][0:max_peaks])
        people.append(kanals)
    return np.array(people)

def train_model(X_train, y_train, model, batch_size=64, n_epochs=10):
    N = X_train.shape[0]
    for epoch in range(n_epochs):
        print(f"Epoch {epoch + 1}/{n_epochs}")
        # Patienten durchmischen
        indices = np.random.permutation(N)

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch_idx = indices[start:end]

            # Raw data holen
            X_batch_raw = X_train[batch_idx]  # (B, 5000, 12)
            y_batch = y_train[batch_idx]  # (B,)

            # Preprocessing
            X_batch_seg = filter_and_segment_data(X_batch_raw)
            B, leads, n_beats, length = X_batch_seg.shape
            X_batch_model = X_batch_seg.reshape(B, leads, n_beats * length)

def build_ecg_model(input_shape=(5, 300, 12), num_classes=2):
    """
    input_shape: (n_beats, beat_length, n_leads)
    num_classes: Anzahl der Zielklassen
    """
    inputs = layers.Input(shape=input_shape)

    # Block 1
    x = layers.Conv2D(32, kernel_size=(3, 7), padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(1, 2))(x)  # nur in Zeitrichtung poolen

    # Block 2
    x = layers.Conv2D(64, kernel_size=(3, 7), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(1, 2))(x)

    # Block 3
    x = layers.Conv2D(128, kernel_size=(3, 7), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(1, 2))(x)

    # Globales Pooling statt riesigem Dense-Input
    x = layers.GlobalAveragePooling2D()(x)

    # Fully Connected Kopf
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs=inputs, outputs=outputs)

    # Wenn deine Labels Integer sind (0..C-1):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def main():
    path = 'data/physionet.org/files/ptb-xl/1.0.3/'
    sampling_rate = 500
    cols = ['scp_codes', 'strat_fold', 'filename_lr', 'filename_hr']
    print(f'Phase 1: Split data, sampling rate: {sampling_rate}')

    # Data split
    X_train_raw, X_test_raw, y_train, y_test = split_data_ecg(path, sampling_rate, cols)
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_train_raw,
        y_train,
        test_size=0.2,
        random_state=42,
        stratify=y_train
    )

    print(f'Train raw shape: { X_train_raw.shape}')
    print(f'Val raw shape: { X_val_raw.shape}')

    # Filtern und Segmentierung
    X_train_seg = filter_and_segment_data(X_train_raw, sampling_rate=sampling_rate, max_peaks=5)
    X_val_seg = filter_and_segment_data(X_val_raw, sampling_rate=sampling_rate, max_peaks=5)

    print(f'Train seg shape: {X_train_seg.shape}')
    print(f'Val seg shape: {X_val_seg.shape}')

    # Datenstruktur anpassen
    X_train_tf = np.transpose(X_train_seg, (0, 2, 3, 1))
    X_val_tf = np.transpose(X_val_seg, (0, 2, 3, 1))

    # Model bauen
    num_classes = len(np.unique(y_train))
    model = build_ecg_model(num_classes = num_classes)

    # Model training
    history = model.fit(
        X_train_tf, y_train,
        validation_data=(X_val_tf, y_val),
        epochs=20,
        batch_size=32,
    )
    # Test vorbereiten
    X_test_seg = filter_and_segment_data(X_test_raw, sampling_rate=sampling_rate, max_peaks=5)
    X_test_tf = np.transpose(X_test_seg, (0, 2, 3, 1))

    # Auswertung
    test_loss, test_acc = model.evaluate(X_test_tf, y_test, batch_size=32)
    print("Test loss:", test_loss)
    print("Test acc: ", test_acc)

main()