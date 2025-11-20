import wfdb
import numpy as np
from pathlib import Path
import wfdb
import time
import sys
import pandas as pd
from biosppy.signals.ecg import ecg
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
)

def target_encoder(x):
    """
    Falls eine der Arten von Herzinfarkt in SCP - Code von einer Person aufgelistet wurde, wird das als 1 klassifiziert, sonst 0
    """
    mi = ["IMI", "ASMI", "ILMI", "AMI", "ALMI", "INJAS", "LMI", "INJAL", "IPLMI", "IPMI", "INJIN", "INJLA", "PMI", "INJIL"]
    for element in mi:
        if element in x:
            return 1
    return 0

def load_raw_data(df, sampling_rate, base_path):
    """
    Lädt Roh-ECG-Signale. Überspringt fehlende Dateien.
    Zusätzlich werden auch logs für fehlende Dateien gespeichert.

    Returns:
        X            : np.ndarray mit Signalen
        df_kept      : DataFrame nur für erfolgreich geladene Reihen
        missing_log  : Liste[(ecg_id, path_str)] mit fehlenden Dateien
        error_log    : Liste[(ecg_id, path_str, repr(err))] mit anderen Fehlern
    """
    # Entweder 100 Hz oder 500 Hz
    col = 'filename_lr' if sampling_rate == 100 else 'filename_hr'


    base_path = Path(base_path)
    X = []
    kept_idx = []
    missing_log = []
    error_log = []
    i = 0
    for ecg_id, rel in df[col].items():
        path = base_path / rel  # robustes Path-Join
        #if i > 100:
        #    break
        try:
            # wfdb erwartet path ohne Dateiendung;
            signal, meta = wfdb.rdsamp(str(path))
            X.append(signal)
            kept_idx.append(ecg_id)

        # Hier wird es abgefangen falls irgendeine ECG Datei fehlt
        except FileNotFoundError:
            missing_log.append((ecg_id, str(path)))

        # Hier wird alles andere abgefangen
        except Exception as e:
            error_log.append((ecg_id, str(path), repr(e)))
        i += 1

    # Nur sicherheit
    if len(X) == 0:
        raise RuntimeError("Keine Signale geladen. Prüfe 'base_path' und die Filename-Spalten.")

    # In Array konvertieren
    X = np.asarray(X)

    # df auf erfolgreich geladene Reihen beschränken
    df_kept = df.loc[kept_idx].copy()

    return X, df_kept, missing_log, error_log

def sizeof_gb_array(arr):
    return arr.nbytes / (1024**3)

def _format_seconds(seconds: float) -> str:
    """Hilfsfunktion: Sekunden schön als H:MM:SS formatieren."""
    if seconds == float("inf"):
        return "∞"
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h:d}:{m:02d}:{s:02d}"
    else:
        return f"{m:02d}:{s:02d}"

def print_eta(start_time: float, current_step: int, total_steps: int, prefix: str = "") -> None:
    """
    Gibt im Terminal eine Fortschrittsanzeige mit geschätzter Restzeit aus.

    Parameters
    ----------
    start_time : float
        Zeitstempel zu Beginn (z.B. time.time()).
    current_step : int
        Aktueller Schritt (1-basiert, also bei der ersten Iteration = 1).
    total_steps : int
        Gesamtzahl der Schritte.
    prefix : str
        Optionaler Text vor der Anzeige (z.B. 'Train', 'Filter', etc.).
    """
    elapsed = time.time() - start_time
    current_step = max(current_step, 1)  # Division durch 0 vermeiden

    # Durchsatz (Steps pro Sekunde)
    rate = current_step / elapsed if elapsed > 0 else 0.0

    # Geschätzte Restzeit
    if rate > 0:
        remaining = (total_steps - current_step) / rate
    else:
        remaining = float("inf")

    progress = current_step / total_steps
    percent = progress * 100

    msg = (
        f"{prefix} {percent:6.2f}% | "
        f"Step {current_step}/{total_steps} | "
        f"Elapsed: {_format_seconds(elapsed)} | "
        f"ETA: {_format_seconds(remaining)}"
    )

    # \r = Zeilenanfang, end='' damit nicht ständig neue Zeilen entstehen
    sys.stdout.write("\r" + msg)
    sys.stdout.flush()

    # Am Ende der Schleife einmal Zeilenumbruch
    if current_step == total_steps:
        sys.stdout.write("\n")

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

def split_data_ecg(path, sampling_rate, cols=['scp_codes', 'strat_fold']):
    test_fold = 10

    # Zielvariable einlesen (max. 5000)
    y = pd.read_csv(path + 'ptbxl_database.csv', index_col='ecg_id')[cols]#[0:100]
    y['target'] = y['scp_codes'].apply(target_encoder)

    print('Die ECG Daten werden eingelesen')
    X, y_kept, missing_log, error_log = load_raw_data(y, sampling_rate, path)
    print(f'Fehlende Dateien: {len(missing_log)}, andere Fehler: {len(error_log)}')

    # Index von y_kept zurücksetzen, damit er zu X passt (0..len(X)-1)
    y_kept = y_kept.reset_index(drop=True)

    # Maske für Train/Test
    mask_train = y_kept['strat_fold'] != test_fold
    mask_test = ~mask_train

    # y-Arrays
    y_train = y_kept.loc[mask_train, 'target'].to_numpy()
    y_test = y_kept.loc[mask_test, 'target'].to_numpy()

    # X nach derselben Maske aufteilen
    X_train = X[mask_train.values]
    X_test = X[mask_test.values]

    print('Datensatz wurde erfolgreich gesplittet')
    return X_train, X_test, y_train, y_test

def filter_and_segment_data(A, sampling_rate=500, max_peaks=5):
    """
    A: Rohdaten, Shape (N, T, n_leads)
    Rückgabe:
        X_seg:  np.ndarray, Shape (N_valid, n_leads, max_peaks, beat_len)
        keep_idx: np.ndarray der Indizes (in A), die behalten wurden
    """
    n_samples, _, n_leads = A.shape
    people = []
    keep_idx = []

    beat_len = None  # wird anhand des ersten nicht-leeren/validen Signals festgelegt

    # Beat-Länge bestimmen
    start_time = time.time()
    for i in range(n_samples):
        for k in range(n_leads):
            signal = A[i, :, k]
            try:
                ecg_data = ecg(signal=signal, sampling_rate=sampling_rate, show=False)
                templates = ecg_data[4]  # (n_beats, beat_length)
            except ValueError:
                # dieses Lead bei diesem Patienten unbrauchbar -> einfach ignorieren
                continue

            if templates is not None and templates.shape[0] > 0:
                beat_len = templates.shape[1]
                break
        if beat_len is not None:
            break

        # Fortschritt
        print_eta(start_time, i + 1, n_samples, prefix="[Beat-Länge]")

    if beat_len is None:
        raise RuntimeError("Konnte keine ECG-Templates finden – checke die Rohdaten.")

    # Personen/Kanäle mit fester Shape aufbauen
    start_time = time.time()
    for i in range(n_samples):
        leads_list = []
        invalid_patient = False  # wenn True -> Patient wird komplett verworfen

        for k in range(n_leads):
            signal = A[i, :, k]
            try:
                ecg_data = ecg(signal=signal, sampling_rate=sampling_rate, show=False)
                templates = ecg_data[4]  # (n_beats, beat_length) oder evtl. leer
            except ValueError:
                # Zu wenige Beats
                invalid_patient = True
                break

            if templates is None or templates.shape[0] == 0:
                # Kein Beat erkannt -> z.B. komplett Nullen
                beats = np.zeros((max_peaks, beat_len))
            else:
                beats = templates

                # Beat-Länge anpassen
                if beats.shape[1] > beat_len:
                    beats = beats[:, :beat_len]
                elif beats.shape[1] < beat_len:
                    pad_w = np.zeros((beats.shape[0], beat_len - beats.shape[1]))
                    beats = np.concatenate([beats, pad_w], axis=1)

                # Maximal max_peaks Beats nehmen
                beats = beats[:max_peaks]

                # Falls weniger Beats als max_peaks -> mit Nullen auffüllen
                n_beats = beats.shape[0]
                if n_beats < max_peaks:
                    pad = np.zeros((max_peaks - n_beats, beat_len))
                    beats = np.concatenate([beats, pad], axis=0)

            # beats:
            leads_list.append(beats)

        if invalid_patient:
            # Diesen Patienten komplett überspringen
            continue

        # (n_leads, max_peaks, beat_len)
        person_arr = np.stack(leads_list, axis=0)
        people.append(person_arr)
        keep_idx.append(i)

        # Fortschritt
        print_eta(start_time, i + 1, n_samples, prefix="[Shape Aufbauen]")

    if len(people) == 0:
        raise RuntimeError("Alle Patienten wurden verworfen – zu viele ungültige Signale?")

    # (N_valid, n_leads, max_peaks, beat_len)
    X_seg = np.stack(people, axis=0)
    keep_idx = np.array(keep_idx, dtype=int)
    return X_seg, keep_idx

def save_to_csv(data, filepath, index=False, header=True):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)  # Ordner automatisch anlegen

    # Fall 1: DataFrame direkt speichern
    if isinstance(data, pd.DataFrame):
        data.to_csv(filepath, index=index, header=header)
        return

    # Fall 2: NumPy-Array → in DataFrame umwandeln
    if isinstance(data, np.ndarray):
        arr = data
        # Wenn mehr als 2D, alle Dimensionen außer der ersten „flach“ machen
        if arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)
        df = pd.DataFrame(arr)
        df.to_csv(filepath, index=index, header=header)
        return

    # Fall 3: alles andere versuchen wir als tabellenartige Struktur zu interpretieren
    df = pd.DataFrame(data)
    df.to_csv(filepath, index=index, header=header)

def load_X_split(x_path, split_name, n_leads=12, max_peaks=5):
    df = pd.read_csv(f"{x_path}{split_name}.csv")  # <- KEIN index_col=0 !
    X_flat = df.to_numpy()                         # (N, F)
    N, F = X_flat.shape

    features_per_sample = n_leads * max_peaks
    if F % features_per_sample != 0:
        raise ValueError(
            f"Spaltenzahl {F} passt nicht zu n_leads={n_leads}, "
            f"max_peaks={max_peaks}. Erwartet Vielfaches von {features_per_sample}."
        )

    beat_len = F // features_per_sample
    print(f"berechnete beat_len: {beat_len}")

    X_4d = X_flat.reshape(N, n_leads, max_peaks, beat_len)  # (N, 12, 5, beat_len)
    X_ready = np.transpose(X_4d, (0, 2, 3, 1))              # (N, max_peaks, beat_len, n_leads)

    return X_ready

def load_y_split(y_path, split_name):
    # WICHTIG: KEIN index_col=0, weil wir beim Speichern index=False hatten!
    df = pd.read_csv(f"{y_path}{split_name}.csv")
    # eine Spalte -> zu 1D-Array machen
    y = df.squeeze("columns").to_numpy().astype("int32")
    return y

def test_model(model, X_test, y_test, batch_size=32, average='binary'):
    """
    Bewertet ein Keras/TensorFlow-Modell auf Testdaten und gibt
    Loss, Accuracy, Precision, Recall, AUC, F1-Score aus.

    Parameters
    ----------
    model : tf.keras.Model
        Trainiertes Modell.
    X_test : np.ndarray oder tf.Tensor
        Test-Features, Shape z.B. (N, H, W, C).
    y_test : array-ähnlich
        True Labels (0/1 oder Klassen-Indices).
    batch_size : int
        Batch-Größe für model.predict().
    average : str
        'binary', 'macro', 'micro' etc. – wird für Precision/Recall/F1 bei Multiclass benutzt.
    """
    y_true = np.array(y_test).ravel()

    # Wahrscheinlichkeiten vorhersagen
    y_proba = model.predict(X_test, batch_size=batch_size)

    # Multiclass (oder binary mit softmax)
    if y_proba.ndim == 2 and y_proba.shape[1] > 1:
        # Klassenindex mit größter Wahrscheinlichkeit
        y_pred = y_proba.argmax(axis=1)

        # Loss (log-loss über alle Klassen)
        loss = log_loss(y_true, y_proba)

        # AUC (One-vs-Rest)
        try:
            auc = roc_auc_score(y_true, y_proba, multi_class='ovr')
        except ValueError:
            auc = np.nan  # z.B. wenn nur eine Klasse im Test vorhanden ist

        avg = 'macro' if average == 'binary' else average

    else:
        # Binärfall mit einer einzigen Output-Spalte (Sigmoid)
        y_proba_1 = y_proba.ravel()
        y_pred = (y_proba_1 >= 0.5).astype(int)

        # Für log_loss brauchen wir 2 Spalten (P(0), P(1))
        proba_stacked = np.vstack([1 - y_proba_1, y_proba_1]).T
        loss = log_loss(y_true, proba_stacked)

        try:
            auc = roc_auc_score(y_true, y_proba_1)
        except ValueError:
            auc = np.nan

        avg = 'binary'

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average=avg, zero_division=0)
    rec = recall_score(y_true, y_pred, average=avg, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=avg, zero_division=0)

    print(f"Loss:      {loss:.4f}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"AUC:       {auc:.4f}")
    print(f"F1-Score:  {f1:.4f}")

    return {
        "loss": loss,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "auc": auc,
        "f1": f1,
    }
