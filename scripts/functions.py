import wfdb
import numpy as np
from pathlib import Path
import wfdb

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

    for ecg_id, rel in df[col].items():
        path = base_path / rel  # robustes Path-Join
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


