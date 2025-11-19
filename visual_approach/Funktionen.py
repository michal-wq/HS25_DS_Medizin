import numpy as np
import wfdb
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import os
from glob import glob

# Funktion zum anwenden des notch Filters
def apply_notch_filter(data, fs, notch_freq):
    Q = 30.0
    w0 = notch_freq / (fs / 2)
    b, a = signal.iirnotch(w0, Q)
    return signal.filtfilt(b, a, data)

# Funktion zum anwenden des bandpass Filters
def apply_bandpass_filter(data, fs, low_cut=0.5, high_cut=45.0):
    nyquist = 0.5 * fs
    low = low_cut / nyquist
    high = high_cut / nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    return signal.filtfilt(b, a, data)

# Funktion um die EKG sequenzen als png zu plotten und speichern
def save_ekg_as_image(signal_data, output_path, y_lim, dpi, figsize):
    """Speichert ein gefiltertes EKG-Segment achsenfrei als PNG."""
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plotten des Signals
    ax.plot(signal_data, color='black', linewidth=1.0)

    # Kritisch: Feste Y-Achsen-Grenzen für Konsistenz
    ax.set_ylim(-y_lim, y_lim) 
    
    # Achsen und Ränder entfernen
    ax.axis('off') 
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)
    
    # Speichern
    plt.savefig(output_path, dpi=dpi, format='png', bbox_inches=None, pad_inches=0)
    plt.close(fig)

# Funktion yum verarbeiten und Speiern der Bilddateien
def process_and_save_ekg_data(data_dir, output_dir, segment_sec, y_lim, dpi, figsize, noise_freq):
    """
    Findet alle EKG-Dateien, filtert sie, segmentiert sie und speichert jedes Segment.
    Die Bilder werden in Unterordnern unter output_dir abgelegt, z.B. output_dir/00000/*.png
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Alle Datensatz-Pfade finden (.hea-Dateien)
    # Suche rekursiv nach allen .hea-Dateien unter data_dir. Das deckt 'records100', 'records500' etc. ab.
    all_record_paths = glob(os.path.join(data_dir, '**', '*.hea'), recursive=True)

    print(f"Gefundene EKG-Header-Dateien: {len(all_record_paths)}")

    # Debug-Hilfe: falls keine Dateien gefunden wurden, liste kurz den Inhalt des data_dir auf
    if len(all_record_paths) == 0:
        try:
            print(f"Verzeichnis {data_dir} enthält (erste Einträge): {os.listdir(data_dir)[:20]}")
        except Exception as ex:
            print(f"Fehler beim Auflisten von {data_dir}: {ex}")
        print("Bitte überprüfe, ob BASE_DIR korrekt ist und ob .hea-Dateien unter diesem Pfad liegen.")
        return

    for record_path in all_record_paths:
        # record_path ist der volle Pfad zur .hea-Datei
        record_id_full = record_path.replace(data_dir, '').replace('.hea', '')
        record_path_clean = record_path.replace('.hea', '')  # Pfad ohne Endung für wfdb

        try:
            # Lese EKG Daten und Metadaten
            X, meta = wfdb.rdsamp(record_path_clean)
            fs = meta['fs']  # Abtastrate
            cols = meta['sig_name']  # Ableitungsnamen

            # Anzahl der Samples pro Segment
            segment_len_samples = int(fs * segment_sec)

            # Bestimme Ziel-Unterordner (z.B. '00000' aus .../records500/00000/00001_lr.hea)
            patient_folder = os.path.basename(os.path.dirname(record_path_clean))
            target_dir = os.path.join(output_dir, patient_folder)
            os.makedirs(target_dir, exist_ok=True)

            # Iteriere über jede Ableitung (Lead)
            for col_idx, lead in enumerate(cols):
                raw_signal = X[:, col_idx]  # Daten für diese Ableitung

                # 2. Filtern
                bandpassed_signal = apply_bandpass_filter(raw_signal, fs)
                filtered_signal = apply_notch_filter(bandpassed_signal, fs, noise_freq)

                # 3. Segmentierung und Speicherung
                for i in range(0, len(filtered_signal), segment_len_samples):
                    segment = filtered_signal[i:i + segment_len_samples]

                    # Überspringe das letzte Segment, wenn es zu kurz ist
                    if len(segment) < segment_len_samples:
                        continue

                    # Erzeuge eindeutigen Dateinamen ohne Pfadinformationen
                    base_record_name = os.path.basename(record_path_clean)  # z.B. '00001_lr'
                    file_name = f"{base_record_name}_{lead}_seg{i//segment_len_samples:02d}.png"
                    output_file_path = os.path.join(target_dir, file_name)

                    # 5. Speichern
                    save_ekg_as_image(segment, output_file_path, y_lim, dpi, figsize)

            print(f"✅ Patient {record_id_full} erfolgreich verarbeitet. Ausgabe in: {target_dir}")

        except Exception as e:
            print(f"❌ Fehler bei der Verarbeitung von {record_id_full}: {e}")