# Funktion process_and_save_akg_data Importieren

from Funktionen import apply_notch_filter
from Funktionen import apply_bandpass_filter
import os
import glob
import numpy as np
import wfdb
import matplotlib.pyplot as plt

# Basispfad zu den EKG-Daten
BASE_DIR = '../notebooks/data/physionet.org/files/ptb-xl/1.0.3/records500/00000'
# Pfad, in dem die fertigen Bilder gespeichert werden
OUTPUT_DIR = 'ekg_images_segmented_clinical_normalized_minmax' # Neuer Output-Pfad

# --- ML-OPTIMIERTE PARAMETER ---
NOISE_FREQ = 50.0   # Netzbrummen Frequenz (z.B. 50.0 oder 60.0 Hz)
SEGMENT_LENGTH_SEC = 2.5 # Dauer jedes Segments in Sekunden
# Feste Y-Achsen-Grenze für min-max-skalierte Daten. +/- 1.1 gibt 10% Puffer um den Datenbereich [-1, 1].
Y_LIM_MINMAX = 1.1  
DPI = 100          # Bildauflösung (Dots per Inch)
# Figurgröße für das 6x2-Layout (Breite x Höhe).
FIGSIZE = (4.48, 2.24) 
TARGET_WIDTH = int(FIGSIZE[0] * DPI)
TARGET_HEIGHT = int(FIGSIZE[1] * DPI)

# Ableitungsreihenfolge (wird für das Plotten nicht verwendet, aber zur Dokumentation beibehalten)
LEAD_ORDER = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


# Load Data
os.makedirs(OUTPUT_DIR, exist_ok=True)

head_paths = sorted(glob.glob(os.path.join(BASE_DIR, '*.head')))

if not head_paths:
    head_paths = sorted(glob.glob(os.path.join(BASE_DIR, '*.hea')))

for head_path in head_paths:
    record_base = os.path.splitext(os.path.basename(head_path))[0]
    record_path = os.path.join(BASE_DIR, record_base)

    try:
        # Lese Signale und Felder mit wfdb
        signals, fields = wfdb.rdsamp(record_path)
    except Exception as e:
        print(f"Skipping {record_base}: cannot read record ({e})")
        continue

    fs = float(fields.get('fs', fields.get('fs', 500)))
    n_samples, n_channels = signals.shape
    if n_channels < 12:
        print(f"Skipping {record_base}: only {n_channels} channel(s) found (need 12).")
        continue

    # Segmentierung des Signals
    segment_samples = min(n_samples, int(SEGMENT_LENGTH_SEC * fs))
    sig_segment = signals[:segment_samples, :12].astype(float) # Erste 12 Kanäle

    # 1. ANWENDUNG DER FILTER
    filtered = np.zeros_like(sig_segment)
    for ch in range(12):
        x = sig_segment[:, ch]
        
        # Robustes Filtern (Notch und Bandpass)
        try:
            x = apply_notch_filter(x, fs, NOISE_FREQ)
        except TypeError:
            x = apply_notch_filter(x, fs)
        try:
            x = apply_bandpass_filter(x, fs)
        except TypeError:
            x = apply_bandpass_filter(x)
        filtered[:, ch] = x

    # 2. MIN-MAX-STANDARDISIERUNG AUF DEN BEREICH [-1, 1]
    
    # Finde das globale Minimum und Maximum über ALLE 12 Kanäle des Segments
    min_val = np.min(filtered)
    max_val = np.max(filtered)
    
    range_val = max_val - min_val
    epsilon = 1e-6 
    
    # Der Plot-Limit ist jetzt fix
    Y_LIM_FINAL = Y_LIM_MINMAX

    if range_val > epsilon:
        # Min-Max-Skalierung: x_norm = 2 * (x - min) / (max - min) - 1
        normalized = 2 * (filtered - min_val) / range_val - 1
    else:
        # Falls das Signal flach ist, einfach auf Null setzen
        normalized = np.zeros_like(filtered)
    
    t = np.arange(normalized.shape[0]) / fs

    # 3. ERSTELLEN UND PLOTTEN IM 6x2-LAYOUT
    fig, axs = plt.subplots(6, 2, figsize=FIGSIZE, dpi=DPI, sharex=True)
    
    for i in range(12):
        # Berechne Zeile (r) und Spalte (c) für das 6x2-Layout
        r = i % 6  
        c = i // 6 
        
        ax = axs[r, c] 
        
        # Plotten der Min-Max-skalierten Daten
        ax.plot(t, normalized[:, i], color='black', linewidth=0.5)
        
        # ML-OPTIMIERTE EINSTELLUNGEN: Feste Skalierung, keine Achsen, kein Rahmen
        ax.set_ylim(-Y_LIM_FINAL, Y_LIM_FINAL)
        ax.set_yticks([]) # Keine Y-Ticks
        ax.set_xticks([]) # Keine X-Ticks
        ax.set_frame_on(False) # Kein Rahmen

    # 4. ANPASSUNG DES ABSTANDS
    plt.subplots_adjust(hspace=0.1, wspace=0.1, left=0.01, right=0.99, top=0.98, bottom=0.02)
    
    # 5. SPEICHERN
    out_path = os.path.join(OUTPUT_DIR, f"{record_base}.png")
    try:
        fig.savefig(out_path, dpi=DPI, bbox_inches='tight', pad_inches=0)
        print(f"Saved {out_path} with Min-Max normalization. Plot limit: {Y_LIM_FINAL:.2f}")
    except Exception as e:
        print(f"Failed saving {out_path}: {e}")
    plt.close(fig)

print(f"\nVerarbeitung abgeschlossen. Bilder gespeichert in: {OUTPUT_DIR}")