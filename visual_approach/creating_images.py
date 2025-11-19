import os
import glob
import numpy as np
import wfdb
import matplotlib.pyplot as plt
import biosppy.signals.ecg as ecg
from Funktionen import apply_notch_filter
from Funktionen import apply_bandpass_filter


# --- KONFIGURATION UND PARAMETER ---
# Basispfad zu den EKG-Daten (alle records500 Unterordner)
BASE_DIR = '../notebooks/data/physionet.org/files/ptb-xl/1.0.3/records500'
# Pfad, in dem die fertigen Bilder gespeichert werden (mit Unterordnern)
OUTPUT_DIR = 'ekg_images_beat_segmented_500hz'

# ML-OPTIMIERTE PARAMETER
NOISE_FREQ = 50.0   # Netzbrummen Frequenz (50.0 Hz EU)
SEGMENT_LENGTH_SEC = 0.8 # Dauer jedes Segments in Sekunden (Einzelner Herzschlag)
Y_LIM_MINMAX = 1.1  # Feste Y-Achsen-Grenze für normalisierte Daten [-1.1, 1.1]
DPI = 200           # Bildauflösung (Dots per Inch)
FIGSIZE = (2.24, 2.24) # Figurgröße für das 6x2-Layout (Breite x Höhe in Zoll)

# Definiere den Index des Referenz-Leads für die R-Peak-Erkennung (Lead II ist typisch)
# Annahme: Ableitungsreihenfolge ist ['I', 'II', 'III', ...] -> Lead II ist Index 1
REFERENCE_LEAD_INDEX = 1 

# --- HAUPTPROZESS ---
def process_and_save_ekg_beats():
    """
    Lädt Roh-EKG-Daten, erkennt R-Zacken, segmentiert jeden Herzschlag (Beat) 
    auf allen 12 Kanälen, filtert, normalisiert und speichert ihn als Bild.
    """
    print("=" * 60)
    print(f"Starte EKG-Segmentierung: {SEGMENT_LENGTH_SEC}s pro Beat.")
    print(f"Zielordner: {OUTPUT_DIR}")
    print("=" * 60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Finde alle Unterordner (00000, 01000, 02000, etc.)
    subdirs = sorted([d for d in glob.glob(os.path.join(BASE_DIR, '*')) 
                      if os.path.isdir(d)])
    
    if not subdirs:
        print(f"❌ Keine Unterordner in {BASE_DIR} gefunden.")
        return
    
    print(f"📁 Gefundene Unterordner: {len(subdirs)}")
    
    total_records = 0
    total_beats = 0
    
    # Loop durch alle Unterordner
    for subdir in subdirs:
        subdir_name = os.path.basename(subdir)
        print(f"\n{'='*60}")
        print(f"📂 Verarbeite Ordner: {subdir_name}")
        print(f"{'='*60}")
        
        # Erstelle entsprechenden Output-Unterordner
        output_subdir = os.path.join(OUTPUT_DIR, subdir_name)
        os.makedirs(output_subdir, exist_ok=True)
        
        # Finde alle .hea/.head Dateien in diesem Unterordner
        head_paths = sorted(glob.glob(os.path.join(subdir, '*.head')))
        if not head_paths:
            head_paths = sorted(glob.glob(os.path.join(subdir, '*.hea')))
        
        if not head_paths:
            print(f"⚠️  Keine .head oder .hea Dateien in {subdir_name} gefunden.")
            continue
        
        print(f"📄 Dateien in {subdir_name}: {len(head_paths)}")
        
        for head_path in head_paths:
            record_base = os.path.splitext(os.path.basename(head_path))[0]
            record_path = os.path.join(subdir, record_base)
            
            try:
                # Lese Signale und Felder
                signals, fields = wfdb.rdsamp(record_path)
            except Exception as e:
                print(f"⚠️  Skipping {record_base}: cannot read record ({e})")
                continue

            fs = float(fields.get('fs', fields.get('fs', 500)))
            n_samples, n_channels = signals.shape
            
            if n_channels < 12:
                print(f"⚠️  Skipping {record_base}: only {n_channels} channel(s) found (need 12).")
                continue
            
            # --- R-ZACKEN-ERKENNUNG MIT BIOSPPY ---
            try:
                # Biosppy verarbeitet den Referenz-Lead und gibt die R-Peak-Indizes zurück.
                # Intern filtert Biosppy das Signal.
                reference_signal = signals[:, REFERENCE_LEAD_INDEX].astype(float)
                out = ecg.ecg(signal=reference_signal, sampling_rate=fs, show=False)
                r_peaks = out['rpeaks']
            except Exception as e:
                print(f"⚠️  Skipping {record_base}: Biosppy R-peak detection failed ({e})")
                continue
            
            # Vorbereiten für die Segmentierung
            half_segment_samples = int((SEGMENT_LENGTH_SEC / 2) * fs)
            
            # Filtern von R-Zacken, die zu nah am Rand liegen
            r_peaks = r_peaks[r_peaks >= half_segment_samples]
            r_peaks = r_peaks[r_peaks <= (n_samples - half_segment_samples)]
            
            # print(f"✅ Bearbeite {record_base}. Gefundene R-Zacken: {len(r_peaks)}")
            total_records += 1
            total_beats += len(r_peaks)
            
            # --- LOOP ÜBER ALLE SEGMENTE/HEZRSCHLÄGE ---
            for beat_idx, r_peak_index in enumerate(r_peaks):
                
                # 1. SEGMENTIERUNG
                start_idx = r_peak_index - half_segment_samples
                end_idx = r_peak_index + half_segment_samples
                
                # Segment ausschneiden
                sig_segment = signals[start_idx:end_idx, :12].astype(float) 

                # 2. FILTERUNG DES SEGMENTS
                filtered = np.zeros_like(sig_segment)
                for ch in range(12):
                    x = sig_segment[:, ch]
                    
                    # Anwendung der Notch- und Bandpass-Filter
                    try:
                        x = apply_notch_filter(x, fs, NOISE_FREQ)
                    except TypeError:
                        x = apply_notch_filter(x, fs)
                    try:
                        x = apply_bandpass_filter(x, fs)
                    except TypeError:
                        x = apply_bandpass_filter(x)
                    filtered[:, ch] = x

                # 3. MIN-MAX-NORMALISIERUNG AUF DEN BEREICH [-1, 1] (Beat-spezifisch)
                min_val = np.min(filtered)
                max_val = np.max(filtered)
                range_val = max_val - min_val
                epsilon = 1e-6 
                
                if range_val > epsilon:
                    # Skalierung: x_norm = 2 * (x - min) / (max - min) - 1
                    normalized = 2 * (filtered - min_val) / range_val - 1
                else:
                    normalized = np.zeros_like(filtered)
                
                t = np.arange(normalized.shape[0]) / fs # Zeitachse für Plot

                # 4. ERSTELLEN UND PLOTTEN IM 6x2-LAYOUT
                fig, axs = plt.subplots(6, 2, figsize=FIGSIZE, dpi=DPI, sharex=True)
                Y_LIM_FINAL = Y_LIM_MINMAX
                
                for i in range(12):
                    r = i % 6
                    c = i // 6 
                    ax = axs[r, c] 
                    
                    # Plotten der normalisierten Daten
                    ax.plot(t, normalized[:, i], color='black', linewidth=0.5)
                    
                    # ML-OPTIMIERTE EINSTELLUNGEN: Feste Skalierung, keine Achsen, kein Rahmen
                    ax.set_ylim(-Y_LIM_FINAL, Y_LIM_FINAL)
                    ax.set_yticks([]) 
                    ax.set_xticks([]) 
                    ax.set_frame_on(False) 

                # Abstandsanpassung für minimale Ränder
                plt.subplots_adjust(hspace=0.1, wspace=0.1, left=0.01, right=0.99, top=0.98, bottom=0.02)
                
                # 5. SPEICHERN
                out_path = os.path.join(output_subdir, f"{record_base}_beat_{beat_idx:03d}.png")
                
                try:
                    # Speichern ohne Ränder (bbox_inches='tight', pad_inches=0)
                    fig.savefig(out_path, dpi=DPI, bbox_inches='tight', pad_inches=0)
                    # print(f"Saved {out_path}") # Nur bei Bedarf aktivieren
                except Exception as e:
                    print(f"Failed saving {out_path}: {e}")
                
                plt.close(fig)
        
        print(f"✅ {subdir_name} abgeschlossen: {len(head_paths)} Records verarbeitet")

    print(f"\n{'='*60}")
    print(f"✅ GESAMTVERARBEITUNG ABGESCHLOSSEN")
    print(f"📊 Statistik:")
    print(f"   - Records verarbeitet: {total_records}")
    print(f"   - Herzschläge gespeichert: {total_beats}")
    print(f"   - Output-Ordner: {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == '__main__':
    process_and_save_ekg_beats()