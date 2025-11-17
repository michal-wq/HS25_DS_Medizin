import pandas as pd
import os
import ast 
import sys
from collections import Counter

# --- KONFIGURATION ---
# Basispfad zur PTB-XL-Datenbank-Datei
BASE_DIR_CSV = '../notebooks/data/physionet.org/files/ptb-xl/1.0.3/'
# Pfad zu dem Ordner, in dem Ihre Min-Max-skalierten EKG-Bilder gespeichert sind
OUTPUT_DIR_IMAGES = 'ekg_images_segmented_clinical_normalized_minmax' 
# Name der Datei, in der das finale Label-DataFrame gespeichert wird
LABEL_CSV_PATH = 'ekg_labels_mi.csv' 

# Codes, die einen Myokardinfarkt (MI) repräsentieren
MI_CODES = [
    'MI', 'AMI', 'IMI', 'ASMI', 'ILMI', 'ALMI', 'PMI', 'LMI', 'APMI', 'IPMI',
    'ISMI', 'ADMI', 'LDMI', 'VAD'
]

# --- FUNKTION ZUR LABEL-GENERIERUNG ---

def get_mi_labels(csv_path, image_dir):
    """
    Lädt die PTB-XL-Datenbank, erstellt eine binäre MI-Klassifikation und 
    verknüpft sie mit den tatsächlich generierten Bilddateien.
    """
    try:
        Y = pd.read_csv('../tabular data/ptbxl_database.csv', index_col='ecg_id')
    except FileNotFoundError:
        print(f"❌ Fehler: 'ptbxl_database.csv' nicht gefunden unter {csv_path}. Bitte Pfad prüfen.")
        sys.exit(1)

    Y['filename'] = Y.index.astype(str).str.zfill(5) + '_hr'
    
    def is_mi(scp_codes_str):
        try:
            codes = ast.literal_eval(scp_codes_str).keys()
            return any(code in codes for code in MI_CODES)
        except:
            return False

    Y['MI_Label'] = Y['scp_codes'].apply(is_mi).astype(int)

    try:
        processed_images = set(
            os.path.splitext(f)[0] for f in os.listdir(image_dir) 
            if f.endswith('.png') or f.endswith('.jpg')
        )
    except FileNotFoundError:
        print(f"❌ Fehler: Bildverzeichnis '{image_dir}' nicht gefunden. Bitte Pfad prüfen.")
        sys.exit(1)

    final_labels = Y[Y['filename'].isin(processed_images)].copy()

    # Korrigierte Pfad-Erstellung für Pandas Series
    final_labels['file_path'] = image_dir + os.path.sep + final_labels['filename'].astype(str) + '.png'

    labels_df = final_labels[['file_path', 'MI_Label']]
    
    return labels_df

# --- HAUPTAUSFÜHRUNG ---
if __name__ == '__main__':
    print("--- 1. Starte Label-Generierung ---")
    
    labels_df = get_mi_labels(BASE_DIR_CSV, OUTPUT_DIR_IMAGES)
    
    # 2. Speichere das DataFrame als CSV
    labels_df.to_csv(LABEL_CSV_PATH, index=False)
    
    # 3. Ausgabe der Label-Verteilung
    print("\n✅ Label-DataFrame erfolgreich generiert und gespeichert unter:", LABEL_CSV_PATH)
    print(f"Gesamtbilder mit Label: {len(labels_df)}")
    print("Verteilung (0: Kein MI / 1: MI):")
    print(labels_df['MI_Label'].value_counts())