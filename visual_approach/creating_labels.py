import pandas as pd
import os
import ast 
import sys
import re # Neu: Für das Parsen der Dateinamen

# --- KONFIGURATION ---
# Basispfad zur PTB-XL-Datenbank-Datei
BASE_DIR_CSV = '../tabular data'
# Pfad zu dem Ordner, in dem Ihre Min-Max-skalierten EKG-Bilder gespeichert sind
OUTPUT_DIR_IMAGES = 'ekg_images_beat_segmented_500hz' 
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
    verknüpft sie mit ALLEN generierten Herzschlag-Bilddateien.
    """
    # 1. PTB-XL-Datenbank laden und Labels generieren (Record-Basis)
    try:
        # Passt den Pfad an Ihre Dateistruktur an
        Y = pd.read_csv(os.path.join(csv_path, 'ptbxl_database.csv'), index_col='ecg_id')
    except FileNotFoundError:
        print(f"❌ Fehler: 'ptbxl_database.csv' nicht gefunden. Bitte Pfad prüfen.")
        sys.exit(1)

    # Funktion zur Bestimmung des MI-Labels pro Record
    def is_mi(scp_codes_str):
        try:
            codes = ast.literal_eval(scp_codes_str).keys()
            return any(code in codes for code in MI_CODES)
        except:
            return False

    Y['MI_Label'] = Y['scp_codes'].apply(is_mi).astype(int)

    # Erstellen einer Spalte mit dem Basis-Dateinamen (z.B. '00001_hr')
    # Records500 verwendet tatsächlich '_hr' Suffix (nicht _lr wie erwartet)
    Y['base_filename'] = Y.index.astype(str).str.zfill(5) + '_hr'

    # Reduziere Y auf die für uns relevanten Labels (inkl. strat_fold)
    record_labels = Y[['base_filename', 'MI_Label', 'strat_fold']].copy()


    # 2. Liste aller generierten Herzschlag-Bilder (Beat-Basis)
    # Durchsuche rekursiv alle Unterordner nach PNG-Dateien
    image_files = []
    try:
        for root, dirs, files in os.walk(image_dir):
            for file in files:
                if file.endswith('.png'):
                    # Speichere den relativen Pfad ab image_dir
                    rel_path = os.path.relpath(os.path.join(root, file), image_dir)
                    image_files.append(rel_path)
    except FileNotFoundError:
        print(f"❌ Fehler: Bildverzeichnis '{image_dir}' nicht gefunden. Bitte Pfad prüfen.")
        sys.exit(1)

    # 3. Erstellen eines DataFrames für die Beat-Bilder
    print(f"🔍 Gefundene Herzschlag-Bilder: {len(image_files)}")

    beat_df = pd.DataFrame(image_files, columns=['beat_filename'])
    beat_df['file_path'] = beat_df['beat_filename'].apply(
        lambda x: os.path.join(image_dir, x)
    )

    # 4. Dateinamen parsen, um die ursprüngliche Record-ID zu erhalten
    # Der RegEx extrahiert den Teil vor dem '_beat_00X' (z.B. '00001_hr' oder '00001_lr')
    # Funktioniert auch mit Pfaden wie '00000/00001_lr_beat_003.png'
    regex_pattern = r'^(?:.*/)?(.+?)_beat_\d{3}\.png$'

    beat_df['base_filename'] = beat_df['beat_filename'].apply(
        lambda x: re.match(regex_pattern, x).group(1) if re.match(regex_pattern, x) else None
    )

    # Lösche Beats, bei denen der Basename nicht extrahiert werden konnte
    beat_df.dropna(subset=['base_filename'], inplace=True)


    # 5. Joine die Beat-Daten mit den Record-Labels
    final_labels_df = pd.merge(
        beat_df[['file_path', 'base_filename']],
        record_labels,
        on='base_filename',
        how='inner' # Nur Records beibehalten, für die wir Bilder UND Labels haben
    )

    # Endgültiges DataFrame mit den benötigten Spalten
    labels_df = final_labels_df[['file_path', 'MI_Label', 'strat_fold']]

    return labels_df

# --- HAUPTAUSFÜHRUNG ---
if __name__ == '__main__':
    print("--- 1. Starte Label-Generierung (Pro-Beat-Basis) ---")

    labels_df = get_mi_labels(BASE_DIR_CSV, OUTPUT_DIR_IMAGES)

    # 2. Speichere das DataFrame als CSV
    labels_df.to_csv(LABEL_CSV_PATH, index=False)

    # 3. Ausgabe der Label-Verteilung
    print("\n✅ Label-DataFrame erfolgreich generiert und gespeichert unter:", LABEL_CSV_PATH)
    print(f"Gesamt-Samples (Herzschläge) mit Label: {len(labels_df)}")
    print("Verteilung (0: Kein MI / 1: MI):")
    print(labels_df['MI_Label'].value_counts())