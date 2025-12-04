import pandas as pd
import os
import ast
from glob import glob

# Konfiguration
DB_PATH = '../tabular data/ptbxl_database.csv'
IMAGE_DIR = 'ekg_images_224x224'
OUTPUT_CSV = 'ekg_labels_mi.csv'

MI_CODES = ['MI', 'AMI', 'IMI', 'ASMI', 'ILMI', 'ALMI', 'PMI', 'LMI', 
            'APMI', 'IPMI', 'ISMI', 'ADMI', 'LDMI', 'VAD']

def create_labels():
    # 1. Lade Datenbank und erstelle MI-Labels
    df = pd.read_csv(DB_PATH, index_col='ecg_id')
    df['MI_Label'] = df['scp_codes'].apply(
        lambda x: int(any(code in ast.literal_eval(x).keys() for code in MI_CODES))
    )
    df['base_filename'] = df.index.astype(str).str.zfill(5) + '_hr'
    
    # 2. Finde alle Bilder
    images = glob(f'{IMAGE_DIR}/**/*.png', recursive=True)
    print(f"Gefundene Bilder: {len(images)}")
    
    # 3. Erstelle Beat-DataFrame
    beats = pd.DataFrame({
        'file_path': images,
        'base_filename': [os.path.basename(f).split('_beat_')[0] for f in images]
    })
    
    # 4. Merge mit Labels
    result = beats.merge(
        df[['base_filename', 'MI_Label', 'strat_fold']], 
        on='base_filename', 
        how='inner'
    )
    
    # 5. Speichere und gebe Info aus
    result[['file_path', 'MI_Label', 'strat_fold']].to_csv(OUTPUT_CSV, index=False)
    
    print(f"\n✅ Gespeichert: {OUTPUT_CSV}")
    print(f"Samples: {len(result)}")
    print(f"MI-Verteilung:\n{result['MI_Label'].value_counts()}")

if __name__ == '__main__':
    create_labels()