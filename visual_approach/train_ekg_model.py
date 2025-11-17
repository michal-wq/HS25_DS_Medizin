import pandas as pd
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, BatchNormalization,
    Flatten, Dense, Dropout, GlobalAveragePooling2D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
from tensorflow.keras.regularizers import l2

# --- KONFIGURATION ---
LABEL_CSV_PATH = 'ekg_labels_mi.csv'
BATCH_SIZE = 16
EPOCHS = 40
ACTUAL_HEIGHT = 448
ACTUAL_WIDTH = 448
INPUT_SHAPE = (ACTUAL_HEIGHT, ACTUAL_WIDTH, 1)

# Seed für Reproduzierbarkeit
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# GPU-Speicherwachstum aktivieren, um Initialisierungsfehler zu vermeiden
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Derzeit muss das Speicherwachstum für alle GPUs gleich sein
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Das Speicherwachstum muss festgelegt werden, bevor die GPUs initialisiert wurden
        print(e)

def create_cnn_model(input_shape=INPUT_SHAPE, regularization=0.01):
    """Erstellt ein verbessertes CNN-Modell für die EKG-Klassifikation."""
    model = Sequential([
        Input(shape=INPUT_SHAPE, name='input_ekg'),
        
        # Block 1 - Erfasst grundlegende Features
        Conv2D(32, (3, 3), activation='relu', 
               kernel_regularizer=l2(regularization), padding='same'),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', 
               kernel_regularizer=l2(regularization), padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Block 2 - Mittlere Features
        Conv2D(64, (3, 3), activation='relu', 
               kernel_regularizer=l2(regularization), padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', 
               kernel_regularizer=l2(regularization), padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Block 3 - Höhere Features
        Conv2D(128, (3, 3), activation='relu', 
               kernel_regularizer=l2(regularization), padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),
        
        # Globales Average Pooling (besser als Flatten)
        GlobalAveragePooling2D(),
        
        # Dense Layer
        Dense(128, activation='relu', kernel_regularizer=l2(regularization)),
        BatchNormalization(),
        Dropout(0.5),
        
        # Ausgabe
        Dense(1, activation='sigmoid', name='output_layer')
    ], name='ekg_mi_classifier_improved')
    
    return model

# --- HAUPTAUSFÜHRUNG ---
if __name__ == '__main__':
    print("=" * 60)
    print("ECG Myocardial Infarction Classifier Training")
    print("=" * 60)
    
    # 1. Labels laden
    print(f"\n📂 Lade Labels aus: {LABEL_CSV_PATH}")
    labels_df = pd.read_csv(LABEL_CSV_PATH)
    print(f"   Gesamtanzahl Samples: {len(labels_df)}")
    
    # Verfügbare Spalten anzeigen
    print(f"\n🔍 Verfügbare Spalten im CSV: {list(labels_df.columns)}")
    
    # Automatische Erkennung der Spalten
    filepath_col = None
    for col in labels_df.columns:
        if 'path' in col.lower() or 'file' in col.lower():
            filepath_col = col
            break
    
    label_col = None
    for col in labels_df.columns:
        if 'label' in col.lower():
            label_col = col
            break
    
    if filepath_col is None or label_col is None:
        raise ValueError(f"❌ Konnte Spalten nicht finden! Verfügbar: {list(labels_df.columns)}")
    
    print(f"   ✓ Verwende Filepath-Spalte: '{filepath_col}'")
    print(f"   ✓ Verwende Label-Spalte: '{label_col}'")
    
    # Überprüfe ob Dateien existieren
    sample_path = labels_df[filepath_col].iloc[0]
    if not os.path.exists(sample_path):
        print(f"\n⚠️  WARNUNG: Beispieldatei nicht gefunden: {sample_path}")
        print("   Bitte überprüfen Sie, ob die Pfade korrekt sind!")
    else:
        print(f"   ✓ Dateipfade scheinen korrekt zu sein")
    
    # Klassenverteilung anzeigen
    class_distribution = labels_df[label_col].value_counts().sort_index()
    print("\n📊 Klassenverteilung:")
    if 0 in class_distribution.index:
        print(f"   Klasse 0 (Kein MI): {class_distribution[0]} ({class_distribution[0]/len(labels_df)*100:.1f}%)")
    if 1 in class_distribution.index:
        print(f"   Klasse 1 (MI):      {class_distribution[1]} ({class_distribution[1]/len(labels_df)*100:.1f}%)")
    
    # 2. Klassen-Gewichte berechnen
    class_labels = labels_df[label_col].unique()
    computed_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.sort(class_labels),
        y=labels_df[label_col]
    )
    class_weight_dict = dict(enumerate(computed_weights))
    print(f"\n⚖️  Berechnete Klassen-Gewichte:")
    for cls, weight in class_weight_dict.items():
        print(f"   Klasse {cls}: {weight:.4f}")
    
    # 3. Labels zu Strings konvertieren (für Keras flow_from_dataframe)
    print("\n🔧 Konvertiere Labels zu Strings...")
    labels_df[label_col] = labels_df[label_col].astype(str)
    
    # 4. Datenaufteilung (Stratifiziert)
    print("\n🔀 Teile Daten auf (80% Train, 20% Val)...")
    train_df, val_df = train_test_split(
        labels_df,
        test_size=0.2,
        stratify=labels_df[label_col],
        random_state=SEED
    )
    
    print(f"   Training:   {len(train_df)} Samples")
    print(f"   Validation: {len(val_df)} Samples")
    
    # Klassenverteilung in Train/Val
    train_dist = train_df[label_col].value_counts().sort_index()
    val_dist = val_df[label_col].value_counts().sort_index()
    print(f"\n   Train - Klasse 0: {train_dist[0]}, Klasse 1: {train_dist[1]}")
    print(f"   Val   - Klasse 0: {val_dist[0]}, Klasse 1: {val_dist[1]}")
    
    # 5. Data Generators
    print("\n🖼️  Erstelle Data Generators...")
    datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col=filepath_col,
        y_col=label_col,
        target_size=(ACTUAL_HEIGHT, ACTUAL_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        color_mode='grayscale',
        shuffle=True,
        seed=SEED,
        classes=['0', '1']  # Explizite Klassenreihenfolge
    )
    
    val_generator = datagen.flow_from_dataframe(
        dataframe=val_df,
        x_col=filepath_col,
        y_col=label_col,
        target_size=(ACTUAL_HEIGHT, ACTUAL_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        color_mode='grayscale',
        shuffle=False,
        classes=['0', '1']  # Explizite Klassenreihenfolge
    )
    
    print(f"   ✓ Train Generator: {len(train_generator)} Batches")
    print(f"   ✓ Val Generator:   {len(val_generator)} Batches")
    
    # 6. Modell erstellen
    print("\n🏗️  Erstelle CNN-Modell...")
    model = create_cnn_model()
    
    # Optimizer mit angepasster Lernrate
    optimizer = Adam(learning_rate=0.0001)
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )
    
    print("\n📋 Modell-Architektur:")
    model.summary()
    
    # 7. Callbacks für besseres Training
    callbacks = [
        # Early Stopping
        EarlyStopping(
            monitor='val_auc',
            patience=10,
            mode='max',
            restore_best_weights=True,
            verbose=1
        ),
        
        # Learning Rate Reduction
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        
        # Model Checkpoint
        ModelCheckpoint(
            'best_ekg_mi_model.keras',
            monitor='val_auc',
            mode='max',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # 8. Training
    print("\n🚀 Starte Training...")
    print("-" * 60)
    
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=EPOCHS,
        validation_data=val_generator,
        validation_steps=len(val_generator),
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )
    
    # 9. Ergebnisse speichern
    print("\n💾 Speichere finales Modell...")
    model.save('ekg_mi_classifier_final.keras')
    
    # Training History speichern
    history_df = pd.DataFrame(history.history)
    history_df.to_csv('training_history.csv', index=False)
    
    print("\n✅ Training abgeschlossen!")
    print(f"   Bestes Modell: best_ekg_mi_model.keras")
    print(f"   Finales Modell: ekg_mi_classifier_final.keras")
    print(f"   Training History: training_history.csv")
    
    # 10. Finale Metriken
    print("\n📈 Finale Validierungs-Metriken:")
    final_metrics = history_df.iloc[-1]
    print(f"   Loss:      {final_metrics['val_loss']:.4f}")
    print(f"   Accuracy:  {final_metrics['val_accuracy']:.4f}")
    print(f"   Precision: {final_metrics['val_precision']:.4f}")
    print(f"   Recall:    {final_metrics['val_recall']:.4f}")
    print(f"   AUC:       {final_metrics['val_auc']:.4f}")
    
    # 11. Beste Metriken (aus Early Stopping)
    best_epoch = history_df['val_auc'].idxmax()
    print(f"\n🏆 Beste Metriken (Epoch {best_epoch + 1}):")
    best_metrics = history_df.iloc[best_epoch]
    print(f"   Loss:      {best_metrics['val_loss']:.4f}")
    print(f"   Accuracy:  {best_metrics['val_accuracy']:.4f}")
    print(f"   Precision: {best_metrics['val_precision']:.4f}")
    print(f"   Recall:    {best_metrics['val_recall']:.4f}")
    print(f"   AUC:       {best_metrics['val_auc']:.4f}")
    print("=" * 60)