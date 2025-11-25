import pandas as pd
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
from sklearn.model_selection import train_test_split
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
import tensorflow.keras.backend as K

# --- KONFIGURATION ---
LABEL_CSV_PATH = 'ekg_labels_mi.csv'
BATCH_SIZE = 16  # Kleinere Batches für stabileres Training
EPOCHS = 60
ACTUAL_HEIGHT = 448
ACTUAL_WIDTH = 448
INPUT_SHAPE = (ACTUAL_HEIGHT, ACTUAL_WIDTH, 1)

# WICHTIG: Verwende ALLE Daten pro Epoch statt Limit
# Dies ist entscheidend für echtes Lernen!
USE_FULL_DATA = True  # Nicht limitieren!

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

def weighted_binary_crossentropy(pos_weight=2.0):
    """
    Gewichtete Binary Crossentropy - besser als Focal Loss für diesen Fall.
    pos_weight > 1 bestraft False Negatives stärker.
    """
    def loss(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        
        # Gewichtete Crossentropy
        loss_pos = -y_true * K.log(y_pred) * pos_weight
        loss_neg = -(1 - y_true) * K.log(1 - y_pred)
        
        return K.mean(loss_pos + loss_neg)
    return loss

def f1_score(y_true, y_pred):
    """F1-Score Metrik."""
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)
    
    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())
    f1 = 2 * precision * recall / (precision + recall + K.epsilon())
    return f1

# GPU-Konfiguration
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.set_visible_devices(gpus[0], 'GPU')
        print(f"✓ GPU aktiviert: {len(gpus)} verfügbar")
    except RuntimeError as e:
        print(f"GPU-Konfigurationsfehler: {e}")

def create_improved_cnn(input_shape=INPUT_SHAPE):
    """
    Verbessertes CNN mit:
    - Mehr Kapazität für komplexe Muster
    - Residual-ähnliche Verbindungen durch tiefere Architektur
    - Stärkere Regularisierung
    - Bessere Feature-Extraktion
    """
    model = Sequential([
        Input(shape=input_shape, name='input_ekg'),
        
        # Block 1 - Initiale Feature-Extraktion
        Conv2D(32, (3, 3), activation='relu', padding='same', 
               kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same',
               kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.2),
        
        # Block 2 - Mittlere Features
        Conv2D(64, (3, 3), activation='relu', padding='same',
               kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same',
               kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),
        
        # Block 3 - Höhere Features
        Conv2D(128, (3, 3), activation='relu', padding='same',
               kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same',
               kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),
        
        # Block 4 - Tiefe Features
        Conv2D(256, (3, 3), activation='relu', padding='same',
               kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.4),
        
        # Global Average Pooling
        GlobalAveragePooling2D(),
        
        # Dense Layers mit starker Regularisierung
        Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.5),
        
        Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.5),
        
        # Output Layer
        Dense(1, activation='sigmoid', name='output_layer')
    ], name='ekg_mi_classifier_v2')
    
    return model

if __name__ == '__main__':
    print("=" * 70)
    print("IMPROVED ECG Myocardial Infarction Classifier Training")
    print("=" * 70)
    
    # 1. Labels laden
    print(f"\n📂 Lade Labels aus: {LABEL_CSV_PATH}")
    labels_df = pd.read_csv(LABEL_CSV_PATH)
    labels_df.columns = labels_df.columns.str.strip()
    print(f"   Gesamtanzahl Samples: {len(labels_df)}")
    
    # Automatische Spalten-Erkennung
    filepath_col = next((col for col in labels_df.columns 
                        if 'path' in col.lower() or 'file' in col.lower()), None)
    label_col = next((col for col in labels_df.columns 
                     if 'label' in col.lower()), None)
    
    if not filepath_col or not label_col:
        raise ValueError(f"❌ Spalten nicht gefunden! Verfügbar: {list(labels_df.columns)}")
    
    print(f"   ✓ Filepath: '{filepath_col}'")
    print(f"   ✓ Label: '{label_col}'")
    
    # Klassenverteilung
    class_dist = labels_df[label_col].value_counts().sort_index()
    print("\n📊 Klassenverteilung:")
    for cls in sorted(class_dist.index):
        count = class_dist[cls]
        pct = count / len(labels_df) * 100
        print(f"   Klasse {cls}: {count:,} ({pct:.1f}%)")
    
    # 2. Klassen-Gewichte berechnen (moderater!)
    # PROBLEM bei dir: 1:4 war viel zu extrem!
    total = len(labels_df)
    n_class_0 = class_dist.get(0, 0)
    n_class_1 = class_dist.get(1, 0)
    
    # Berechne balancierte Gewichte aber limitiere das Verhältnis
    weight_0 = total / (2 * n_class_0) if n_class_0 > 0 else 1.0
    weight_1 = total / (2 * n_class_1) if n_class_1 > 0 else 1.0
    
    # Normalisiere und limitiere auf max 3:1 Verhältnis
    max_ratio = 3.0
    if weight_1 / weight_0 > max_ratio:
        weight_1 = weight_0 * max_ratio
    
    class_weight_dict = {
        0: float(weight_0),
        1: float(weight_1)
    }
    
    print(f"\n⚖️  Klassen-Gewichte (automatisch balanciert):")
    for cls, weight in class_weight_dict.items():
        print(f"   Klasse {cls}: {weight:.4f}")
    print(f"   Verhältnis: 1:{weight_1/weight_0:.2f}")
    
    # 3. Labels zu Strings
    labels_df[label_col] = labels_df[label_col].astype(str)
    
    # 4. Datenaufteilung
    print("\n🔀 Teile Daten auf...")
    test_df = labels_df[labels_df['strat_fold'] == 10].copy()
    trainval_df = labels_df[labels_df['strat_fold'] != 10].copy()
    
    train_df, val_df = train_test_split(
        trainval_df,
        test_size=0.2,
        stratify=trainval_df[label_col],
        random_state=SEED
    )
    
    print(f"   Training:   {len(train_df):,} Samples")
    print(f"   Validation: {len(val_df):,} Samples")
    print(f"   Test:       {len(test_df):,} Samples")
    
    # 5. Data Generators mit Augmentation für Training
    print("\n🖼️  Erstelle Data Generators...")
    
    # Training mit Augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=5,  # Leichte Rotation
        width_shift_range=0.05,  # Horizontale Verschiebung
        height_shift_range=0.05,  # Vertikale Verschiebung
        zoom_range=0.05,  # Leichter Zoom
        fill_mode='constant',
        cval=0
    )
    
    # Validation/Test ohne Augmentation
    val_test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col=filepath_col,
        y_col=label_col,
        target_size=(ACTUAL_HEIGHT, ACTUAL_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        color_mode='grayscale',
        shuffle=True,
        seed=SEED,
        classes=['0', '1']
    )
    
    val_generator = val_test_datagen.flow_from_dataframe(
        dataframe=val_df,
        x_col=filepath_col,
        y_col=label_col,
        target_size=(ACTUAL_HEIGHT, ACTUAL_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        color_mode='grayscale',
        shuffle=False,
        classes=['0', '1']
    )
    
    test_generator = val_test_datagen.flow_from_dataframe(
        dataframe=test_df,
        x_col=filepath_col,
        y_col=label_col,
        target_size=(ACTUAL_HEIGHT, ACTUAL_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        color_mode='grayscale',
        shuffle=False,
        classes=['0', '1']
    )
    
    print(f"   ✓ Train: {len(train_generator)} Batches")
    print(f"   ✓ Val:   {len(val_generator)} Batches")
    print(f"   ✓ Test:  {len(test_generator)} Batches")
    
    # 6. Modell erstellen
    print("\n🏗️  Erstelle verbessertes CNN-Modell...")
    model = create_improved_cnn()
    
    # Optimizer mit adaptiver Lernrate
    optimizer = Adam(learning_rate=0.0001, clipnorm=1.0)
    
    # WICHTIG: Verwende weighted BCE statt Focal Loss
    model.compile(
        optimizer=optimizer,
        loss=weighted_binary_crossentropy(pos_weight=2.0),
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc'),
            f1_score
        ]
    )
    
    print("\n📋 Modell-Architektur:")
    model.summary()
    
    total_params = model.count_params()
    print(f"\n   Total Parameters: {total_params:,}")
    
    # 7. Callbacks
    callbacks = [
        # Early Stopping auf Basis von F1-Score (Balance!)
        EarlyStopping(
            monitor='val_f1_score',
            patience=12,
            mode='max',
            restore_best_weights=True,
            verbose=1
        ),
        
        # Learning Rate Reduction
        ReduceLROnPlateau(
            monitor='val_f1_score',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            mode='max',
            verbose=1
        ),
        
        # Model Checkpoint - speichere bestes Modell
        ModelCheckpoint(
            'best_ekg_mi_model_v2.keras',
            monitor='val_f1_score',
            mode='max',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # 8. Training
    print("\n🚀 Starte Training...")
    print("-" * 70)
    print(f"   Batch Size: {BATCH_SIZE}")
    
    if USE_FULL_DATA:
        steps_per_epoch = len(train_generator)
        validation_steps = len(val_generator)
        print(f"   ✅ Verwende ALLE Daten pro Epoch")
    else:
        steps_per_epoch = min(1000, len(train_generator))
        validation_steps = min(250, len(val_generator))
        print(f"   ⚠️  Limitiert auf {steps_per_epoch} Steps")
    
    print(f"   Steps per Epoch: {steps_per_epoch}")
    print(f"   Validation Steps: {validation_steps}")
    print(f"   Total Epochs: {EPOCHS}")
    print("-" * 70)
    
    import time
    start_time = time.time()
    
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        validation_data=val_generator,
        validation_steps=validation_steps,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )
    
    training_duration = time.time() - start_time
    hours = int(training_duration // 3600)
    minutes = int((training_duration % 3600) // 60)
    print(f"\n⏱️  Trainingsdauer: {hours}h {minutes}m")
    
    # 9. Speichern
    print("\n💾 Speichere Modelle...")
    model.save('ekg_mi_classifier_final_v2.keras')
    
    history_df = pd.DataFrame(history.history)
    history_df.to_csv('training_history_v2.csv', index=False)
    
    # 10. Beste Metriken
    best_epoch = history_df['val_f1_score'].idxmax()
    print(f"\n🏆 Beste Metriken (Epoch {best_epoch + 1}):")
    best_metrics = history_df.iloc[best_epoch]
    print(f"   Accuracy:  {best_metrics['val_accuracy']:.4f}")
    print(f"   Precision: {best_metrics['val_precision']:.4f}")
    print(f"   Recall:    {best_metrics['val_recall']:.4f}")
    print(f"   F1-Score:  {best_metrics['val_f1_score']:.4f}")
    print(f"   AUC:       {best_metrics['val_auc']:.4f}")
    
    # 11. Test-Evaluation
    print("\n🧪 Evaluiere auf Test-Set...")
    test_results = model.evaluate(test_generator, verbose=0)
    
    print("\n📊 FINALE TEST-SET METRIKEN:")
    print("=" * 70)
    print(f"   Accuracy:  {test_results[1]:.4f} ({test_results[1]*100:.1f}%)")
    print(f"   Precision: {test_results[2]:.4f} ({test_results[2]*100:.1f}%)")
    print(f"   Recall:    {test_results[3]:.4f} ({test_results[3]*100:.1f}%)")
    print(f"   F1-Score:  {test_results[5]:.4f}")
    print(f"   AUC:       {test_results[4]:.4f}")
    print("=" * 70)
    
    # 12. Medizinische Interpretation
    print("\n💡 MEDIZINISCHE BEWERTUNG:")
    
    recall = test_results[3]
    precision = test_results[2]
    f1 = test_results[5]
    
    if recall < 0.7:
        print("   ❌ KRITISCH: Recall zu niedrig - zu viele MI übersehen!")
        print("      → Gefahr: False Negatives (übersehene Infarkte)")
    elif recall < 0.85:
        print("   ⚠️  Recall akzeptabel, aber verbesserungswürdig")
    else:
        print("   ✅ Recall gut - meiste MI-Fälle werden erkannt")
    
    if precision < 0.5:
        print("   ⚠️  Precision niedrig - viele Fehlalarme")
    elif precision < 0.7:
        print("   ⚡ Precision akzeptabel")
    else:
        print("   ✅ Precision gut - wenige Fehlalarme")
    
    if f1 < 0.6:
        print("   ❌ F1-Score zu niedrig - Modell nicht einsatzbereit")
    elif f1 < 0.75:
        print("   ⚡ F1-Score akzeptabel - weitere Optimierung empfohlen")
    else:
        print("   ✅ F1-Score gut - ausgewogenes Modell")
    
    print("\n📌 EMPFEHLUNG:")
    if f1 > 0.7 and recall > 0.75:
        print("   ✅ Modell kann für weitere klinische Validierung betrachtet werden")
    else:
        print("   ⚠️  Modell benötigt weitere Optimierung vor klinischem Einsatz")
    
    print("\n" + "=" * 70)