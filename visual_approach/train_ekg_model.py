import pandas as pd
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
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
import tensorflow.keras.backend as K

# --- KONFIGURATION ---
LABEL_CSV_PATH = 'ekg_labels_mi.csv'
BATCH_SIZE = 32  # Reduced from 64 to avoid GPU OOM errors
EPOCHS = 40
ACTUAL_HEIGHT = 448
ACTUAL_WIDTH = 448
INPUT_SHAPE = (ACTUAL_HEIGHT, ACTUAL_WIDTH, 1)
STEPS_PER_EPOCH = 1000  # Limit steps per epoch for faster iterations
VALIDATION_STEPS = 250  # Limit validation steps

# Seed für Reproduzierbarkeit
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

def focal_loss(gamma=2.0, alpha=0.25):
    """Focal Loss für Klassenungleichgewicht - fokussiert auf schwierige Beispiele."""
    def focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        
        # Focal Loss Formel
        ce = -y_true * K.log(y_pred)
        focal_weight = alpha * K.pow(1 - y_pred, gamma)
        focal_loss = focal_weight * ce
        
        # Für negative Klasse
        ce_neg = -(1 - y_true) * K.log(1 - y_pred)
        focal_weight_neg = (1 - alpha) * K.pow(y_pred, gamma)
        focal_loss_neg = focal_weight_neg * ce_neg
        
        return K.mean(focal_loss + focal_loss_neg)
    return focal_loss_fixed

def f1_score(y_true, y_pred):
    """F1-Score Metrik für Training."""
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)
    
    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())
    f1 = 2 * precision * recall / (precision + recall + K.epsilon())
    return f1

# GPU-Konfiguration optimieren
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Speicherwachstum aktivieren für effiziente GPU-Nutzung
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # Verwende nur die erste GPU (kann auf beide erweitert werden)
        tf.config.set_visible_devices(gpus[0], 'GPU')
        print(f"✓ GPU aktiviert: {len(gpus)} verfügbar, verwende GPU:0")
    except RuntimeError as e:
        print(f"GPU-Konfigurationsfehler: {e}")
else:
    print("⚠️  Keine GPUs gefunden - Training läuft auf CPU (sehr langsam!)")

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
    
    # Spaltennamen bereinigen (Leerzeichen entfernen)
    labels_df.columns = labels_df.columns.str.strip()
    
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
    
    # 2. Klassen-Gewichte berechnen (erhöht für MI-Klasse)
    # Verwende manuelles Gewicht 1:4 statt balanced um False Negatives stark zu bestrafen
    class_weight_dict = {
        0: 1.0,   # Kein MI - normales Gewicht
        1: 4.0    # MI - 4x höheres Gewicht (False Negatives sind gefährlich!)
    }
    print(f"\n⚖️  Klassen-Gewichte (manuell optimiert):")
    for cls, weight in class_weight_dict.items():
        print(f"   Klasse {cls}: {weight:.4f}")
    print("   → MI-Klasse erhält 4x Gewicht um Recall zu verbessern")
    
    # 3. Labels zu Strings konvertieren (für Keras flow_from_dataframe)
    print("\n🔧 Konvertiere Labels zu Strings...")
    labels_df[label_col] = labels_df[label_col].astype(str)
    
    # 4. Datenaufteilung basierend auf strat_fold
    print("\n🔀 Teile Daten auf (strat_fold=10 für Test, Rest für Train/Val)...")
    
    # Test-Set: strat_fold=10
    test_df = labels_df[labels_df['strat_fold'] == 10].copy()
    
    # Train/Val-Set: strat_fold != 10
    trainval_df = labels_df[labels_df['strat_fold'] != 10].copy()
    
    # Train/Val aufteilen (80/20)
    train_df, val_df = train_test_split(
        trainval_df,
        test_size=0.2,
        stratify=trainval_df[label_col],
        random_state=SEED
    )
    
    print(f"   Training:   {len(train_df)} Samples")
    print(f"   Validation: {len(val_df)} Samples")
    print(f"   Test:       {len(test_df)} Samples")
    
    # Klassenverteilung in Train/Val/Test
    train_dist = train_df[label_col].value_counts().sort_index()
    val_dist = val_df[label_col].value_counts().sort_index()
    test_dist = test_df[label_col].value_counts().sort_index()
    print(f"\n   Train - Klasse 0: {train_dist.get('0', 0)}, Klasse 1: {train_dist.get('1', 0)}")
    print(f"   Val   - Klasse 0: {val_dist.get('0', 0)}, Klasse 1: {val_dist.get('1', 0)}")
    print(f"   Test  - Klasse 0: {test_dist.get('0', 0)}, Klasse 1: {test_dist.get('1', 0)}")
    
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
    
    test_generator = datagen.flow_from_dataframe(
        dataframe=test_df,
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
    print(f"   ✓ Test Generator:  {len(test_generator)} Batches")
    
    # 6. Modell erstellen
    print("\n🏗️  Erstelle CNN-Modell...")
    model = create_cnn_model()
    
    # Optimizer mit angepasster Lernrate
    optimizer = Adam(learning_rate=0.0001)
    
    model.compile(
        optimizer=optimizer,
        loss=focal_loss(gamma=2.0, alpha=0.25),  # Focal Loss statt Binary Crossentropy
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
    
    # 7. Callbacks für besseres Training
    callbacks = [
        # Early Stopping - fokussiert auf Recall (wichtiger als AUC!)
        EarlyStopping(
            monitor='val_recall',
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
        
        # Model Checkpoint - speichere bestes Modell basierend auf Recall
        ModelCheckpoint(
            'best_ekg_mi_model.keras',
            monitor='val_recall',
            mode='max',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # 8. Training
    print("\n🚀 Starte Training...")
    print("-" * 60)
    print(f"   Batch Size: {BATCH_SIZE}")
    print(f"   Steps per Epoch: {STEPS_PER_EPOCH}")
    print(f"   Validation Steps: {VALIDATION_STEPS}")
    print(f"   Total Epochs: {EPOCHS}")
    print(f"   Samples per Epoch: ~{STEPS_PER_EPOCH * BATCH_SIZE:,}")
    print("-" * 60)
    
    import time
    start_time = time.time()
    
    history = model.fit(
        train_generator,
        steps_per_epoch=STEPS_PER_EPOCH,  # Limit steps for faster epochs
        epochs=EPOCHS,
        validation_data=val_generator,
        validation_steps=VALIDATION_STEPS,  # Limit validation steps
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )
    
    end_time = time.time()
    training_duration = end_time - start_time
    hours = int(training_duration // 3600)
    minutes = int((training_duration % 3600) // 60)
    seconds = int(training_duration % 60)
    
    print(f"\n⏱️  Trainingsdauer: {hours}h {minutes}m {seconds}s ({training_duration:.2f} Sekunden)")
    
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
    print(f"   F1-Score:  {final_metrics['val_f1_score']:.4f}")
    
    # 11. Beste Metriken (aus Early Stopping - basierend auf Recall)
    best_epoch = history_df['val_recall'].idxmax()
    print(f"\n🏆 Beste Metriken (Epoch {best_epoch + 1}, höchster Recall):")
    best_metrics = history_df.iloc[best_epoch]
    print(f"   Loss:      {best_metrics['val_loss']:.4f}")
    print(f"   Accuracy:  {best_metrics['val_accuracy']:.4f}")
    print(f"   Precision: {best_metrics['val_precision']:.4f}")
    print(f"   Recall:    {best_metrics['val_recall']:.4f}")
    print(f"   AUC:       {best_metrics['val_auc']:.4f}")
    print(f"   F1-Score:  {best_metrics['f1_score']:.4f}")
    
    # 12. Evaluation auf Test-Set
    print("\n🧪 Evaluiere auf Test-Set (strat_fold=10)...")
    test_results = model.evaluate(test_generator, verbose=1)
    
    print("\n📊 Test-Set Metriken:")
    print(f"   Loss:      {test_results[0]:.4f}")
    print(f"   Accuracy:  {test_results[1]:.4f}")
    print(f"   Precision: {test_results[2]:.4f}")
    print(f"   Recall:    {test_results[3]:.4f}")
    print(f"   AUC:       {test_results[4]:.4f}")
    print(f"   F1-Score:  {test_results[5]:.4f}")
    
    # Interpretation der Metriken
    print("\n💡 Interpretation:")
    if test_results[3] < 0.6:  # Recall < 60%
        print("   ⚠️  Recall ist noch niedrig - viele MI-Fälle werden übersehen")
    elif test_results[3] < 0.75:
        print("   ⚡ Recall ist akzeptabel - weiteres Tuning möglich")
    else:
        print("   ✅ Recall ist gut - Modell erkennt meiste MI-Fälle")
    
    if test_results[2] > 0.7:  # Precision > 70%
        print("   ✅ Precision ist gut - wenige Fehlalarme")
    else:
        print("   ⚠️  Precision könnte besser sein - viele Fehlalarme")
    
    print("=" * 60)