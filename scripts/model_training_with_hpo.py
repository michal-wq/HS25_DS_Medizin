from functions import build_ecg_model, load_X_split, load_y_split, test_model
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
import time
from pathlib import Path
import numpy as np

# ================================
# Random Search Setup (seed = 7)
# ================================

rng = np.random.RandomState(7)
tf.random.set_seed(7)


def sample_hyperparams(rng):
    """
    Sample eine zufällige Hyperparameter-Kombination.

    Entspricht:
    - number of layers            -> num_conv_layers
    - filter size                 -> kernel_time[i]
    - number of feature maps      -> filters[i]
    - stride                      -> stride_time[i]
    - pooling regions / sizes     -> pool_time[i]
    - units in fully-connected    -> dense_units
    """
    num_conv_layers = rng.randint(2, 5)  # {2, 3, 4}

    # Hyperparameter-Sampling
    cfg = {
        "num_conv_layers": num_conv_layers,
        "filters": [
            rng.choice([32, 64, 96, 128, 192, 256])
            for _ in range(num_conv_layers)
        ],
        "kernel_time": [
            rng.choice([3, 5, 7, 9])
            for _ in range(num_conv_layers)
        ],
        "stride_time": [
            rng.choice([1, 2])
            for _ in range(num_conv_layers)
        ],
        "pool_time": [
            rng.choice([2, 3, 4])
            for _ in range(num_conv_layers)
        ],
        "dense_units": rng.choice([64, 128, 256, 512]),
        "dropout_rate": rng.uniform(0.2, 0.6),
        "learning_rate": rng.choice([1e-4, 3e-4, 1e-3]),
    }

    for i in range(num_conv_layers):
        cfg["filters"].append(rng.choice([32, 64, 96, 128, 192, 256]))
        cfg["kernel_time"].append(rng.choice([3, 5, 7, 9]))
        cfg["stride_time"].append(rng.choice([1, 2]))
        cfg["pool_time"].append(rng.choice([2, 3, 4]))

    return cfg


def build_cnn_from_config(cfg, input_shape, num_classes):
    """
    Baue ein CNN entsprechend der Hyperparameter-Konfiguration cfg.
    """
    inputs = layers.Input(shape=input_shape)
    x = inputs

    # Convolutional-Layer (number of layers, feature maps, filter size, stride, pooling)
    for i in range(cfg["num_conv_layers"]):
        x = layers.Conv2D(
            filters=cfg["filters"][i],                 # number of feature maps
            kernel_size=(3, cfg["kernel_time"][i]),   # filter size (Zeitachse)
            strides=(1, cfg["stride_time"][i]),       # stride in Zeit
            padding="same",
            activation="relu",
            name=f"conv_{i}",
        )(x)
        x = layers.BatchNormalization(name=f"bn_{i}")(x)
        x = layers.MaxPooling2D(
            pool_size=(1, cfg["pool_time"][i]),       # pooling size in Zeit
            strides=(1, cfg["pool_time"][i]),
            padding="same",
            name=f"pool_{i}",
        )(x)

    # Globales Pooling
    x = layers.GlobalAveragePooling2D(name="gap")(x)

    # Fully-connected Layer
    x = layers.Dense(cfg["dense_units"], activation="relu", name="fc1")(x)
    x = layers.Dropout(cfg["dropout_rate"], name="dropout")(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = models.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def run_random_search(
    X_train,
    y_train,
    X_val,
    y_val,
    input_shape,
    num_classes,
    num_trials=20,
    batch_size=32,
    max_epochs=20,
):
    """
    Einfacher Random Search über num_trials zufällige Hyperparameter-Samples.
    """
    X_train = X_train.astype("float32")
    X_val = X_val.astype("float32")

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )

    results = []

    for t in range(num_trials):
        cfg = sample_hyperparams(rng)
        print(f"\n=== Random Search Trial {t+1}/{num_trials} ===")
        print("Config:", cfg)

        model = build_cnn_from_config(cfg, input_shape=input_shape, num_classes=num_classes)

        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=max_epochs,
            batch_size=batch_size,
            verbose=0,
            callbacks=[early_stop],
        )

        best_val_acc = float(max(history.history["val_accuracy"]))
        print(f"Best val_accuracy (trial {t+1}): {best_val_acc:.4f}")
        results.append((cfg, best_val_acc))

    # Beste Konfiguration wählen
    best_cfg, best_score = max(results, key=lambda x: x[1])
    print("\n============================")
    print("Beste Val-Accuracy:", best_score)
    print("Beste Hyperparameter:", best_cfg)
    print("============================")

    # Bestes Modell mit bester Config nochmal trainieren
    best_model = build_cnn_from_config(best_cfg, input_shape=input_shape, num_classes=num_classes)
    best_model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=max_epochs,
        batch_size=batch_size,
        verbose=1,
        callbacks=[early_stop],
    )

    return best_model, best_cfg, best_score


# ================================
# Dein main() mit Random Search
# ================================

def main():
    l = 100
    x_path = 'ready_data/X/'
    y_path = 'ready_data/y/'  # aufpassen: klein "y", wie beim Speichern!

    print('Datenstrukur wird angepasst')
    print('='*l)

    # === X laden ===
    X_train = load_X_split(x_path, "train")
    X_val   = load_X_split(x_path, "val")
    X_test  = load_X_split(x_path, "test")

    print("X_train:", X_train.shape)
    print("X_val:  ", X_val.shape)
    print("X_test: ", X_test.shape)

    # === y laden ===
    y_train = load_y_split(y_path, "train")
    y_val   = load_y_split(y_path, "val")
    y_test  = load_y_split(y_path, "test")

    print("y_train:", y_train.shape)
    print("y_val:  ", y_val.shape)
    print("y_test: ", y_test.shape)
    print('=' * l)

    # ============================
    # Hyperparameter-Tuning: Random Search
    # ============================
    num_classes = len(np.unique(y_train))
    input_shape = X_train.shape[1:]

    print('='*l)
    print('Starte Hyperparameter-Optimierung (Random Search, seed=7)')
    best_model, best_cfg, best_score = run_random_search(
        X_train,
        y_train,
        X_val,
        y_val,
        input_shape=input_shape,
        num_classes=num_classes,
        num_trials=100,   # kannst du variieren
        batch_size=32,
        max_epochs=20,
    )
    print('='*l)

    # Auswertung mit Test-Set
    print('=' * l)
    print('Evaluierung auf Test-Set')
    metrics = test_model(best_model, X_test, y_test, batch_size=32)
    print(metrics)
    print('=' * l)

    # Save Model
    test_acc = metrics['accuracy']
    model_dir = Path('models')
    model_dir.mkdir(parents=True, exist_ok=True)
    model_name = f'CNN_RS_{test_acc*100:.2f}_.keras'
    best_model.save(model_dir / model_name)
    print(f"Bestes Modell gespeichert unter: {model_dir / model_name}")


main()
