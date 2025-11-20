from functions import build_ecg_model, load_X_split, load_y_split, test_model
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
import time
from pathlib import Path
import numpy as np

def main():
    l = 100
    x_path = 'ready_data/X/'
    y_path = 'ready_data/y/'  # aufpassen: klein "y", wie beim Speichern!

    print('Datenstrukur wird angepasst')
    print('='*l)

    # === X laden & reshapen ===
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
    # Model bauen
    print('='*l)
    print('Model wird gebaut')
    num_classes = len(np.unique(y_train))
    input_shape = X_train.shape[1:]
    model = build_ecg_model(input_shape=input_shape, num_classes=num_classes)
    print('='*l)

    # Model training
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=32,
    )
    print('=' * l)

    # Auswertung
    print('=' * l)
    print('=' * l)
    print('=' * l)
    metrics = test_model(model, X_test, y_test, batch_size=32)
    print('=' * l)
    print('=' * l)
    print('=' * l)

    # Save Model
    test_acc = metrics['accuracy']
    model_dir = Path('models')
    model_dir.mkdir(parents=True, exist_ok=True)
    test_acc = metrics['accuracy']
    model_name = f'CNN_{test_acc*100:.2f}_.keras'
    model.save(model_dir / model_name)

main()