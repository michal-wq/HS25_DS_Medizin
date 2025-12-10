import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load the model
model = keras.models.load_model(os.path.join(script_dir, 'best_ekg_mi_model.keras'))

# Load labels CSV
LABEL_CSV_PATH = os.path.join(script_dir, 'ekg_labels_mi.csv')
labels_df = pd.read_csv(LABEL_CSV_PATH)
labels_df.columns = labels_df.columns.str.strip()

# Find column names
filepath_col = None
label_col = None
for col in labels_df.columns:
    if 'path' in col.lower() or 'file' in col.lower():
        filepath_col = col
    if 'label' in col.lower():
        label_col = col

print("=" * 70)
print("Paper-Ready Grad-CAM Visualisierung für EKG MI-Klassifikation")
print("=" * 70)

# Configuration
N_SAMPLES_PER_CLASS = 1000
LAYER_NAME = 'conv2d_6'

# EKG lead labels (standard 12-lead ECG layout)
# Standard layout: 6 rows x 2 columns
ECG_LEADS = [
    ['I', 'aVR'],      # Row 1
    ['II', 'aVL'],     # Row 2
    ['III', 'aVF'],    # Row 3
    ['V1', 'V4'],      # Row 4
    ['V2', 'V5'],      # Row 5
    ['V3', 'V6']       # Row 6
]

def load_and_preprocess_image(filepath):
    """Load and preprocess an ECG image."""
    if not os.path.isabs(filepath):
        full_path = os.path.join(script_dir, filepath)
        if not os.path.exists(full_path):
            full_path = os.path.join(os.path.dirname(script_dir), filepath)
    else:
        full_path = filepath
    
    img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (224, 224))
    img_with_channel = np.expand_dims(img, axis=-1)
    img_input = np.expand_dims(img_with_channel, axis=0).astype('float32') / 255.0
    return img_input, img

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    """Compute Grad-CAM heatmap."""
    last_conv_layer = model.get_layer(last_conv_layer_name)
    last_conv_layer_model = keras.Model(model.inputs, last_conv_layer.output)
    
    classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    
    conv_layer_found = False
    for layer in model.layers:
        if layer.name == last_conv_layer_name:
            conv_layer_found = True
            continue
        if conv_layer_found:
            x = layer(x)
    
    classifier_model = keras.Model(classifier_input, x)
    
    with tf.GradientTape() as tape:
        conv_outputs = last_conv_layer_model(img_array)
        tape.watch(conv_outputs)
        predictions = classifier_model(conv_outputs)
        top_pred_index = predictions[:, 0]
    
    grads = tape.gradient(top_pred_index, conv_outputs)
    
    if grads is None:
        return None, None
    
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    
    max_val = tf.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val
    
    pred_value = float(predictions.numpy()[0, 0])
    
    return heatmap.numpy(), pred_value

def compute_average_heatmap(df, class_name):
    """Compute average heatmap for a class."""
    heatmaps = []
    predictions = []
    example_img = None
    
    print(f"\nBerechne Heatmaps für {class_name}...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"  {class_name}"):
        filepath = row[filepath_col]
        
        try:
            img_input, img = load_and_preprocess_image(filepath)
            if img_input is None:
                continue
            
            heatmap, pred = make_gradcam_heatmap(img_input, model, LAYER_NAME)
            
            if heatmap is not None:
                heatmap_resized = cv2.resize(heatmap, (224, 224))
                heatmaps.append(heatmap_resized)
                predictions.append(pred)
                
                if example_img is None:
                    example_img = img
        except Exception as e:
            continue
    
    if len(heatmaps) == 0:
        return None, None, None
    
    heatmaps_array = np.array(heatmaps)
    avg_heatmap = np.mean(heatmaps_array, axis=0)
    avg_prediction = np.mean(predictions)
    
    print(f"   ✓ {len(heatmaps)} erfolgreiche Heatmaps")
    print(f"   ✓ Durchschnittliche Vorhersage: {avg_prediction:.4f}")
    
    return avg_heatmap, example_img, predictions

# Sample data
class_0_df = labels_df[labels_df[label_col] == 0].sample(
    n=min(N_SAMPLES_PER_CLASS, len(labels_df[labels_df[label_col] == 0])), 
    random_state=42
)
class_1_df = labels_df[labels_df[label_col] == 1].sample(
    n=min(N_SAMPLES_PER_CLASS, len(labels_df[labels_df[label_col] == 1])), 
    random_state=42
)

print(f"\n📊 Daten: {len(class_0_df)} Kein-MI, {len(class_1_df)} MI Samples")

# Compute heatmaps
avg_heatmap_0, example_img_0, preds_0 = compute_average_heatmap(class_0_df, "Kein MI")
avg_heatmap_1, example_img_1, preds_1 = compute_average_heatmap(class_1_df, "MI")

# Create paper-ready figure
print("\n📊 Erstelle publikationsreife Visualisierung...")

# Helper function to add ECG lead labels
def add_ecg_labels(ax, img_shape=(224, 224)):
    """Add 12-lead ECG labels to the image."""
    # Calculate positions (6 rows x 2 columns)
    row_height = img_shape[0] / 6
    col_width = img_shape[1] / 2
    
    for row_idx, row_leads in enumerate(ECG_LEADS):
        for col_idx, lead in enumerate(row_leads):
            x = col_width * col_idx + col_width * 0.05
            y = row_height * row_idx + row_height * 0.2
            
            ax.text(x, y, lead, 
                   fontsize=8, fontweight='bold',
                   color='red', 
                   bbox=dict(boxstyle='round,pad=0.3', 
                           facecolor='white', 
                           edgecolor='red',
                           alpha=0.8))

# ===== Bild C: Kein MI - Überlagerung mit Ableitungsbeschriftung =====
if avg_heatmap_0 is not None and example_img_0 is not None:
    fig_c = plt.figure(figsize=(8, 6))
    ax_c = fig_c.add_subplot(111)
    
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * avg_heatmap_0), cv2.COLORMAP_JET)
    ekg_rgb = cv2.cvtColor(example_img_0, cv2.COLOR_GRAY2RGB)
    superimposed = cv2.addWeighted(ekg_rgb, 0.6, heatmap_colored, 0.4, 0)
    ax_c.imshow(cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB))
    add_ecg_labels(ax_c)
    ax_c.axis('off')
    
    plt.savefig('bild_c_kein_mi_overlay.png', dpi=300, bbox_inches='tight')
    plt.close(fig_c)
    print("💾 Bild C gespeichert:")
    print("   - bild_c_kein_mi_overlay.png (300 DPI)")

# ===== Bild F: MI - Überlagerung mit Ableitungsbeschriftung =====
if avg_heatmap_1 is not None and example_img_1 is not None:
    fig_f = plt.figure(figsize=(8, 6))
    ax_f = fig_f.add_subplot(111)
    
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * avg_heatmap_1), cv2.COLORMAP_JET)
    ekg_rgb = cv2.cvtColor(example_img_1, cv2.COLOR_GRAY2RGB)
    superimposed = cv2.addWeighted(ekg_rgb, 0.6, heatmap_colored, 0.4, 0)
    ax_f.imshow(cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB))
    add_ecg_labels(ax_f)
    ax_f.axis('off')
    
    plt.savefig('bild_f_mi_overlay.png', dpi=300, bbox_inches='tight')
    plt.close(fig_f)
    print("💾 Bild F gespeichert:")
    print("   - bild_f_mi_overlay.png (300 DPI)")
print("\n" + "=" * 70)
print("✅ Einzelbilder C und F erfolgreich gespeichert!")
print("=" * 70)