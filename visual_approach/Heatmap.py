import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
import os
import matplotlib.pyplot as plt

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load the model
model = keras.models.load_model(os.path.join(script_dir, 'best_ekg_mi_model.keras'))

print("=" * 60)
print("Grad-CAM Heatmap Generator für EKG Herzinfarkt-Klassifikation")
print("=" * 60)

# Print model architecture
print("\n🔍 Modell-Layer:")
for i, layer in enumerate(model.layers):
    if isinstance(layer, keras.layers.Conv2D):
        print(f"   {i}: {layer.name} - Conv2D")

# Find all Conv2D layers
conv_layers = []
for layer in model.layers:
    if isinstance(layer, keras.layers.Conv2D):
        conv_layers.append(layer.name)

print(f"\n   Verfügbare Conv2D Layer: {conv_layers}")

# Load EKG data
image_path = os.path.join(script_dir, 'ekg_images_224x224/00000/00001_hr_beat_000.png')
ekg_data = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if ekg_data is None:
    raise ValueError(f"❌ Bild konnte nicht geladen werden: {image_path}")

print(f"\n📂 Lade Bild: {os.path.basename(image_path)}")
print(f"   Original Shape: {ekg_data.shape}")

# Resize and prepare input
ekg_data = cv2.resize(ekg_data, (224, 224))
ekg_data_with_channel = np.expand_dims(ekg_data, axis=-1)
ekg_input = np.expand_dims(ekg_data_with_channel, axis=0).astype('float32') / 255.0

print(f"   Input Shape: {ekg_input.shape}")

# Alternative Grad-CAM implementation that works with BatchNormalization
def make_gradcam_heatmap_alternative(img_array, model, last_conv_layer_name):
    """
    Alternative Grad-CAM Implementation die mit BatchNorm funktioniert.
    Verwendet einen Submodel-Ansatz.
    """
    # Get the last conv layer
    last_conv_layer = model.get_layer(last_conv_layer_name)
    last_conv_layer_model = keras.Model(model.inputs, last_conv_layer.output)
    
    # Create a model that maps conv output to final prediction
    classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    
    # Find where the conv layer is and build the rest
    conv_layer_found = False
    for layer in model.layers:
        if layer.name == last_conv_layer_name:
            conv_layer_found = True
            continue
        if conv_layer_found:
            x = layer(x)
    
    classifier_model = keras.Model(classifier_input, x)
    
    # Get conv outputs
    with tf.GradientTape() as tape:
        conv_outputs = last_conv_layer_model(img_array)
        tape.watch(conv_outputs)
        predictions = classifier_model(conv_outputs)
        top_pred_index = predictions[:, 0]
    
    # Get gradients
    grads = tape.gradient(top_pred_index, conv_outputs)
    
    if grads is None:
        return None, None, None, None
    
    # Pool gradients and compute heatmap
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    
    max_val = tf.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val
    
    # Get predictions
    pred_value = float(predictions.numpy()[0, 0])
    predicted_class = int(pred_value > 0.5)
    confidence = pred_value if predicted_class == 1 else (1 - pred_value)
    
    return heatmap.numpy(), pred_value, predicted_class, confidence

# Test layers
print(f"\n🔍 Teste Conv2D Layer für Grad-CAM...")
last_conv_layer_name = None
heatmap = None

for layer_name in reversed(conv_layers):
    print(f"   Versuche Layer: '{layer_name}'", end=" ")
    try:
        result = make_gradcam_heatmap_alternative(ekg_input, model, layer_name)
        if result[0] is not None:
            last_conv_layer_name = layer_name
            heatmap, pred_value, predicted_class, confidence = result
            print(f"✓ Funktioniert!")
            break
        else:
            print(f"✗ (Gradienten sind None)")
    except Exception as e:
        print(f"✗ (Fehler: {str(e)[:50]})")
        continue

if last_conv_layer_name is None or heatmap is None:
    print("\n" + "=" * 60)
    print("⚠️  WARNUNG: Grad-CAM konnte nicht berechnet werden!")
    print("=" * 60)
    print("\nDas Problem liegt an den BatchNormalization-Layern im Modell.")
    print("Diese blockieren die Gradienten-Berechnung in Keras.")
    print("\nAlternativen:")
    print("1. Modell ohne BatchNorm nach den Conv-Layern neu trainieren")
    print("2. Layer Relevance Propagation (LRP) verwenden")
    print("3. Integrated Gradients verwenden")
    print("4. Activation-based visualization (nur Aktivierungen, keine Gradienten)")
    print("\nMöchten Sie eine dieser Alternativen sehen?")
    exit(1)

print(f"\n✓ Verwende Layer: '{last_conv_layer_name}'")

print(f"\n📊 Vorhersage:")
print(f"   Raw Output: {pred_value:.4f}")
print(f"   Klasse: {predicted_class} ({'MI' if predicted_class == 1 else 'Kein MI'})")
print(f"   Konfidenz: {confidence:.2%}")

# Resize heatmap
heatmap_resized = cv2.resize(heatmap, (224, 224))

# Create colored heatmap
heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)

# Convert grayscale to RGB
ekg_rgb = cv2.cvtColor(ekg_data, cv2.COLOR_GRAY2RGB)

# Superimpose
superimposed_img = cv2.addWeighted(ekg_rgb, 0.6, heatmap_colored, 0.4, 0)

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].imshow(ekg_data, cmap='gray')
axes[0].set_title('Original EKG Bild', fontsize=14, fontweight='bold')
axes[0].axis('off')

im = axes[1].imshow(heatmap_resized, cmap='jet')
axes[1].set_title(
    f'Grad-CAM Heatmap (Layer: {last_conv_layer_name})\n'
    f'Vorhersage: {"MI" if predicted_class == 1 else "Kein MI"} '
    f'({confidence:.1%} Konfidenz)',
    fontsize=14,
    fontweight='bold'
)
axes[1].axis('off')
plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

axes[2].imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
axes[2].set_title('Überlagerung\n(Wichtige Regionen hervorgehoben)', fontsize=14, fontweight='bold')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('ekg_gradcam_heatmap.png', dpi=200, bbox_inches='tight')
print(f"\n💾 Heatmap gespeichert als: 'ekg_gradcam_heatmap.png'")
plt.show()

print("\n" + "=" * 60)
print("✅ Grad-CAM Analyse abgeschlossen!")
print("=" * 60)