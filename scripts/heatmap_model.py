from functions import make_gradcam_heatmap, load_X_split
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np  # wichtig für np.newaxis

def main():
    l = 100
    x_path = 'ready_data/X/'
    y_path = 'ready_data/y/'  # aufpassen: klein "y", wie beim Speichern!

    print('Datenstrukur wird angepasst')
    print('='*l)

    # === X laden ===
    X_test = load_X_split(x_path, "test")

    # === Model laden ===
    model = tf.keras.models.load_model("models/CNN_84.71_.keras")

    # Beispiel auswählen
    i = 0
    input_image = X_test[i:i + 1]  # Shape (1, H, W, C)

    # Grad-CAM Heatmap berechnen
    heatmap = make_gradcam_heatmap(
        img_array=input_image,
        model=model,
        last_conv_layer_name="conv2d_2",  # ggf. anpassen an deinen Layernamen
        class_index=None,                 # None -> nimmt argmax prediction
    )

    # Heatmap auf Eingabegröße bringen (Beats x Zeit)
    inp = input_image[0]      # z.B. Shape: (5, 300, 12)
    H_in, W_in = inp.shape[:2]

    heatmap_resized = tf.image.resize(
        heatmap[np.newaxis, ..., np.newaxis],  # (1, Hc, Wc, 1)
        (H_in, W_in)
    )[0, ..., 0].numpy()  # -> (H_in, W_in), z.B. (5, 300)

    # Heatmap anzeigen
    plt.figure(figsize=(10, 4))
    plt.imshow(heatmap_resized, cmap="jet", aspect="auto")
    plt.colorbar(label="Relevanz")
    plt.xlabel("Zeit (Samples)")
    plt.ylabel("Beats")
    plt.title("Grad-CAM Heatmap")
    plt.tight_layout()
    plt.show()

main()
