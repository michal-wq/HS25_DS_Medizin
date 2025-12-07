from functions import make_gradcam_heatmap, load_X_split
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


# ----------------------------
# Lead Layout
# ----------------------------

LEAD_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

ECG_LAYOUT = [
    ["I", "aVR"],
    ["II", "aVL"],
    ["III", "aVF"],
    ["V1", "V4"],
    ["V2", "V5"],
    ["V3", "V6"],
]


# ----------------------------
# Hilfsfunktionen
# ----------------------------

def find_last_conv2d_layer(model):
    conv_layers = [l.name for l in model.layers if isinstance(l, tf.keras.layers.Conv2D)]
    if not conv_layers:
        raise ValueError("Keine Conv2D-Layer im Modell gefunden.")
    return conv_layers[-1]


def resize_heatmap_to_target(heatmap_2d, target_h, target_w):
    hm = heatmap_2d.astype(np.float32)
    hm = hm[None, ..., None]  # (1, H, W, 1)
    hm_resized = tf.image.resize(hm, size=(target_h, target_w), method="bilinear").numpy()
    hm_resized = hm_resized[0, ..., 0]

    mx = hm_resized.max()
    if mx > 0:
        hm_resized = hm_resized / (mx + 1e-8)
    return hm_resized


def compute_time_importance_from_beat_time_heatmap(hm_beat_time):
    time_imp = hm_beat_time.mean(axis=0)  # (300,)
    mx = time_imp.max()
    if mx > 0:
        time_imp = time_imp / (mx + 1e-8)
    return time_imp


def plot_12lead_overlay(fig, sub_gs, x_sample, time_importance, beat_idx=0):
    lead_to_idx = {name: i for i, name in enumerate(LEAD_ORDER)}

    T = x_sample.shape[1]
    t = np.arange(T)

    for r in range(6):
        for c in range(2):
            lead_name = ECG_LAYOUT[r][c]
            ax = fig.add_subplot(sub_gs[r, c])

            if lead_name not in lead_to_idx:
                ax.set_axis_off()
                continue

            li = lead_to_idx[lead_name]
            signal = x_sample[beat_idx, :, li]

            # Heatmap-Hintergrund (Zeit-Wichtigkeit)
            band = np.tile(time_importance[None, :], (2, 1))

            ymin = float(np.min(signal))
            ymax = float(np.max(signal))
            pad = 0.05 * (ymax - ymin + 1e-8)
            ymin -= pad
            ymax += pad

            ax.imshow(
                band,
                extent=[t[0], t[-1], ymin, ymax],
                aspect="auto",
                alpha=0.35,
                cmap="jet",
                origin="lower",
            )

            # EKG-Linie
            ax.plot(t, signal, linewidth=1.2, color="black")

            ax.set_title(lead_name, fontsize=9, fontweight="bold", loc="left")
            ax.set_xticks([])
            ax.set_yticks([])

            for spine in ax.spines.values():
                spine.set_alpha(0.15)


# ----------------------------
# MAIN
# ----------------------------

def main():
    x_path = 'ready_data/X/'

    # === X laden ===
    X_test = load_X_split(x_path, "test")
    print("X_test Shape:", X_test.shape)  # (N, 5, 300, 12)

    # === Modell laden ===
    model = tf.keras.models.load_model("models/CNN_84.71_.keras")

    # === Conv Layer finden ===
    last_conv_layer_name = find_last_conv2d_layer(model)
    print("Letzter Conv2D Layer:", last_conv_layer_name)

    # === Beispiel auswählen ===
    i = 0
    input_image = X_test[i:i + 1].astype(np.float32)   # (1, 5, 300, 12)
    x_sample = X_test[i]                               # (5, 300, 12)

    # === Grad-CAM berechnen ===
    heatmap = make_gradcam_heatmap(input_image, model, last_conv_layer_name)
    heatmap = np.squeeze(heatmap)

    target_h, target_w = x_sample.shape[0], x_sample.shape[1]  # (5, 300)

    if heatmap.shape != (target_h, target_w):
        heatmap_bt = resize_heatmap_to_target(heatmap, target_h, target_w)
    else:
        heatmap_bt = heatmap

    # === Zeit-Wichtigkeit (1D) ===
    time_importance = compute_time_importance_from_beat_time_heatmap(heatmap_bt)

    # ----------------------------
    # NUR PANEL C: 12-Lead Overlay
    # ----------------------------
    fig = plt.figure(figsize=(8, 10))
    gs = fig.add_gridspec(6, 2, hspace=0.3, wspace=0.2)

    plot_12lead_overlay(
        fig,
        gs,
        x_sample,
        time_importance,
        beat_idx=0
    )

    fig.suptitle(
        "Grad-CAM Zeit-Wichtigkeit – 12 Ableitungen (Time-Series CNN)",
        fontsize=14,
        fontweight="bold"
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("timeseries_gradcam_overlay_only.png", dpi=300, bbox_inches="tight")
    plt.savefig("timeseries_gradcam_overlay_only.pdf", dpi=300, bbox_inches="tight")

    print("Gespeichert als:")
    print(" - timeseries_gradcam_overlay_only.png")
    print(" - timeseries_gradcam_overlay_only.pdf")

    plt.show()


if __name__ == "__main__":
    main()
