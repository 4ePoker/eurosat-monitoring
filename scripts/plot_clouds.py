"""Draw the embedding clouds, so "cloud of points" stops being a metaphor.

Each image becomes 2048 numbers. Read those as coordinates and the image is a
single point in a 2048-dimensional space; a batch of images is a cloud of points.
Drift is that cloud moving.

2048 axes cannot be drawn, so this projects onto the two directions along which
the reference varies most -- the first two principal components of the profile
the detector already uses. It is a shadow of the real cloud, but it is the same
cloud the detector measures.

    python scripts/plot_clouds.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from monitoring.detect.reference import ReferenceProfile

ROOT = Path(__file__).resolve().parents[1]
BATCHES = ROOT / "data" / "batches"
PROFILE = ROOT / "artifacts" / "reference_profile"

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
SERIES_1 = "#2a78d6"   # reference
SERIES_2 = "#eb6834"   # the batch being compared

PANELS = [
    ("none_s0.50_test_n500", "sem drift", "as duas nuvens sobrepoem-se"),
    ("label_shift_coastal_s0.50_test_n500", "mudou a mistura", "desloca-se, mas nao faz dano"),
    ("new_sensor_s1.00_test_n500", "mudaram as fotos", "nuvem separada, 83 pontos de dano"),
]


def main() -> None:
    profile = ReferenceProfile.load(PROFILE)
    ref2d = profile.pca.transform(profile.embeddings)[:, :2]

    # Shared limits, derived from the data rather than guessed: one frame for all
    # three panels, or a cloud would look displaced because its axes differ.
    allpts = [ref2d] + [
        profile.pca.transform(np.load(BATCHES / b / "embeddings.npy").astype(np.float64))[:, :2]
        for b, _, _ in PANELS
    ]
    stacked = np.vstack(allpts)
    pad = 0.06 * (stacked.max(0) - stacked.min(0))
    XLIM = (stacked[:, 0].min() - pad[0], stacked[:, 0].max() + pad[0])
    YLIM = (stacked[:, 1].min() - pad[1], stacked[:, 1].max() + pad[1])

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8), facecolor=SURFACE)

    for ax, (batch, title, subtitle) in zip(axes, PANELS):
        cur = np.load(BATCHES / batch / "embeddings.npy").astype(np.float64)
        cur2d = profile.pca.transform(cur)[:, :2]

        ax.set_facecolor(SURFACE)
        ax.scatter(ref2d[:, 0], ref2d[:, 1], s=13, c=SERIES_1, alpha=0.40,
                   linewidths=0, label="referencia (o normal)")
        ax.scatter(cur2d[:, 0], cur2d[:, 1], s=13, c=SERIES_2, alpha=0.45,
                   linewidths=0, label="hoje")

        ax.set_title(title, color=TEXT_PRIMARY, fontsize=13, pad=14, loc="left")
        ax.text(0.0, 1.015, subtitle, transform=ax.transAxes, color=TEXT_SECONDARY,
                fontsize=9.5, va="bottom")

        # Recessive frame: the data is the content, the box is not.
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#d8d7d2")
        ax.tick_params(colors=TEXT_SECONDARY, labelsize=8, length=3)
        ax.set_xlabel("direcao principal 1", color=TEXT_SECONDARY, fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel("direcao principal 2", color=TEXT_SECONDARY, fontsize=9)

        # Same limits everywhere, or "moved" would be an artefact of the axes.
        ax.set_xlim(*XLIM)
        ax.set_ylim(*YLIM)

    # One legend for the figure: identity is never colour-alone.
    handles, labels = axes[0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
                     fontsize=10, markerscale=2.4, bbox_to_anchor=(0.5, -0.015))
    for text in leg.get_texts():
        text.set_color(TEXT_PRIMARY)

    fig.suptitle("Cada ponto e uma imagem. Cada imagem sao 2048 numeros, lidos como coordenadas.",
                 color=TEXT_PRIMARY, fontsize=12.5, y=1.0, x=0.008, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))

    out = ROOT / "assets" / "embedding_clouds.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
