"""Generate docs/assets/methodology.png, the general three-phase methodology diagram shown in the README."""

from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt


def box(ax, xy, w, h, title, body, face, edge, title_color, body_color="#424242", lw=1.5):
    """Draw a rounded box with a bold title and a short body."""
    ax.add_patch(
        patches.FancyBboxPatch(
            xy, w, h, boxstyle="round,pad=0.1,rounding_size=0.15", facecolor=face, edgecolor=edge, lw=lw
        )
    )
    cx = xy[0] + w / 2
    ax.text(cx, xy[1] + h - 0.3, title, fontsize=10, fontweight="bold", ha="center", va="center", color=title_color)
    ax.text(cx, xy[1] + h - 0.55, body, fontsize=8.3, ha="center", va="top", color=body_color, linespacing=1.5)


def phase(ax, x, title, subtitle, face, edge, color, sub_color):
    """Draw a dashed phase panel with its heading."""
    ax.add_patch(
        patches.FancyBboxPatch(
            (x, 1.2),
            5.4,
            6.4,
            boxstyle="round,pad=0.2,rounding_size=0.3",
            facecolor=face,
            edgecolor=edge,
            lw=2,
            linestyle="--",
        )
    )
    ax.text(x + 2.7, 7.3, title, fontsize=13, fontweight="bold", ha="center", color=color)
    ax.text(x + 2.7, 7.0, subtitle, fontsize=10, ha="center", color=sub_color)


def arrow(ax, start, end, color, lw=1.8, style="->", linestyle="-"):
    """Draw an arrow from start to end."""
    ax.annotate(
        "", xy=end, xytext=start, arrowprops={"arrowstyle": style, "lw": lw, "color": color, "linestyle": linestyle}
    )


def create_diagram(output_path: str = "docs/assets/methodology.png") -> None:
    """Render the methodology diagram.

    Args:
        output_path: Destination PNG path.
    """
    fig, ax = plt.subplots(figsize=(18, 9), dpi=300)
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 9)
    ax.axis("off")
    fig.patch.set_facecolor("#FAFAFA")

    title = "LeukoAdapt: Methodology Overview"
    ax.text(9.0, 8.5, title, fontsize=20, fontweight="bold", ha="center", color="#1A237E")
    subtitle = "Unsupervised attention-guided domain adaptation for Acute Lymphoblastic Leukemia diagnosis"
    ax.text(9.0, 8.1, subtitle, fontsize=12, style="italic", ha="center", color="#424242")

    # Phase 1: unsupervised domain translation
    blue, red = "#1A237E", "#C62828"
    phase(ax, 0.5, "PHASE 1: Domain Translation", "Unpaired, unsupervised GAN", "#E8EAF6", "#3F51B5", blue, "#5C6BC0")
    box(
        ax, (0.8, 4.9), 2.3, 1.7, "Source Domain",
        "C-NMC segmented cells\nLabelled (ALL / Normal)\nPasted onto real\ntarget backgrounds",
        "#FFFFFF", "#3949AB", "#283593",
    )  # fmt: skip
    box(
        ax, (0.8, 2.1), 2.3, 1.7, "Target Domain",
        "ALL-IDB cells with\nblood background\nLabels never used\nby the classifier", "#FFFFFF", "#3949AB", "#283593",
    )  # fmt: skip
    box(
        ax, (3.45, 4.9), 2.25, 1.7, "Attention Generator",
        "Spatial attention blocks\nRestyles only the cell\n(known cell mask);\nbackground left untouched",
        "#C5CAE9", blue, blue, body_color="#283593", lw=2,
    )  # fmt: skip
    box(
        ax, (3.45, 2.1), 2.25, 1.7, "PatchGAN Discriminator",
        "Compares whole translated\nand real target images",
        "#FFCDD2", red, "#B71C1C",
    )  # fmt: skip
    arrow(ax, (3.1, 5.75), (3.45, 5.75), blue)
    arrow(ax, (3.1, 2.95), (3.45, 2.95), red)
    arrow(ax, (1.95, 3.8), (1.95, 4.9), "#3949AB", linestyle="--")
    ax.text(2.02, 4.35, "cell-free\nbackground\npatches", fontsize=7.5, ha="left", va="center", color="#3949AB")
    arrow(ax, (4.575, 3.8), (4.575, 4.9), "#D32F2F", style="<->", linestyle=":")
    ax.text(4.65, 4.35, "Adversarial +\ncycle-consistency +\npixel loss", fontsize=7.5, fontweight="bold",
            ha="left", va="center", color=red)  # fmt: skip

    # Phase 2: classifier training on translated source images
    orange = "#E65100"
    phase(ax, 6.3, "PHASE 2: Classifier Training", "Supervised on the translated source", "#FFF8E1", "#FFA000",
          orange, "#FB8C00")  # fmt: skip
    arrow(ax, (5.7, 5.75), (6.6, 5.75), blue, lw=2.5)
    box(
        ax, (6.6, 4.9), 2.25, 1.7, "Translated Source",
        "Source cells in target style\nLabels inherited\nfrom the source", "#FFFFFF", "#FFB300", orange,
    )  # fmt: skip
    box(
        ax, (9.25, 4.9), 2.2, 1.7, "ResNet34 Classifier",
        "ImageNet-pretrained\nLearns ALL vs Normal\nCross-entropy loss", "#FFE082", orange, "#BF360C",
        body_color="#5D4037", lw=2,
    )  # fmt: skip
    arrow(ax, (8.85, 5.75), (9.25, 5.75), orange, lw=2)
    box(
        ax, (7.75, 2.1), 2.7, 1.7, "Trained Classifier",
        "Never sees target labels\nModel selection without\ntarget test data", "#FFFFFF", "#FFA000", orange,
    )  # fmt: skip
    arrow(ax, (10.35, 4.9), (9.4, 3.8), orange)

    # Phase 3: blind evaluation on the target domain
    green = "#1B5E20"
    phase(ax, 12.1, "PHASE 3: Target Evaluation", "Blind test on unseen target cells", "#E8F5E9", "#43A047", green,
          "#4CAF50")  # fmt: skip
    arrow(ax, (11.45, 5.75), (12.4, 5.75), "#2E7D32", lw=2.5)
    box(
        ax, (12.4, 4.9), 2.25, 1.7, "Target Test Set",
        "Real, unmodified ALL-IDB cells\nScenario 1: cell-level split\nScenario 2: slide-group split",
        "#FFFFFF", "#4CAF50", green,
    )  # fmt: skip
    box(
        ax, (15.05, 4.9), 2.2, 1.7, "Inference",
        "Single forward pass\nPer-image predictions", "#A5D6A7", green, green, body_color=green, lw=2,
    )  # fmt: skip
    arrow(ax, (14.65, 5.75), (15.05, 5.75), "#2E7D32", lw=2)
    box(
        ax, (13.2, 2.1), 3.3, 1.7, "Evaluation",
        "Accuracy, balanced accuracy, AUROC\nPrecision, recall, specificity, F-score\nStatistical comparison (McNemar)",
        "#FFFFFF", "#2E7D32", green, body_color="#2E7D32", lw=2,
    )  # fmt: skip
    arrow(ax, (16.15, 4.9), (15.3, 3.8), "#2E7D32")

    footnote = "Target labels never reach the classifier: it learns only from labels inherited from the source domain."
    ax.text(9.0, 0.6, footnote, fontsize=9, style="italic", ha="center", color="#616161")

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_file, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"Methodology diagram saved to {out_file.resolve()}")


if __name__ == "__main__":
    create_diagram()
