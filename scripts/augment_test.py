import cv2
import numpy as np
import easyocr
import matplotlib.pyplot as plt
from pathlib import Path
from preprocess import preprocess_optimized
from clean_text import clean_plate_text

# =============================================
# CONFIGURATION
# =============================================
CROPS_DIR  = Path("data/cropped/valid")
OUTPUT_DIR = Path("results/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

reader = easyocr.Reader(['en'], gpu=False, verbose=False)

# =============================================
# FONCTIONS DE DÉGRADATION
# =============================================
def apply_blur(image, level=7):
    """Simule une image floue (mouvement, mise au point)"""
    return cv2.GaussianBlur(image, (level, level), 0)

def apply_night(image, alpha=0.35):
    """Simule une prise de nuit (sous-exposition)"""
    return cv2.convertScaleAbs(image, alpha=alpha, beta=0)

def apply_noise(image, sigma=35):
    """Simule du bruit (capteur bas de gamme)"""
    noise  = np.random.normal(0, sigma, image.shape).astype(np.int16)
    noisy  = np.clip(image.astype(np.int16) + noise, 0, 255)
    return noisy.astype(np.uint8)

def apply_angle(image, degrees=20):
    """Simule une plaque inclinée"""
    h, w = image.shape[:2]
    M    = cv2.getRotationMatrix2D((w//2, h//2), degrees, 1)
    return cv2.warpAffine(image, M, (w, h),
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(200, 200, 200))

CONDITIONS = {
    "Original" : lambda img: img,
    "Flou"     : apply_blur,
    "Nuit"     : apply_night,
    "Bruit"    : apply_noise,
    "Incliné"  : apply_angle,
}

# =============================================
# OCR SUR UNE IMAGE
# =============================================
def run_ocr(image):
    processed = preprocess_optimized(image)
    results   = reader.readtext(processed)
    if not results:
        return "", 0.0
    best = max(results, key=lambda x: x[2])
    raw, conf = best[1], best[2]
    cleaned = clean_plate_text(raw, easyocr_confidence=conf)
    return cleaned['cleaned'], conf

# =============================================
# TEST SUR N IMAGES
# =============================================
N_IMAGES = 10
test_images = list(CROPS_DIR.glob("*.jpg"))[:N_IMAGES]

print(f"Test sur {len(test_images)} images\n")
print(f"{'Condition':<12} {'Lues':>6} {'Taux':>8} {'Conf moy':>10}")
print("=" * 42)

# Stocker les résultats par condition
results_by_cond = {cond: {"lues": 0, "confs": []} for cond in CONDITIONS}

for img_path in test_images:
    image = cv2.imread(str(img_path))
    if image is None:
        continue
    for cond_name, cond_fn in CONDITIONS.items():
        degraded       = cond_fn(image)
        text, conf     = run_ocr(degraded)
        if text:
            results_by_cond[cond_name]["lues"] += 1
        results_by_cond[cond_name]["confs"].append(conf)

# Affichage console
for cond_name, data in results_by_cond.items():
    lues     = data["lues"]
    taux     = lues / len(test_images) * 100
    conf_avg = sum(data["confs"]) / len(data["confs"]) if data["confs"] else 0
    print(f"{cond_name:<12} {lues:>6} {taux:>7.0f}% {conf_avg:>10.2f}")

# =============================================
# GRAPHIQUE COMPARATIF
# =============================================
COLORS_MAP = {
    "Original" : "#10B981",
    "Flou"     : "#F59E0B",
    "Nuit"     : "#1A56DB",
    "Bruit"    : "#EF4444",
    "Incliné"  : "#06B6D4",
}

taux_list = []
conf_list = []
for cond_name, data in results_by_cond.items():
    taux_list.append(data["lues"] / len(test_images) * 100)
    conf_list.append(
        sum(data["confs"]) / len(data["confs"]) if data["confs"] else 0
    )

plt.style.use("dark_background")
BG   = "#0A1628"
CARD = "#0D2137"
GRAY = "#64748B"

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor(BG)
fig.suptitle("Robustesse OCR — Conditions difficiles",
             color="white", fontsize=15, fontweight="bold")

cond_names = list(CONDITIONS.keys())
bar_colors = [COLORS_MAP[c] for c in cond_names]

# ── Graphique 1 : Taux de lecture ──────────────────
ax1 = axes[0]
ax1.set_facecolor(CARD)
bars1 = ax1.bar(cond_names, taux_list, color=bar_colors,
                width=0.55, edgecolor=BG, linewidth=1.5)
for bar, val in zip(bars1, taux_list):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f"{val:.0f}%", ha="center", va="bottom",
             color="white", fontsize=11, fontweight="bold")
ax1.set_ylim(0, 115)
ax1.set_ylabel("Taux de lecture (%)", color="white", fontsize=11)
ax1.tick_params(colors="white", labelsize=10)
ax1.spines[["top","right","left","bottom"]].set_color(GRAY)
ax1.yaxis.grid(True, color=GRAY, alpha=0.3, linestyle="--")
ax1.set_title("Taux de lecture par condition",
              color="#06B6D4", fontsize=12, fontweight="bold")

# ── Graphique 2 : Confiance moyenne ────────────────
ax2 = axes[1]
ax2.set_facecolor(CARD)
bars2 = ax2.bar(cond_names, conf_list, color=bar_colors,
                width=0.55, edgecolor=BG, linewidth=1.5)
for bar, val in zip(bars2, conf_list):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f"{val:.2f}", ha="center", va="bottom",
             color="white", fontsize=11, fontweight="bold")
ax2.set_ylim(0, 1.15)
ax2.set_ylabel("Confiance moyenne EasyOCR", color="white", fontsize=11)
ax2.tick_params(colors="white", labelsize=10)
ax2.spines[["top","right","left","bottom"]].set_color(GRAY)
ax2.yaxis.grid(True, color=GRAY, alpha=0.3, linestyle="--")
ax2.set_title("Confiance par condition",
              color="#06B6D4", fontsize=12, fontweight="bold")

plt.tight_layout()
out_path = OUTPUT_DIR / "robustesse.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight",
            facecolor=BG, edgecolor="none")
plt.close()
print(f"\n✅ Graphique sauvegardé → {out_path}")

# =============================================
# VISUALISATION EXEMPLES (1 image × 5 conditions)
# =============================================
sample_img = cv2.imread(str(test_images[0]))
if sample_img is not None:
    fig2, axes2 = plt.subplots(1, 5, figsize=(20, 4))
    fig2.patch.set_facecolor(BG)
    fig2.suptitle("Exemples de conditions simulées",
                  color="white", fontsize=14, fontweight="bold")

    for ax, (cond_name, cond_fn) in zip(axes2, CONDITIONS.items()):
        degraded = cond_fn(sample_img)
        text, conf = run_ocr(degraded)
        ax.imshow(cv2.cvtColor(degraded, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{cond_name}\n'{text}'\n{conf:.0%}",
                     color=COLORS_MAP[cond_name], fontsize=9)
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_edgecolor(COLORS_MAP[cond_name])
            spine.set_linewidth(2)

    plt.tight_layout()
    ex_path = OUTPUT_DIR / "exemples_conditions.png"
    plt.savefig(ex_path, dpi=150, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close()
    print(f"✅ Exemples visuels sauvegardés → {ex_path}")