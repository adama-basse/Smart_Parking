import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# =============================================
# CHARGEMENT DES DONNÉES
# =============================================
results_path = Path("results/metrics/batch_results.json")
with open(results_path, "r", encoding="utf-8") as f:
    results = json.load(f)

total       = len(results)
lisibles    = sum(1 for r in results if r['valid'])
marocaines  = sum(1 for r in results if r['plate_type'] == 'marocaine')
europeennes = sum(1 for r in results if r['plate_type'] == 'européenne')
autres      = sum(1 for r in results if r['plate_type'] == 'autre')
vides       = sum(1 for r in results if not r['raw'])
t_avg       = sum(r['time_s'] for r in results) / total

# Confidences
confs       = [r['confidence'] for r in results if r['confidence'] and r['confidence'] > 0]
conf_avg    = sum(confs) / len(confs) if confs else 0
conf_high   = sum(1 for c in confs if c >= 0.7)
conf_mid    = sum(1 for c in confs if 0.3 <= c < 0.7)
conf_low    = sum(1 for c in confs if c < 0.3)

# =============================================
# AFFICHAGE CONSOLE
# =============================================
print(f"""
{'='*60}
RAPPORT COMPARATIF OCR — {total} images
{'='*60}

📊 TAUX DE LECTURE
  Plaques lisibles        : {lisibles:>5} ({lisibles/total*100:.1f}%)
  dont marocaines         : {marocaines:>5} ({marocaines/total*100:.1f}%)
  dont européennes        : {europeennes:>5} ({europeennes/total*100:.1f}%)
  dont autres (vanity)    : {autres:>5} ({autres/total*100:.1f}%)
  Aucun texte lu          : {vides:>5} ({vides/total*100:.1f}%)

⏱️  PERFORMANCE
  Temps moyen / image     : {t_avg:.3f}s
  Temps total estimé      : {t_avg * total / 60:.1f} min

🎯 CONFIANCE EASYOCR
  Confiance moyenne       : {conf_avg:.2f}
  Haute (≥0.7)            : {conf_high:>5} ({conf_high/total*100:.1f}%)
  Moyenne (0.3-0.7)       : {conf_mid:>5} ({conf_mid/total*100:.1f}%)
  Faible (<0.3)           : {conf_low:>5} ({conf_low/total*100:.1f}%)
{'='*60}
""")

# =============================================
# GRAPHIQUES
# =============================================
output_dir = Path("results/figures")
output_dir.mkdir(parents=True, exist_ok=True)

# Style sombre — cohérent avec le projet
plt.style.use("dark_background")
COLORS = {
    "cyan"  : "#06B6D4",
    "green" : "#10B981",
    "amber" : "#F59E0B",
    "blue"  : "#1A56DB",
    "red"   : "#EF4444",
    "gray"  : "#64748B",
    "bg"    : "#0A1628",
    "card"  : "#0D2137",
}

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor(COLORS["bg"])
fig.suptitle("Analyse OCR — Smart Parking", 
             color="white", fontsize=16, fontweight="bold", y=1.02)

# ---- Graphique 1 : Distribution des types de plaques ----
ax1 = axes[0]
ax1.set_facecolor(COLORS["card"])
labels  = ["Marocaines", "Européennes", "Vanity/Autres", "Non lues"]
values  = [marocaines, europeennes, autres, vides]
colors  = [COLORS["amber"], COLORS["blue"], COLORS["cyan"], COLORS["gray"]]
explode = (0.05, 0.05, 0.05, 0.05)

wedges, texts, autotexts = ax1.pie(
    values, labels=labels, colors=colors,
    autopct="%1.1f%%", explode=explode,
    textprops={"color": "white", "fontsize": 9},
    wedgeprops={"linewidth": 1.5, "edgecolor": COLORS["bg"]}
)
for at in autotexts:
    at.set_fontsize(8)
ax1.set_title("Distribution des types", color=COLORS["cyan"], 
              fontsize=12, fontweight="bold", pad=15)

# ---- Graphique 2 : Taux de lecture (barres) ----
ax2 = axes[1]
ax2.set_facecolor(COLORS["card"])
categories = ["Lisibles\n(total)", "Marocaines", "Européennes", "Vanity", "Non lues"]
vals       = [lisibles, marocaines, europeennes, autres, vides]
bar_colors = [COLORS["green"], COLORS["amber"], COLORS["blue"], 
              COLORS["cyan"], COLORS["gray"]]

bars = ax2.bar(categories, [v/total*100 for v in vals], 
               color=bar_colors, width=0.6, edgecolor=COLORS["bg"], linewidth=1.5)

# Valeur au-dessus de chaque barre
for bar, val in zip(bars, vals):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f"{val/total*100:.1f}%", ha="center", va="bottom",
             color="white", fontsize=9, fontweight="bold")

ax2.set_ylim(0, 95)
ax2.set_ylabel("Pourcentage (%)", color="white", fontsize=10)
ax2.tick_params(colors="white", labelsize=8)
ax2.spines[["top","right","left","bottom"]].set_color(COLORS["gray"])
ax2.set_title("Taux de lecture (%)", color=COLORS["cyan"],
              fontsize=12, fontweight="bold")
ax2.yaxis.grid(True, color=COLORS["gray"], alpha=0.3, linestyle="--")

# ---- Graphique 3 : Distribution des confidences ----
ax3 = axes[2]
ax3.set_facecolor(COLORS["card"])
conf_labels = ["Haute\n(≥0.7)", "Moyenne\n(0.3-0.7)", "Faible\n(<0.3)"]
conf_vals   = [conf_high, conf_mid, conf_low]
conf_colors = [COLORS["green"], COLORS["amber"], COLORS["red"]]

bars3 = ax3.bar(conf_labels, [v/total*100 for v in conf_vals],
                color=conf_colors, width=0.5, edgecolor=COLORS["bg"], linewidth=1.5)

for bar, val in zip(bars3, conf_vals):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f"{val}", ha="center", va="bottom",
             color="white", fontsize=10, fontweight="bold")

ax3.set_ylim(0, 80)
ax3.set_ylabel("Pourcentage (%)", color="white", fontsize=10)
ax3.tick_params(colors="white", labelsize=9)
ax3.spines[["top","right","left","bottom"]].set_color(COLORS["gray"])
ax3.set_title(f"Confiance EasyOCR\n(moy: {conf_avg:.2f})", color=COLORS["cyan"],
              fontsize=12, fontweight="bold")
ax3.yaxis.grid(True, color=COLORS["gray"], alpha=0.3, linestyle="--")

plt.tight_layout()
out_path = output_dir / "compare_ocr.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight",
            facecolor=COLORS["bg"], edgecolor="none")
plt.close()
print(f"✅ Graphique sauvegardé → {out_path}")

# =============================================
# AXE LOW-COST — Simulation Raspberry Pi
# =============================================
print(f"""
{'='*55}
AXE LOW-COST — Simulation Raspberry Pi 4
{'='*55}

📊 PERFORMANCE MESURÉE (CPU only, gpu=False)
  Temps moyen / image     : {t_avg:.2f}s
  Images / minute         : {60/t_avg:.1f}
  Images / heure          : {3600/t_avg:.0f}

 COMPATIBILITÉ RASPBERRY PI 4
  Processeur              : ARM Cortex-A72 (4 cœurs)
  RAM recommandée         : 4 GB
  Coût matériel           : ~50€
  GPU                     : Non requis (cpu only) ✅

🚗 CAPACITÉ PARKING
  Débit max               : {60/t_avg:.0f} véhicules/minute
  Adapté trafic faible    : {'✅ Oui' if t_avg < 10 else '❌ Non'}
  Adapté trafic moyen     : {'✅ Oui' if t_avg < 5 else '⚠️ Limite'}

 OPTIMISATIONS POSSIBLES
  YOLOv8n (nano)          : ~2x plus rapide
  Quantification INT8     : ~3x plus rapide
  Résolution réduite      : ~1.5x plus rapide
  Temps estimé optimisé   : ~{t_avg/3:.2f}s/image
{'='*55}
""")

# Graphique low-cost
fig_lc, ax_lc = plt.subplots(1, 2, figsize=(12, 4))
fig_lc.patch.set_facecolor(COLORS["bg"])
fig_lc.suptitle("Axe Low-cost — Simulation Raspberry Pi",
                color="white", fontsize=14, fontweight="bold")

# Graphique 1 : comparaison temps actuel vs optimisé
ax_lc[0].set_facecolor(COLORS["card"])
configs    = ["Actuel\n(EasyOCR)", "YOLOv8n\n+EasyOCR", "Optimisé\n(INT8)"]
temps      = [t_avg, t_avg * 0.6, t_avg / 3]
bar_colors = [COLORS["amber"], COLORS["blue"], COLORS["green"]]
bars       = ax_lc[0].bar(configs, temps, color=bar_colors,
                           width=0.5, edgecolor=COLORS["bg"], linewidth=1.5)
for bar, val in zip(bars, temps):
    ax_lc[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                  f"{val:.2f}s", ha="center", va="bottom",
                  color="white", fontsize=10, fontweight="bold")
ax_lc[0].set_ylabel("Temps / image (s)", color="white")
ax_lc[0].tick_params(colors="white")
ax_lc[0].spines[["top","right","left","bottom"]].set_color(COLORS["gray"])
ax_lc[0].set_title("Temps de traitement", color=COLORS["cyan"], fontweight="bold")
ax_lc[0].yaxis.grid(True, color=COLORS["gray"], alpha=0.3, linestyle="--")

# Graphique 2 : coût matériel comparaison
ax_lc[1].set_facecolor(COLORS["card"])
materiel   = ["Raspberry Pi 4\n(notre choix)", "Jetson Nano", "PC Standard"]
couts      = [50, 150, 600]
mat_colors = [COLORS["green"], COLORS["amber"], COLORS["red"]]
bars2      = ax_lc[1].bar(materiel, couts, color=mat_colors,
                           width=0.5, edgecolor=COLORS["bg"], linewidth=1.5)
for bar, val in zip(bars2, couts):
    ax_lc[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                  f"{val}€", ha="center", va="bottom",
                  color="white", fontsize=10, fontweight="bold")
ax_lc[1].set_ylabel("Coût (€)", color="white")
ax_lc[1].tick_params(colors="white")
ax_lc[1].spines[["top","right","left","bottom"]].set_color(COLORS["gray"])
ax_lc[1].set_title("Comparaison coût matériel", color=COLORS["cyan"], fontweight="bold")
ax_lc[1].yaxis.grid(True, color=COLORS["gray"], alpha=0.3, linestyle="--")

plt.tight_layout()
lc_path = output_dir / "lowcost.png"
plt.savefig(lc_path, dpi=150, bbox_inches="tight",
            facecolor=COLORS["bg"], edgecolor="none")
plt.close()
print(f"✅ Graphique low-cost sauvegardé → {lc_path}")