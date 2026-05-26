import cv2
import pytesseract
import easyocr
import json
import time
from pathlib import Path
from clean_text import clean_plate_text

# =============================================
# CONFIGURATION
# =============================================

CROPPED_DIR = Path("data/cropped/valid")
OUTPUT_DIR  = Path("results/tesseract")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Config Tesseract optimisée pour les plaques
TESS_CONFIG = "--psm 7 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789|"


# =============================================
# PIPELINE
# =============================================

results = []
all_images = list(CROPPED_DIR.glob("*.jpg"))[:20]
print(f"📂 {len(all_images)} crops trouvés\n")

for i, img_path in enumerate(all_images):
    image = cv2.imread(str(img_path))
    if image is None:
        continue

    # ---------- TESSERACT ----------
    t0 = time.time()
    tess_raw = pytesseract.image_to_string(image, config=TESS_CONFIG).strip()
    tess_time = time.time() - t0
    tess_clean = clean_plate_text(tess_raw)

   
    # ---------- SAUVEGARDE ----------
    results.append({
        "image": img_path.name,
        "tesseract": {
            "raw"    : tess_raw,
            "cleaned": tess_clean['cleaned'],
            "valid"  : tess_clean['valid'],
            "type"   : tess_clean['plate_type'],
            "time_s" : round(tess_time, 3)
        },
        
    })

    # Progression toutes les 50 images
    if (i + 1) % 50 == 0:
        tess_ok = sum(1 for r in results if r['tesseract']['valid'])
        print(f"⏳ {i+1}/{len(all_images)} | Tess: {tess_ok} valides")



# =============================================
# SAUVEGARDE JSON
# =============================================

output_path = OUTPUT_DIR / "batch_results_tesseract.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

# =============================================
# RÉSUMÉ FINAL
# =============================================

total = len(results)
tess_valid = sum(1 for r in results if r['tesseract']['valid'])
tess_maroc = sum(1 for r in results if r['tesseract']['type'] == 'marocaine')
tess_eu    = sum(1 for r in results if r['tesseract']['type'] == 'étrangère')
tess_time_avg = sum(r['tesseract']['time_s'] for r in results) / total

print(f"""
{'='*55}
RÉSULTATS FINAUX — {total} images
{'='*55}
{'Métrique':<35} {'Tesseract':>8} 
{'-'*55}
{'Plaques valides lues':<35} {tess_valid:>8} 
{'Taux de lecture (%)':<35} {tess_valid/total*100:>7.1f}% 
{'  dont marocaines':<35} {tess_maroc:>8} 
{'  dont étrangères':<35} {tess_eu:>8} 
{'Temps moyen / image (s)':<35} {tess_time_avg:>8.3f} 
{'='*55}
""")

print(f"✅ Résultats sauvegardés → {output_path}")