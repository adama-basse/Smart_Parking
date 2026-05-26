from matplotlib.pyplot import box

import cv2
import easyocr
import json
import time
import numpy as np
from pathlib import Path
from clean_text import clean_plate_text

# =============================================
# CONFIGURATION
# =============================================
CROPPED_DIR = Path("data/cropped/valid")
OUTPUT_DIR  = Path("results/metrics")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Un seul reader EN — pas d'arabe pour éviter la confusion
reader = easyocr.Reader(['en'], gpu=False, verbose=False)
print("✅ EasyOCR prêt\n")

# =============================================
# PREPROCESSING ADAPTÉ AUX PLAQUES
# =============================================
from preprocess import preprocess_optimized

# =============================================
# PIPELINE
# =============================================
results = []
all_images = list(CROPPED_DIR.glob("*.jpg"))
print(f"📂 {len(all_images)} crops à traiter\n")

for i, img_path in enumerate(all_images):
    image = cv2.imread(str(img_path))
    if image is None:
        continue

    processed = preprocess_optimized(image)

    # EasyOCR sur image preprocessée
    t0 = time.time()
    easy_result = reader.readtext(processed,width_ths=0.9,contrast_ths=0.1,adjust_contrast=0.5)
    easy_time = time.time() - t0

    if easy_result:
        # Filtrer par confiance minimale
        filtered = [r for r in easy_result if r[2] > 0.3]
        if not filtered:
            filtered = easy_result

        # Trier par surface décroissante — le texte principal est le plus grand
        def box_area(r):
            pts = r[0]
            w = max(pts[1][0], pts[2][0]) - min(pts[0][0], pts[3][0])
            h = max(pts[2][1], pts[3][1]) - min(pts[0][1], pts[1][1])
            return abs(w * h)

        filtered = sorted(filtered, key=box_area, reverse=True)

    # Garder seulement les boîtes dont la surface > 15% de la plus grande
        max_area = box_area(filtered[0])
        main     = [r for r in filtered if box_area(r) > max_area * 0.15]

    # Trier de gauche à droite
        main     = sorted(main, key=lambda r: r[0][0][0])
        easy_raw = ' '.join([r[1] for r in main])
        easy_conf = sum(r[2] for r in main) / len(main)
    else:
        easy_raw, easy_conf = "", 0.0
    
    cleaned = clean_plate_text(easy_raw, easyocr_confidence=easy_conf)

    results.append({
        "image"     : img_path.name,
        "raw"       : easy_raw,
        "cleaned"   : cleaned['cleaned'],
        "valid"     : cleaned['valid'],
        "plate_type": cleaned['plate_type'],
        "confidence": round(easy_conf, 2),
        "time_s"    : round(easy_time, 3)
    })

    if (i + 1) % 20 == 0:
        valid = sum(1 for r in results if r['valid'])
        print(f"⏳ {i+1}/{len(all_images)} — {valid} valides")

# =============================================
# RÉSULTATS
# =============================================
output_path = OUTPUT_DIR / "batch_results.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

total       = len(results)
lisibles    = sum(1 for r in results if r['valid'])
marocaines  = sum(1 for r in results if r['plate_type'] == 'marocaine')
europeennes = sum(1 for r in results if r['plate_type'] == 'européenne')
autres      = sum(1 for r in results if r['plate_type'] == 'autre')
vides       = sum(1 for r in results if not r['raw'])
t_avg       = sum(r['time_s'] for r in results) / total

print(f"""
{'='*55}
RÉSULTATS — {total} images
{'='*55}
Plaques lisibles        : {lisibles} ({lisibles/total*100:.1f}%)
  dont marocaines       : {marocaines} ({marocaines/total*100:.1f}%)
  dont européennes      : {europeennes} ({europeennes/total*100:.1f}%)
  dont autres (vanity)  : {autres} ({autres/total*100:.1f}%)
Aucun texte lu          : {vides} ({vides/total*100:.1f}%)
Temps moyen/image       : {t_avg:.2f}s
Temps total estimé      : {t_avg * 1171 / 60:.0f} min sur dataset complet
{'='*55}
""")