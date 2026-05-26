from ultralytics import YOLO
import cv2
from pathlib import Path
import time

def detect_and_crop(images_dir, model_path="models/best_v8.pt", save_dir="data/cropped"):
    model = YOLO(model_path)
    images_dir = Path(images_dir)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    all_images = list(images_dir.glob("*.jpg"))
    print(f"📂 {len(all_images)} images trouvées dans {images_dir}")

    total_crops = 0
    total_no_detection = 0
    start_total = time.time()

    for i, img_path in enumerate(all_images):
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        results = model(image, verbose=False)
        boxes = results[0].boxes.xyxy

        if len(boxes) == 0:
            total_no_detection += 1
            continue

        for j, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            # Petite marge pour ne pas couper les bords
            x1 = max(0, x1 - 5)
            y1 = max(0, y1 - 5)
            x2 = min(image.shape[1], x2 + 5)
            y2 = min(image.shape[0], y2 + 5)

            crop = image[y1:y2, x1:x2]
            crop_path = save_dir / f"{img_path.stem}_crop_{j}.jpg"
            cv2.imwrite(str(crop_path), crop)
            total_crops += 1

        # Affiche progression toutes les 100 images
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_total
            print(f"  ⏳ {i+1}/{len(all_images)} images traitées — {total_crops} crops — {elapsed:.0f}s écoulées")

    elapsed_total = time.time() - start_total
    print(f"\n✅ Terminé en {elapsed_total:.1f}s")
    print(f"   {total_crops} crops sauvegardés dans {save_dir}")
    print(f"   {total_no_detection} images sans détection")
    return total_crops
def detect_and_crop_array(image_bgr, model_path="models/best_v8.pt"):
    """
    Retourne (crop, image_annotée) ou (None, image_originale)
    """
    model   = YOLO(model_path)
    results = model(image_bgr, verbose=False)

    if not results[0].boxes.xyxy.shape[0]:
        return None, image_bgr

    # Meilleure détection
    best_idx        = results[0].boxes.conf.argmax()
    box             = results[0].boxes.xyxy[best_idx]
    conf_yolo       = float(results[0].boxes.conf[best_idx])
    x1, y1, x2, y2 = map(int, box)

    # Crop
    x1c = max(0, x1 - 5)
    y1c = max(0, y1 - 5)
    x2c = min(image_bgr.shape[1], x2 + 5)
    y2c = min(image_bgr.shape[0], y2 + 5)
    crop = image_bgr[y1c:y2c, x1c:x2c]

    # Image annotée avec rectangle + label
    annotated = image_bgr.copy()
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (6, 182, 212), 3)
    label = f"Plaque {conf_yolo:.0%}"
    cv2.rectangle(annotated, (x1, y1-35), (x1+len(label)*13, y1), (6, 182, 212), -1)
    cv2.putText(annotated, label, (x1+5, y1-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return crop, annotated

if __name__ == "__main__":
    print("=== YOLO Crop — valid ===")
    detect_and_crop(
        images_dir="data/dataset/valid",
        model_path="models/best_v8.pt",
        save_dir="data/cropped/valid"
    )