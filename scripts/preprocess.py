import cv2
import numpy as np
from pathlib import Path

def preprocess_basic(image):
    """Grayscale simple — comme tu avais fait"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray

def preprocess_threshold(image):
    """Grayscale + threshold binaire — comme tu avais fait"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    return thresh

def preprocess_blur(image):
    """Grayscale + blur — comme tu avais fait"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return blur

def preprocess_optimized(image):
    """
    Preprocessing optimisé pour les crops de plaques — nouveau.
    Combine resize + débruitage + contraste adaptatif.
    C'est celui qu'on utilise dans batch_ocr.py
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Agrandir x2 — essentiel pour les petits crops
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    # Débruitage léger
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    # Contraste adaptatif
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    return gray

def save_preprocessed(image, save_path):
    """Sauvegarde une image preprocessée"""
    cv2.imwrite(str(save_path), image)


# =============================================
# TEST VISUEL — sur tes images de test manuelles
# =============================================
if __name__ == "__main__":
    import easyocr
    import pytesseract

    BASE_DIR = Path(__file__).resolve().parent
    img_path = BASE_DIR.parent / "data" / "raw" / "plate.jpg"
    processed_dir = BASE_DIR.parent / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(img_path))
    if image is None:
        print("Erreur : image non trouvée")
        exit()

    reader = easyocr.Reader(['en'], verbose=False)

    versions = [
        ("ORIGINALE",   image),
        ("GRAYSCALE",   preprocess_basic(image)),
        ("THRESHOLD",   preprocess_threshold(image)),
        ("BLUR",        preprocess_blur(image)),
        ("OPTIMISÉE",   preprocess_optimized(image)),
    ]

    for label, img in versions:
        save_path = processed_dir / f"{label.lower()}.jpg"
        cv2.imwrite(str(save_path), img)

        tess = pytesseract.image_to_string(img).strip()
        easy = reader.readtext(str(save_path))
        easy_text = " ".join([r[1] for r in easy])

        print(f"\n===== {label} =====")
        print("Tesseract :", tess)
        print("EasyOCR   :", easy_text)