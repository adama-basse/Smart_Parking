import cv2
import pytesseract
import easyocr
from pathlib import Path

# Dossier du script actuel
BASE_DIR = Path(__file__).resolve().parent

# Chemins vers les images
img1_path = BASE_DIR.parent / "data" / "raw" / "plate.jpg"
img2_path = BASE_DIR.parent / "data" / "raw" / "plate2.jpg"
img3_path = BASE_DIR.parent / "data" / "raw" / "plate3.jpg"

print("Image 1 :", img1_path)
print("Image 2 :", img2_path)
print("Image 3 :", img3_path)

# Initialiser EasyOCR une seule fois
reader = easyocr.Reader(['en'], verbose=False)

# =====  Image facile =====
image = cv2.imread(str(img1_path))

if image is None:
    print("Erreur : image 1 non trouvée")
else:
    print("Image 1 chargée")
    text_tesseract_1 = pytesseract.image_to_string(image).strip()
    print("Tesseract (facile):", text_tesseract_1)

    result_1 = reader.readtext(str(img1_path))
    text_easyocr_1 = " ".join([res[1] for res in result_1])
    print("EasyOCR (facile):", text_easyocr_1)

# =====  Image floue =====
image_blurred = cv2.imread(str(img2_path))

if image_blurred is None:
    print("Erreur : image 2 non trouvée")
else:
    print("Image 2 chargée")
    text_tesseract_2 = pytesseract.image_to_string(image_blurred).strip()
    print("Tesseract (floue):", text_tesseract_2)

    result_2 = reader.readtext(str(img2_path))
    text_easyocr_2 = " ".join([res[1] for res in result_2])
    print("EasyOCR (floue):", text_easyocr_2)

# =====  Image inclinée =====
image_rotated = cv2.imread(str(img3_path))

if image_rotated is None:
    print("Erreur : image 3 non trouvée")
else:
    print("Image 3 chargée")
    text_tesseract_3 = pytesseract.image_to_string(image_rotated).strip()
    print("Tesseract (inclinée):", text_tesseract_3)

    result_3 = reader.readtext(str(img3_path))
    text_easyocr_3 = " ".join([res[1] for res in result_3])
    print("EasyOCR (inclinée):", text_easyocr_3)