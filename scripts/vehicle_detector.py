from ultralytics import YOLO
import cv2
from pathlib import Path

# =============================================
# MODÈLES
# =============================================
MODEL_MARQUES_PATH = "models/best_marques.pt"

# Classes du modèle — on garde uniquement les *-car (pas les logos)
# Format : {class_id: "marque"}
CAR_CLASSES = {
    0:  "Abarth",      2:  "Acura",       4:  "Alfa Romeo",
    6:  "Audi",        8:  "Bentley",      10: "BMW",
    12: "Brabus",      14: "Bugatti",      16: "Cadillac",
    18: "Changan",     20: "Chery",        22: "Chevrolet",
    24: "Chrysler",    27: "Citroën",      28: "Cupra",
    30: "Dacia",       32: "DS",           34: "Ferrari",
    36: "Fiat",        38: "Ford",         40: "GMC",
    42: "Honda",       44: "Hyundai",      46: "Infiniti",
    48: "Jaguar",      50: "Jeep",         52: "Kia",
    54: "Lamborghini", 56: "Land Rover",   58: "Lexus",
    60: "Maserati",    62: "Mazda",        64: "Mercedes",
    66: "MG",          68: "Mini",         70: "Mitsubishi",
    72: "Nissan",      74: "Opel",         76: "Peugeot",
    78: "Porsche",     80: "Renault",      82: "Rolls-Royce",
    84: "Seat",        86: "Škoda",        88: "Subaru",
    90: "Suzuki",      92: "Tesla",        94: "Toyota",
    96: "Volkswagen",  98: "Volvo",
}

# Cache du modèle — chargé une seule fois
_model_marques = None

def _get_model():
    global _model_marques
    if _model_marques is None:
        _model_marques = YOLO(MODEL_MARQUES_PATH)
    return _model_marques


# =============================================
# DÉTECTION PRINCIPALE
# =============================================
def detect_vehicle(image_bgr):
    """
    Détecte la marque du véhicule dans l'image entière.

    Returns:
        dict : {
            'brand'     : "Toyota",
            'type'      : "car",
            'confidence': 0.87,
            'all_brands': [("Toyota", 0.87), ("Honda", 0.45)]
        }
    """
    model   = _get_model()
    results = model(image_bgr, verbose=False)

    detections = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        # Garder uniquement les classes *-car (IDs pairs)
        if cls_id in CAR_CLASSES and conf > 0.3:
            detections.append((CAR_CLASSES[cls_id], conf))

    if not detections:
        return {
            'brand'     : 'Non détecté',
            'type'      : 'véhicule',
            'confidence': 0.0,
            'all_brands': []
        }

    # Trier par confiance décroissante
    detections.sort(key=lambda x: x[1], reverse=True)
    best_brand, best_conf = detections[0]

    return {
        'brand'     : best_brand,
        'type'      : 'car',
        'confidence': best_conf,
        'all_brands': detections[:3]   # top 3 pour affichage
    }


# =============================================
# MATCHING PLAQUE + MARQUE
# =============================================
def match_vehicle(plate_result, vehicle_result, db_conn):
    """
    Stratégie d'identification combinée plaque + marque.

    Cas 1 — Plaque fiable (conf ≥ 0.7) + marque détectée
             → Identification certaine, on enregistre les deux

    Cas 2 — Plaque partielle (0.3 ≤ conf < 0.7)
             → Cherche dans l'historique une plaque similaire
             → Confirme avec la marque si match trouvé

    Cas 3 — Plaque illisible (conf < 0.3)
             → Marque comme identifiant principal
             → Alerte vérification manuelle

    Returns:
        dict : plate, brand, confidence_combined, status, message
    """
    plate      = plate_result.get('cleaned', '')
    plate_conf = plate_result.get('confidence') or 0.0
    valid      = plate_result.get('valid', False)
    brand      = vehicle_result.get('brand', 'Non détecté')
    veh_conf   = vehicle_result.get('confidence', 0.0)

    # ── Cas 1 : Plaque très fiable ──────────────────────────
    if plate_conf >= 0.7 and valid:
        return {
            'plate'               : plate,
            'brand'               : brand,
            'confidence_combined' : plate_conf,
            'status'              : 'certain',
            'message'             : f'✅ Identification certaine — {brand}'
        }

    # ── Cas 2 : Plaque partielle → historique ───────────────
    if 0.3 <= plate_conf < 0.7 and plate:
        # Cherche une plaque similaire (4 premiers caractères)
        prefix = plate[:4].replace('|', '')
        known  = db_conn.execute(
            "SELECT plate, brand FROM known_vehicles WHERE plate LIKE ?",
            (f"{prefix}%",)
        ).fetchone()

        if known:
            known_plate = known[0]
            known_brand = known[1] or brand
            # Confirmer avec la marque si cohérent
            if brand != 'Non détecté' and known_brand == brand:
                return {
                    'plate'               : known_plate,
                    'brand'               : brand,
                    'confidence_combined' : (plate_conf + veh_conf) / 2,
                    'status'              : 'probable',
                    'message'             : f'⚠️ Plaque probable — {brand} confirmé par historique'
                }
            return {
                'plate'               : known_plate,
                'brand'               : known_brand,
                'confidence_combined' : plate_conf,
                'status'              : 'probable',
                'message'             : f'⚠️ Plaque probable — trouvé dans historique'
            }

    # ── Cas 3 : Plaque illisible → marque principale ────────
    if not plate or plate_conf < 0.3:
        if brand != 'Non détecté':
            return {
                'plate'               : plate or 'ILLISIBLE',
                'brand'               : brand,
                'confidence_combined' : veh_conf * 0.5,
                'status'              : 'incertain',
                'message'             : f'❌ Plaque illisible — {brand} détecté (vérification requise)'
            }
        return {
            'plate'               : 'ILLISIBLE',
            'brand'               : 'Non détecté',
            'confidence_combined' : 0.0,
            'status'              : 'echec',
            'message'             : '❌ Plaque et véhicule non identifiés'
        }

    # ── Cas par défaut ───────────────────────────────────────
    return {
        'plate'               : plate,
        'brand'               : brand,
        'confidence_combined' : plate_conf,
        'status'              : 'standard',
        'message'             : f'🔵 Identification standard — {brand}'
    }


# =============================================
# TEST
# =============================================
if __name__ == "__main__":
    import sys
    test_img = sys.argv[1] if len(sys.argv) > 1 else None

    if test_img:
        image = cv2.imread(test_img)
        result = detect_vehicle(image)
        print(f"\n=== Détection véhicule ===")
        print(f"Marque      : {result['brand']}")
        print(f"Confiance   : {result['confidence']:.0%}")
        print(f"Top 3       : {result['all_brands']}")
    else:
        print("Usage : python vehicle_detector.py <image.jpg>")
        print("\nModèle chargé avec succès ✅")
        print(f"Classes disponibles : {len(CAR_CLASSES)} marques")
        print("Marques :", ", ".join(list(set(CAR_CLASSES.values()))[:10]), "...")