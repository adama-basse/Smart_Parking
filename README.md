# 🅿️ Smart Parking — Reconnaissance Automatique de Plaques

Système de smart parking utilisant la vision par ordinateur pour détecter, 
lire et gérer automatiquement les plaques d'immatriculation.

**Projet Computer Vision — 2025/2026**

---

## 🏗️ Architecture du système
Image voiture → YOLOv5 (détection plaque) → EasyOCR (lecture) → SQLite (gestion) → Streamlit (interface)

## 🚀 Technologies utilisées

| Composant | Technologie |
|---|---|
| Détection plaque | YOLOv5 / YOLOv8 |
| Détection marque | YOLOv8 (50 marques) |
| OCR | EasyOCR (EN + AR) |
| Post-traitement | Python + Regex |
| Base de données | SQLite |
| Interface | Streamlit |
| Low-cost | Compatible Raspberry Pi 4 |

## 📊 Résultats

- **Dataset** : 8 276 images (Maroc + International)
- **Détection YOLO** : mAP@50 = 94.1% (YOLOv5 retenu)
- **Taux de lecture OCR** : 77.8% sur 1 171 crops réels
- **Temps moyen** : 4.28s/image (CPU only)

## 🗂️ Structure du projet
smart-parking-ocr/
├── app/
│   └── streamlit_app.py      # Interface principale
├── scripts/
│   ├── preprocess.py         # Preprocessing images
│   ├── clean_text.py         # Post-traitement OCR
│   ├── yolo_crop.py          # Détection et crop YOLO
│   ├── batch_ocr.py          # Pipeline batch complet
│   ├── compare_ocr.py        # Métriques et graphiques
│   ├── vehicle_detector.py   # Détection marque véhicule
│   └── augment_test.py       # Test conditions difficiles
├── data/
│   └── raw/                  # Images de test
├── results/
│   └── figures/              # Graphiques générés
├── models/                   # Modèles YOLO (non inclus)
├── requirements.txt
└── README.md

## ⚙️ Installation

```bash
# Cloner le repo
git clone https://github.com/adama-basse/Smart_Parking.git
cd smart-parking-ocr

# Créer l'environnement virtuel
python -m venv .venv
.venv\Scripts\activate  # Windows

# Installer les dépendances
pip install -r requirements.txt

# Ajouter les modèles (non inclus dans le repo)
# Copier best_v5.pt, best_v8.pt, best_marques.pt dans models/
```

## ▶️ Lancement

```bash
# Interface Streamlit
streamlit run app/streamlit_app.py

# Pipeline batch OCR
python scripts/batch_ocr.py

# Graphiques comparatifs
python scripts/compare_ocr.py
```

## 👥 Équipe

| Membre | Partie |
|---|---|
| [Djibril Sall] | Détection YOLO — Dataset — Entraînement |
| [Adama Basse] | OCR — Post-traitement — Interface — Smart Parking |

---
*Projet Computer Vision — 2025/2026*