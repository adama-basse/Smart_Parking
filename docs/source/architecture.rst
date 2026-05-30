Architecture du Système
=======================

Pipeline complet
----------------

.. code-block:: text

   Photo voiture
        ↓
   YOLOv8 — Détection et crop de la plaque
        ↓
   Preprocessing — Resize x2, filtre bilatéral, CLAHE
        ↓
   EasyOCR (double passe EN + AR) — Lecture OCR
        ↓
   clean_text.py — Nettoyage et validation
        ↓
   vehicle_detector.py — Détection marque (YOLOv8)
        ↓
   SQLite — Enregistrement session
        ↓
   Streamlit — Affichage résultat + reçu PDF

Structure des fichiers
----------------------

.. code-block:: text

   smart-parking-ocr/
   ├── app/
   │   └── streamlit_app.py       # Interface principale
   ├── scripts/
   │   ├── preprocess.py          # Preprocessing images
   │   ├── clean_text.py          # Post-traitement OCR
   │   ├── yolo_crop.py           # Détection YOLO
   │   ├── batch_ocr.py           # Pipeline batch
   │   ├── compare_ocr.py         # Métriques et graphiques
   │   ├── vehicle_detector.py    # Détection marque
   │   └── augment_test.py        # Test conditions difficiles
   ├── models/                    # Modèles YOLO (non inclus)
   ├── data/raw/                  # Images de test
   ├── results/figures/           # Graphiques générés
   ├── requirements.txt
   └── README.md