Introduction
============

Smart Parking est un système de vision par ordinateur qui détecte automatiquement
les plaques d'immatriculation à l'entrée et à la sortie d'un parking, lit le texte
via OCR, identifie la marque du véhicule et gère les sessions de stationnement.

Objectifs
---------

- Détecter les plaques avec **YOLOv8** (mAP\@50 = 93.8%)
- Lire le texte avec **EasyOCR** (double passe EN + AR)
- Identifier la marque du véhicule parmi **50 marques**
- Gérer les entrées/sorties via une interface **Streamlit** hors connexion
- Stocker les sessions dans **SQLite** avec calcul automatique du prix

Technologies
------------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Composant
     - Technologie
   * - Détection plaque
     - YOLOv5 / YOLOv8
   * - Détection marque
     - YOLOv8 (50 marques, mAP\@50 = 78.4%)
   * - OCR
     - EasyOCR (EN + AR double passe)
   * - Post-traitement
     - Python + Regex
   * - Base de données
     - SQLite
   * - Interface
     - Streamlit (hors connexion)
   * - Low-cost
     - Compatible Raspberry Pi 4