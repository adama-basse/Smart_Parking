Résultats
=========

Détection YOLO
--------------

.. list-table::
   :widths: 40 30 30
   :header-rows: 1

   * - Métrique
     - YOLOv5
     - YOLOv8
   * - mAP\@50
     - **94.1%** ✅
     - 93.8%
   * - Précision
     - **96.8%**
     - 96.9%
   * - Recall
     - **90.1%**
     - 90.6%
   * - Modèle retenu
     - ✅ Oui
     - NonDétection YOLO — Comparaison
-----------------------------

Deux modèles entraînés sur le même dataset (8 276 images).

.. list-table::
   :widths: 40 30 30
   :header-rows: 1

   * - Métrique
     - YOLOv8
     - YOLOv5
   * - mAP@50
     - 93.8%
     - 94.1%
   * - mAP@50-95
     - 75.2%
     - 75.3%
   * - Précision
     - **96.9%** ✅
     - 96.8%
   * - Recall
     - **90.6%** ✅
     - 90.1%

.. note::

   **Modèle retenu : YOLOv8** — mAP@50 : 93.8% • Précision : 96.9% • Recall : 90.6%

   YOLOv8 est retenu car il offre une meilleure précision et recall,
   et bénéficie d'une architecture plus récente avec une meilleure
   compatibilité avec les outils de déploiement.

OCR sur dataset
---------------

Dataset : **1 171 crops** générés par YOLOv5 sur 8 276 images.

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - Métrique
     - Valeur
   * - Plaques lisibles
     - **911 (77.8%)**
   * - dont marocaines
     - 41 (3.5%)
   * - dont européennes
     - 128 (11.0%)
   * - dont vanity/autres
     - 742 (63.4%)
   * - Aucun texte lu
     - 207 (17.7%)
   * - Temps moyen/image
     - 4.28s (CPU only)

Robustesse — Conditions difficiles
------------------------------------

.. list-table::
   :widths: 25 25 25 25
   :header-rows: 1

   * - Condition
     - Taux lecture
     - Confiance
     - Commentaire
   * - Original
     - 60%
     - 0.20
     - Référence
   * - Flou
     - 70%
     - 0.29
     - CLAHE compense
   * - Nuit
     - 70%
     - 0.31
     - Bon résultat
   * - Bruit
     - 60%
     - 0.30
     - Stable
   * - Incliné 15°
     - **90%**
     - 0.21
     - Meilleur cas

Axe Low-cost
------------

- Temps actuel (EasyOCR CPU) : **4.28s/image**
- Estimation avec YOLOv8n : **~2.5s/image**
- Estimation optimisé INT8 : **~1.4s/image**
- Matériel : **Raspberry Pi 4 — ~50€**