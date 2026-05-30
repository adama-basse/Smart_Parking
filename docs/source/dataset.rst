Dataset
=======

Le dataset utilisé pour l'entraînement YOLO est issu de la fusion
de deux sources publiques disponibles sur **Roboflow**.

Source 1 — Plaques Marocaines (Lamya Alhyane)
----------------------------------------------

.. list-table::
   :widths: 40 60
   :header-rows: 0

   * - Workspace
     - ``lamya-alhyane-2cfvf``
   * - Projet
     - ``license-plate-2dqgy``
   * - Images
     - **3 078 images** de plaques marocaines
   * - Type
     - Face ET dos des véhicules
   * - mAP@50 auteur
     - 99.5% sur son modèle

Source 2 — Plaques Internationales (Haeun Kim)
-----------------------------------------------

.. list-table::
   :widths: 40 60
   :header-rows: 0

   * - Workspace
     - ``haeun-kim-ri91b``
   * - Projet
     - ``license-plate-detection-wienp``
   * - Images
     - **2 156 images** internationales
   * - Type
     - Plaques européennes variées
   * - Popularité
     - 476 téléchargements

Source 3 — Marques de Véhicules (Car Brand Detection)
------------------------------------------------------

.. list-table::
   :widths: 40 60
   :header-rows: 0

   * - Images
     - **4 600 images** annotées
   * - Classes
     - 100 (50 marques × car + logo)
   * - Marques incluses
     - Renault, Dacia, Toyota, BMW, Ford, Peugeot...
   * - mAP@50
     - 78.4%
   * - Précision
     - 80.9%
   * - Recall
     - 68.7% — 50 epochs GPU T4

Fusion & Data Augmentation
---------------------------

Les deux premiers datasets (plaques) ont été fusionnés via un script
Python qui normalise toutes les classes en une seule : ``LicensePlate``.

.. code-block:: python

   # Fusion des datasets
   fusionner_datasets(dataset_maroc.location,
                      dataset_eu.location,
                      output_path='/content/dataset_final')

.. list-table::
   :widths: 30 20 20 20 10
   :header-rows: 1

   * - Métrique
     - Total
     - Train
     - Validation
     - Test
   * - Images
     - **8 276**
     - 6 716
     - 1 051
     - 509
   * - Classe
     - 1
     - 1
     - 1
     - 1
     - LicensePlate

.. note::

   Toutes les annotations ont été normalisées en classe ``0 : LicensePlate``
   pour les deux sources. Le dataset de marques est utilisé séparément
   pour entraîner le modèle de détection de marques.

Téléchargement via Roboflow
----------------------------

.. code-block:: python

   from roboflow import Roboflow

   rf = Roboflow(api_key="VOTRE_CLE")

   # Dataset marocain
   project1 = rf.workspace("lamya-alhyane-2cfvf")\
                 .project("license-plate-2dqgy")
   dataset_maroc = project1.version(1).download("yolov8")

   # Dataset international
   project2 = rf.workspace("haeun-kim-ri91b")\
                 .project("license-plate-detection-wienp")
   dataset_eu = project2.version(1).download("yolov8")