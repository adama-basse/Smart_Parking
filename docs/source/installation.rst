Installation
============

Prérequis
---------

- Python 3.9+
- Git
- 4 GB RAM minimum (8 GB recommandé)

Étapes
------

1. Cloner le repository :

.. code-block:: bash

   git clone https://github.com/adama-basse/Smart_Parking.git
   cd Smart_Parking

2. Créer l'environnement virtuel :

.. code-block:: bash

   python -m venv .venv
   .venv\Scripts\activate   # Windows
   source .venv/bin/activate # Linux/Mac

3. Installer les dépendances :

.. code-block:: bash

   pip install -r requirements.txt

4. Ajouter les modèles YOLO dans ``models/`` :

.. code-block:: text

   models/
   ├── best_v5.pt       # YOLOv5 — détection plaque
   ├── best_v8.pt       # YOLOv8 — détection plaque
   └── best_marques.pt  # YOLOv8 — détection marque (50 classes)

Lancement
---------

.. code-block:: bash

   streamlit run app/streamlit_app.py

L'interface est accessible sur ``http://localhost:8501``
et fonctionne **entièrement hors connexion**.