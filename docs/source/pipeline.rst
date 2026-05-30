Pipeline OCR
============

Preprocessing
-------------

4 étapes appliquées sur chaque crop de plaque :

.. code-block:: python

   def preprocess_optimized(image):
       gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
       gray  = cv2.resize(gray, None, fx=2, fy=2,
                          interpolation=cv2.INTER_CUBIC)
       gray  = cv2.bilateralFilter(gray, 9, 75, 75)
       clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
       return clahe.apply(gray)

Double passe EasyOCR
--------------------

Pour gérer les plaques marocaines (lettres arabes) et européennes :

.. code-block:: python

   reader_en = easyocr.Reader(['en'], gpu=False)  # Chiffres latins
   reader_ar = easyocr.Reader(['ar'], gpu=False)  # Lettre arabe marocaine

   results = reader_en.readtext(processed) + reader_ar.readtext(processed)
   # Fusion et tri gauche → droite par position X

Post-traitement — clean_text.py
--------------------------------

Validation en 3 niveaux :

.. list-table::
   :widths: 25 40 35
   :header-rows: 1

   * - Type
     - Format
     - Exemple
   * - Marocaine
     - ``\d{5}\|[A-Z]{1,2}\|\d{1,2}``
     - ``48175|B|15``
   * - Européenne
     - Lettres + chiffres + séparateurs
     - ``FM-915-LM``
   * - Autre (vanity)
     - ≥ 3 caractères alphanumériques
     - ``FRESPCH``

Détection marque
----------------

.. code-block:: python

   from vehicle_detector import detect_vehicle, match_vehicle

   vehicle = detect_vehicle(image)
   # → {'brand': 'Ford', 'confidence': 0.87, 'all_brands': [...]}

   matched = match_vehicle(plate_result, vehicle, db_conn)
   # → {'status': 'certain', 'message': 'Identification certaine — Ford'}