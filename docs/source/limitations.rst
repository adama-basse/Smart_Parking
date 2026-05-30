Limitations & Perspectives
===========================

Limitations actuelles
---------------------

.. warning::

   Ces limitations ont été identifiées lors des tests sur le dataset réel.

**1. Lecture des plaques marocaines**

La lettre arabe centrale des plaques marocaines est parfois mal lue.
EasyOCR peut retourner ``48175|0`` au lieu de ``48175|B|15``.

Cause : le séparateur ``|`` n'est pas reconstruit correctement
lorsque la lettre arabe est absente ou mal détectée dans le crop.

.. code-block:: text

   Plaque réelle  : 48175 | ب | 15
   OCR lit        : 48175 | 0        ← lettre arabe perdue
   Attendu        : 48175 | B | 15

**2. Temps de traitement**

Le pipeline tourne en **CPU only** (sans GPU) pour être compatible
Raspberry Pi. Temps moyen : **4.28s/image**, ce qui limite le débit
à environ 14 véhicules/minute.

**3. Plaques européennes personnalisées (vanity plates)**

Le dataset international contient de nombreuses plaques américaines
personnalisées sans format standard (ex: ``FRESPCH``, ``MRS BANX``).
Ces plaques sont classées "autre" — elles sont lues mais non validées
selon un format strict.

**4. Dépendance à la qualité du crop YOLO**

Si YOLO rate la détection ou produit un crop de mauvaise qualité
(flou extrême, angle trop prononcé), l'OCR ne peut pas compenser.

Perspectives de développement
------------------------------

.. note::

   Ces axes d'amélioration permettraient d'augmenter significativement
   les performances du système.

**1. Fine-tuning EasyOCR sur plaques marocaines**

Entraîner EasyOCR spécifiquement sur un dataset de plaques marocaines
annotées permettrait d'améliorer le taux marocain de 3.5% → 30%+.

**2. Modèle allégé YOLOv8n**

Remplacer le modèle de détection par YOLOv8 nano réduirait
le temps de traitement de 4.28s → ~1.5s/image,
rendant le système utilisable en temps réel sur Raspberry Pi 4.

**3. Déploiement embarqué**

.. code-block:: text

   Matériel cible  : Raspberry Pi 4 (4 cœurs ARM, 4GB RAM)
   Coût estimé     : ~50€
   Temps estimé    : ~2.5s/image (avec optimisations)
   Débit estimé    : ~24 véhicules/minute

**4. Caméra temps réel**

Remplacer l'upload photo par un flux caméra en temps réel
à l'entrée et à la sortie du parking pour une automatisation totale.

**5. Base de données distante**

Migrer de SQLite vers PostgreSQL pour permettre la gestion
multi-sites et l'accès distant au dashboard.