import re


# =============================================
# PATTERNS DE VALIDATION
# =============================================

# Marocaine stricte : 27549|A|3
PATTERN_MAROC = re.compile(r'^\d{5,5}\|[A-Z]{1,2}\|\d{1,2}$')

# Européenne standard : AB-123-CD / TL 03 DWA

PATTERN_EU = re.compile(
    r'^(?=.*[A-Z])(?=.*[0-9])'
    r'[A-Z0-9]{1,4}[-\s·]?[A-Z0-9]{1,4}[-\s·]?[A-Z0-9]{1,4}$'
)
# Couvre les vanity plates, américaines, etc.
# Accepte 2+ caractères alphanumériques (pas juste 3+)
PATTERN_READABLE = re.compile(r'^[A-Z0-9]{2,}([-\s]?[A-Z0-9]*)*$')


# =============================================
# FONCTIONS DE NETTOYAGE
# =============================================

def remove_arabic(text):
    """Supprime les caractères arabes lus par EasyOCR"""
    return re.sub(r'[\u0600-\u06FF\u0750-\u077F]+', '', text)
# Dictionnaire des lettres arabes de plaques marocaines
ARABIC_TO_LATIN = {
    'أ': 'A', 'ا': 'A',
    'ب': 'B',
    'ت': 'T',   # ← déjà présent mais vérifier
    'ط': 'T',   # ← Ta emphatique, souvent confondu avec ت
    'و': 'W',
    'ه': 'H',
    'ي': 'Y',
    'د': 'D',
    'ر': 'R',
    'س': 'S',
    'ن': 'N',
    'إ': 'A',
    'آ': 'A',
    'ڭ': 'G',
    'ق': 'Q',
    'م': 'M',
    'ف': 'F',
    'ع': 'A',
    'ح': 'H',
    'ج': 'J',
    'ز': 'Z',
    'ك': 'K',
    'ل': 'L',
    'ش': 'SH',
    'ص': 'S',   # ← Sa emphatique
    'ض': 'D',   # ← Da emphatique
    'ظ': 'Z',   # ← Za emphatique
    'غ': 'GH',  # ← Ghain
    'ث': 'TH',  # ← Tha
    'خ': 'KH',  # ← Kha
}

def convert_arabic_letters(text):
    """
    Convertit les lettres arabes de plaque marocaine en latin.
    Ex: '42675 | ب | 40' → '42675 | B | 40'
    Supprime les autres caractères arabes (mots parasites).
    """
    result = ''
    for char in text:
        if char in ARABIC_TO_LATIN:
            result += ARABIC_TO_LATIN[char]
        elif '\u0600' <= char <= '\u06FF':
            # Caractère arabe non reconnu → on l'ignore
            continue
        else:
            result += char
    return result

def remove_unwanted_words(text):
    """Supprime les mots parasites fréquents sur les plaques"""
    unwanted = ["ROYAUME", "MAROC", "DU", "QUEBEC", "SOUVIENS",
                "ONTARIO", "ALBERTA", "CANADA", "FRANCE", "JE", "ME"]
    for word in unwanted:
        text = text.replace(word, ' ')
    return text

def normalize_separator_maroc(text):
    """
    Règles de reconstruction du format marocain CHIFFRES|LETTRE|CHIFFRES
    
    Observation clé sur les plaques marocaines :
    - Le séparateur | est lu comme I, i, 1, l, /
    - La lettre de plaque (arabe convertie) est souvent lue comme L, l, I
    - Donc : 5chiffres + (I/1/l) + (L/A/B/...) + chiffres → reconstruction
    """
    # Cas 1 : séparateur explicite déjà là mais lettre mal lue
    # "27549|IL|3" ou "27549|L|3" → "27549|A|3"
    text = re.sub(r'^(\d{3,6})\|[IL]?\|(\d{1,2})$', r'\1|A|\2', text)
    text = re.sub(r'^(\d{3,6})\|IL(\d{1,2})$', r'\1|A|\2', text)
    text = re.sub(r'^(\d{3,6})\|L\|(\d{1,2})$', r'\1|A|\2', text)

    # Cas 2 : PATTERN PRINCIPAL — chiffres + faux_sep + lettre + chiffres
    # "27549IL3" → sep=I, lettre=L → "27549|A|3"
    # "27549I B3" → sep=I, lettre=B → "27549|B|3"
    def reconstruct(m):
        chiffres_debut = m.group(1)
        lettre         = m.group(2)
        chiffres_fin   = m.group(3)
        # Si la lettre est L ou I seul → c'est un A mal lu
        if lettre in ('L', 'I'):
            lettre = 'A'
        return f"{chiffres_debut}|{lettre}|{chiffres_fin}"

    # Format : chiffres + [I/i/1/l/!] + [lettre] + chiffres
    text = re.sub(
        r'^(\d{3,6})[Ii1l!]\s*([A-Z])\s*(\d{1,2})$',
        reconstruct, text
    )

    # Cas 3 : chiffres + espace + 1 + espace + chiffres
    # "37947 1 16" → "37947|A|16" (le 1 isolé = séparateur + lettre A)
    text = re.sub(
        r'^(\d{3,6})\s+[1Ii]\s+(\d{1,2})$',
        r'\1|A|\2', text
    )

    # Cas 4 : espaces simples avec vraie lettre : "37947 B 40"
    text = re.sub(r'(\d{3,6})\s+([A-Z])\s+(\d{1,2})', reconstruct, text)

    # Cas 5 : tout collé avec vraie lettre : "37947B40"
    text = re.sub(r'^(\d{3,6})([A-Z]{1,2})(\d{1,2})$', reconstruct, text)

    # Cas 6 : "379471 16" → chiffres collés + espace + chiffres finaux
    # Le dernier chiffre du premier groupe = séparateur mal lu
    text = re.sub(
        r'^(\d{3,5})[1Ii]\s+(\d{1,2})$',
        r'\1|A|\2', text
    )
    text = re.sub(
        r'^(\d{4,6})\s*L\s*(\d{1,2})$',
        lambda m: f"{m.group(1)}|B|{m.group(2)}",
        text
    )
    text = re.sub(
        r'^(\d{4,6})L(\d{1,2})$',
        lambda m: f"{m.group(1)}|B|{m.group(2)}",
        text
    )

    # Normalise espaces autour de |
    text = re.sub(r'\s*\|\s*', '|', text)
    return text
    

def fix_digits(text):
    parts = text.split('|')
    fixed = []
    for i, part in enumerate(parts):
        if i == 0 or i == 2:  # parties numériques
            part = part.replace('O', '0').replace('o', '0')
            part = part.replace('l', '1')
            if len(part) > 1:
                part = part.replace('I', '1')
        elif i == 1:  # partie lettre
            # I seul comme lettre de plaque → probablement A
            # (aucune plaque marocaine n'utilise I comme lettre)
            if part == 'I':
                part = 'A'
            # L seul → probablement B ou A mal lu
            if part == 'L':
                part = 'A'
        fixed.append(part)
    return '|'.join(fixed)

def fix_digits_eu(text):
    """
    Pour les plaques européennes : correction plus légère,
    on ne touche pas à l'ordre des caractères
    """
    # Seulement les 0/O évidents dans un contexte numérique
    text = re.sub(r'(?<=[A-Z])0(?=[A-Z])', 'O', text)  # 0 entre lettres → O
    return text

def detect_plate_type(text):
    # Cas 1 : déjà reconstruit avec |
    if '|' in text:
        return 'maroc'
    # Cas 2 : pattern chiffres + lettre isolée + chiffres (avec espaces)
    # Ex: "37947 I 16" ou "37947 l 16" ou "37947 1 16"
    if re.search(r'\d{5}\s+[A-Z0-9]\s+\d{1,2}', text):
        return 'maroc'
    # Cas 3 : beaucoup de chiffres groupés
    if re.search(r'\d{4,}', text):
        return 'maroc'
    return 'foreign'



def validate(text):
    """
    Validation en 3 niveaux.
    Retourne (is_valid, plate_type)
    """
    if not text:
        return False, 'inconnue'

    # Niveau 1 — Marocaine stricte
    if PATTERN_MAROC.match(text):
        return True, 'marocaine'

    # Niveau 2 — Européenne semi-stricte
    if PATTERN_EU.match(text):
        return True, 'européenne'

    # Niveau 3 — Lisible : au moins 3 caractères alphanumériques
    # On compte juste les lettres et chiffres, peu importe les espaces
    alphanum = re.sub(r'[^A-Z0-9]', '', text)
    if len(alphanum) >= 3:
        return True, 'autre'

    return False, 'inconnue'
# =============================================
# PIPELINE PRINCIPAL
# =============================================

def clean_plate_text(text, easyocr_confidence=None, confidence_threshold=0.05):
    """
    Nettoie et valide un texte de plaque.

    Args:
        text                 : texte brut retourné par OCR
        easyocr_confidence   : score EasyOCR entre 0 et 1 (optionnel)
        confidence_threshold : seuil minimum de confiance

    Returns:
        dict : cleaned, valid, plate_type, confidence
    """
    if easyocr_confidence is not None and easyocr_confidence < confidence_threshold:
        return {'cleaned': '', 'valid': False, 'plate_type': 'inconnue',
                'confidence': easyocr_confidence}

    # 1. Convertit lettres arabes utiles
    text = convert_arabic_letters(text)
    # 2. Majuscules
    text = text.upper()
    # 3. Supprime mots parasites
    text = remove_unwanted_words(text)
    # 4. Garde caractères valides
    text = re.sub(r'[^A-Z0-9|\-\s]', '', text)
    # 5. Nettoie espaces multiples
    text = re.sub(r'\s+', ' ', text).strip()

    
   

    plate_type = detect_plate_type(text)
    # Normalise les séparateurs européens (·, ., -, espace)
    if plate_type == 'foreign':
        text = re.sub(r'[·•\.]', '-', text)   # point médian → tiret
        text = re.sub(r'-{2,}', '-', text)     # double tiret → simple
        text = re.sub(r'\s+', ' ', text).strip()

    if plate_type == 'maroc':
        text = normalize_separator_maroc(text)
        text = fix_digits(text)
    else:
        text = fix_digits_eu(text)

    
    is_valid, detected_type = validate(text)
    text = text.strip()
    return {
        'cleaned'    : text,
        'valid'      : is_valid,
        'plate_type' : detected_type,
        'confidence' : easyocr_confidence
    }
# =============================================
# TEST
# =============================================

if __name__ == "__main__":
    
    test_maroc = [
    ("37947 I 16",  0.8,  "Séparateur I avec espaces"),
    ("37947I16",    0.8,  "Tout collé"),
    ("37947 l 16",  0.8,  "Séparateur l minuscule"),
    ("379471 16",   0.8,  "1 au lieu de I"),
    ("37947/16",    0.8,  "Slash comme séparateur"),
    ("37947 1 16",  0.8,  "Chiffre 1 comme séparateur"),
]
    print("\n=== TEST PLAQUES MAROCAINES ===")
    for text, conf, desc in test_maroc:
        r = clean_plate_text(text, easyocr_confidence=conf)
        status = "✅" if r['valid'] else "❌"
        print(f"{desc:<35} → {r['cleaned']:<15} {r['plate_type']:<12} {status}")

    test_reels = [
    ("27549IL3",   0.5, "27549|IL3 → doit donner 27549|L|3 ou 27549|A|3"),
    ("27549 I L3", 0.5, "espaces variés"),
    ("42675B40",   0.8, "arabe déjà converti collé"),
    ("37947I16",   0.8, "I seul comme séparateur"),
]
    print("\n=== TEST CAS RÉELS ===")
    for text, conf, desc in test_reels:
        r = clean_plate_text(text, easyocr_confidence=conf)
        status = "✅" if r['valid'] else "❌"
        print(f"{desc[:40]:<42} → '{r['cleaned']:<15}' {status}")