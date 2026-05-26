import streamlit as st
import sqlite3
import cv2
import easyocr
import numpy as np
from PIL import Image
from pathlib import Path
from datetime import datetime
import time
import sys
import re

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from preprocess import preprocess_optimized
from clean_text import clean_plate_text
from yolo_crop import detect_and_crop_array
from vehicle_detector import detect_vehicle, match_vehicle

st.set_page_config(page_title="Smart Parking", page_icon="🅿️",
                   layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    .stApp { background-color: #0A1628; color: white; }
    .metric-card { background: #0D2137; border: 1px solid #1A56DB;
        border-radius: 10px; padding: 20px; text-align: center; }
    .plate-display { background: #0D2137; border: 2px solid #06B6D4;
        border-radius: 8px; padding: 15px; text-align: center;
        font-size: 2em; font-weight: bold; color: #06B6D4;
        font-family: monospace; letter-spacing: 4px; }
    .valid-plate   { border-color: #10B981; color: #10B981; }
    .invalid-plate { border-color: #F59E0B; color: #F59E0B; }
    .alert-box { background: #7F1D1D; border: 1px solid #EF4444;
        border-radius: 8px; padding: 10px; color: #FCA5A5; }
    .success-box { background: #064E3B; border: 1px solid #10B981;
        border-radius: 8px; padding: 10px; color: #6EE7B7; }
    div[data-testid="stTabs"] button { color: white !important; font-size: 16px; }
</style>
""", unsafe_allow_html=True)

DB_PATH  = Path("data/parking.db")
TARIF    = 0.005
CAPACITY = 50
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

@st.cache_resource
def init_db():
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.execute("""CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT, plate TEXT NOT NULL,
            plate_type TEXT, brand TEXT DEFAULT 'Non détecté',
            entry_time TEXT, exit_time TEXT, duration_min REAL,
            price REAL, spot INTEGER, confidence REAL)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS known_vehicles (
            plate TEXT PRIMARY KEY, plate_type TEXT, brand TEXT,
            visit_count INTEGER DEFAULT 1, last_seen TEXT)""")
    conn.commit()
    return conn

@st.cache_resource
def load_ocr():
    # Reader anglais pour plaques EU/vanity
    reader_en = easyocr.Reader(['en'], gpu=False, verbose=False)
    # Reader arabe pour lettres marocaines
    reader_ar = easyocr.Reader(['ar'], gpu=False, verbose=False)
    return reader_en, reader_ar

conn = init_db()
reader_en, reader_ar = load_ocr()

if 'last_exit' not in st.session_state:
    st.session_state.last_exit = None

# =============================================
# UTILITAIRES DB
# =============================================
def get_parked_vehicles():
    cur = conn.execute("SELECT plate, plate_type, brand, entry_time, spot, confidence "
                       "FROM sessions WHERE exit_time IS NULL ORDER BY entry_time DESC")
    return cur.fetchall()

def get_history(limit=20):
    cur = conn.execute("SELECT plate, plate_type, entry_time, exit_time, duration_min, price "
                       "FROM sessions WHERE exit_time IS NOT NULL "
                       "ORDER BY exit_time DESC LIMIT ?", (limit,))
    return cur.fetchall()

def get_next_spot():
    used = {r[0] for r in conn.execute(
        "SELECT spot FROM sessions WHERE exit_time IS NULL").fetchall()}
    for i in range(1, CAPACITY + 1):
        if i not in used:
            return i
    return None

def do_exit(session_id, entry_time_str, plate):
    entry_dt = datetime.fromisoformat(entry_time_str)
    now      = datetime.now()
    duration = (now - entry_dt).total_seconds()
    price    = duration * TARIF
    conn.execute("UPDATE sessions SET exit_time=?, duration_min=?, price=? WHERE id=?",
                 (now.isoformat(), duration/60, price, session_id))
    conn.commit()
    return entry_dt, now, duration, price

# =============================================
# OCR DOUBLE PASSE — CLÉ DE L'AMÉLIORATION
# =============================================
def box_area(r):
    pts = r[0]
    w = max(pts[1][0], pts[2][0]) - min(pts[0][0], pts[3][0])
    h = max(pts[2][1], pts[3][1]) - min(pts[0][1], pts[1][1])
    return abs(w * h)

def read_plate(image_np):
    """
    Double passe OCR :
    - Passe 1 (anglais) : lit les chiffres et lettres latines
    - Passe 2 (arabe)   : lit la lettre arabe de la plaque marocaine
    Combine les deux pour reconstruire le texte complet gauche→droite.
    """
    processed = preprocess_optimized(image_np)

    # ── Passe 1 : Anglais ──────────────────────────────────
    results_en = reader_en.readtext(processed)
    
    # ── Passe 2 : Arabe ────────────────────────────────────
    results_ar = reader_ar.readtext(processed)

    # Fusionner tous les résultats
    all_results = results_en + results_ar
    if not all_results:
        return "", 0.0

    # Filtrer par surface > 20% de la plus grande boîte
    if not all_results:
        return "", 0.0
    max_area = max(box_area(r) for r in all_results)
    main = [r for r in all_results if box_area(r) > max_area * 0.20 and r[2] > 0.05]

    if not main:
        return "", 0.0

    # Trier gauche → droite par position X
    main = sorted(main, key=lambda r: r[0][0][0])

    # Déduplique les boîtes qui se chevauchent (garder la plus confiante)
    deduped = []
    for r in main:
        cx = (r[0][0][0] + r[0][2][0]) / 2
        # Vérifie si une boîte similaire existe déjà
        overlap = False
        for d in deduped:
            dcx = (d[0][0][0] + d[0][2][0]) / 2
            if abs(cx - dcx) < 50:  # même zone horizontale
                overlap = True
                break
        if not overlap:
            deduped.append(r)

    raw  = ' '.join([r[1] for r in deduped])
    conf = sum([r[2] for r in deduped]) / len(deduped)
    return raw, conf

def process_image(uploaded_file):
    img_pil = Image.open(uploaded_file).convert("RGB")
    img_np  = np.array(img_pil)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    crop, annotated = detect_and_crop_array(img_bgr, model_path="models/best_v8.pt")
    if crop is not None:
        raw, conf = read_plate(crop)
    else:
        raw, conf = "", 0.0
    cleaned = clean_plate_text(raw, easyocr_confidence=conf)
    vehicle = detect_vehicle(img_bgr)
    matched = match_vehicle(cleaned, vehicle, conn)
    return img_pil, annotated, crop, raw, cleaned, conf, vehicle, matched

def generate_receipt(plate, brand, entry_dt, exit_dt, duration_sec, price, spot):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    import io
    buffer = io.BytesIO()
    doc    = SimpleDocTemplate(buffer, pagesize=A4, topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    story  = []
    title_style = ParagraphStyle('title', parent=styles['Title'], fontSize=22,
                                 textColor=colors.HexColor('#06B6D4'), spaceAfter=10)
    sub_style   = ParagraphStyle('sub', parent=styles['Normal'], fontSize=11,
                                 textColor=colors.grey, spaceAfter=20)
    story.append(Paragraph("SMART PARKING", title_style))
    story.append(Paragraph("Système de Reconnaissance Automatique de Plaques", sub_style))
    story.append(Spacer(1, 0.5*cm))
    story.append(Table([['']], colWidths=[17*cm],
                       style=[('LINEBELOW', (0,0), (-1,-1), 2, colors.HexColor('#06B6D4'))]))
    story.append(Spacer(1, 0.5*cm))
    data = [["REÇU DE STATIONNEMENT", ""],
            ["Plaque", plate], ["Marque", brand], ["Place", f"N° {spot}"],
            ["Entrée", entry_dt.strftime('%d/%m/%Y à %H:%M:%S')],
            ["Sortie", exit_dt.strftime('%d/%m/%Y à %H:%M:%S')],
            ["Durée", f"{duration_sec/60:.0f} minutes"],
            ["Tarif", f"{TARIF * 3600:.1f} MAD / heure"],
            ["TOTAL À PAYER", f"{price:.2f} MAD"]]
    table = Table(data, colWidths=[7*cm, 10*cm])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0A1628')),
        ('TEXTCOLOR',  (0,0), (-1,0), colors.HexColor('#06B6D4')),
        ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'), ('FONTSIZE', (0,0), (-1,0), 13),
        ('SPAN',       (0,0), (-1,0)), ('ALIGN', (0,0), (-1,0), 'CENTER'),
        ('FONTNAME',   (0,1), (0,-2), 'Helvetica-Bold'),
        ('FONTNAME',   (1,1), (1,-2), 'Helvetica'), ('FONTSIZE', (0,1), (-1,-2), 11),
        ('TEXTCOLOR',  (0,1), (0,-2), colors.HexColor('#1A56DB')),
        ('ROWBACKGROUNDS', (0,1), (-1,-2), [colors.HexColor('#F8FAFC'), colors.white]),
        ('BACKGROUND', (0,-1), (-1,-1), colors.HexColor('#F59E0B')),
        ('TEXTCOLOR',  (0,-1), (-1,-1), colors.white),
        ('FONTNAME',   (0,-1), (-1,-1), 'Helvetica-Bold'), ('FONTSIZE', (0,-1), (-1,-1), 14),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
        ('TOPPADDING', (0,0), (-1,-1), 8), ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING', (0,0), (-1,-1), 12)]))
    story.append(table)
    story.append(Spacer(1, 1*cm))
    footer = ParagraphStyle('footer', parent=styles['Normal'], fontSize=9,
                            textColor=colors.grey, alignment=1)
    story.append(Paragraph(
        f"Reçu généré le {exit_dt.strftime('%d/%m/%Y à %H:%M:%S')} • Smart Parking System", footer))
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

# =============================================
# HEADER
# =============================================
st.markdown("""
<div style='background:#0D2137; border-bottom:3px solid #06B6D4;
            padding:20px; margin-bottom:20px; border-radius:8px;'>
    <h1 style='color:white; margin:0; font-size:2.2em;'>🅿️ Smart Parking System</h1>
    <p style='color:#06B6D4; margin:5px 0 0 0;'>
        Système de Reconnaissance Automatique de Plaques • YOLOv5 + EasyOCR
    </p>
</div>""", unsafe_allow_html=True)

# =============================================
# MÉTRIQUES — recalculées à chaque rerun
# =============================================
parked       = get_parked_vehicles()
nb_parked    = len(parked)
nb_free      = CAPACITY - nb_parked
occupation   = nb_parked / CAPACITY * 100
history_rows = get_history(limit=1000)
revenus      = sum(r[5] for r in history_rows if r[5])

col1, col2, col3, col4 = st.columns(4)
with col1:
    color = "#EF4444" if occupation > 80 else "#10B981"
    st.markdown(f"""<div class='metric-card'>
        <div style='font-size:2.5em;color:{color};font-weight:bold;'>{nb_parked}/{CAPACITY}</div>
        <div style='color:#94A3B8;'>Places occupées</div>
        <div style='color:{color};font-size:0.9em;'>{occupation:.0f}% d'occupation</div>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""<div class='metric-card'>
        <div style='font-size:2.5em;color:#06B6D4;font-weight:bold;'>{nb_free}</div>
        <div style='color:#94A3B8;'>Places libres</div>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""<div class='metric-card'>
        <div style='font-size:2.5em;color:#F59E0B;font-weight:bold;'>{revenus:.1f}</div>
        <div style='color:#94A3B8;'>MAD générés</div>
    </div>""", unsafe_allow_html=True)
with col4:
    st.markdown(f"""<div class='metric-card'>
        <div style='font-size:2.5em;color:#10B981;font-weight:bold;'>{len(history_rows)}</div>
        <div style='color:#94A3B8;'>Véhicules traités</div>
    </div>""", unsafe_allow_html=True)

if occupation > 80:
    st.markdown(f"""<div class='alert-box'>
        ⚠️ <b>ALERTE CONGESTION</b> — {occupation:.0f}% ({nb_parked}/{CAPACITY})
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📷  Entrée / Sortie", "🚗  Véhicules présents", "📊  Statistiques"])

# ============================================================
# ONGLET 1
# ============================================================
with tab1:
    if st.session_state.last_exit:
        ex = st.session_state.last_exit
        st.markdown(f"""
        <div style='background:#0D2137;border:2px solid #10B981;
                    border-radius:12px;padding:25px;text-align:center;margin-bottom:20px;'>
            <div style='font-size:1.4em;font-weight:bold;color:#10B981;'>✅ Sortie confirmée automatiquement</div>
            <div style='font-size:2em;font-weight:bold;color:#F59E0B;margin:15px 0;font-family:monospace;'>{ex['plate']}</div>
            <div style='color:#E2E8F0;font-size:1.1em;'>🚗 {ex['brand']} &nbsp;|&nbsp; 🅿️ Place {ex['spot']} &nbsp;|&nbsp; ⏱ {ex['duration']/60:.0f} min</div>
            <div style='font-size:2em;font-weight:bold;color:#F59E0B;margin:15px 0;'>💰 {ex['price']:.2f} MAD</div>
            <div style='color:#94A3B8;font-size:0.9em;'>
                Entrée : {ex['entry_dt'].strftime('%d/%m/%Y %H:%M:%S')} &nbsp;|&nbsp;
                Sortie : {ex['exit_dt'].strftime('%d/%m/%Y %H:%M:%S')}
            </div>
        </div>""", unsafe_allow_html=True)
        col_pdf, col_new = st.columns(2)
        with col_pdf:
            pdf_bytes = generate_receipt(ex['plate'], ex['brand'], ex['entry_dt'],
                                         ex['exit_dt'], ex['duration'], ex['price'], ex['spot'])
            st.download_button("🧾 Télécharger le reçu PDF", data=pdf_bytes,
                               file_name=f"recu_{ex['plate']}_{ex['exit_dt'].strftime('%Y%m%d_%H%M')}.pdf",
                               mime="application/pdf", use_container_width=True, type="primary")
        with col_new:
            if st.button("📷 Scanner un nouveau véhicule", use_container_width=True):
                st.session_state.last_exit = None
                st.rerun()
    else:
        col_left, col_right = st.columns([1, 1])
        with col_left:
            st.markdown("#### 📤 Upload photo du véhicule")
            uploaded = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
            if uploaded:
                img_pil, annotated, crop, raw, cleaned, conf, vehicle, matched = process_image(uploaded)
                col_orig, col_crop = st.columns([3, 2])
                with col_orig:
                    st.markdown("**📷 Photo originale**")
                    st.image(img_pil, width='stretch')
                with col_crop:
                    st.markdown("**🔍 Plaque détectée**")
                    if crop is not None:
                        st.image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), width='stretch')
                    else:
                        st.warning("Aucune plaque détectée")

                if st.button("🎬 Voir la détection YOLO en live"):
                    from ultralytics import YOLO as YOLOModel
                    model_yolo   = YOLOModel("models/best_v8.pt")
                    img_bgr_anim = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                    results_anim = model_yolo(img_bgr_anim, verbose=False)
                    video_ph = st.empty()
                    if results_anim[0].boxes.xyxy.shape[0] > 0:
                        best_idx = results_anim[0].boxes.conf.argmax()
                        box_anim = results_anim[0].boxes.xyxy[best_idx]
                        conf_val = float(results_anim[0].boxes.conf[best_idx])
                        x1, y1, x2, y2 = map(int, box_anim)
                        h, w = img_bgr_anim.shape[:2]
                        for step in range(11):
                            frame = img_bgr_anim.copy()
                            t = step / 10
                            cx1 = int(w//2 + (x1-w//2)*t); cy1 = int(h//2 + (y1-h//2)*t)
                            cx2 = int(w//2 + (x2-w//2)*t); cy2 = int(h//2 + (y2-h//2)*t)
                            cv2.rectangle(frame, (cx1,cy1), (cx2,cy2), (6,182,int(255*t)), 3)
                            if step == 10:
                                lbl = f"Plaque {conf_val:.0%}"
                                cv2.rectangle(frame,(x1,y1-35),(x1+len(lbl)*13,y1),(6,182,212),-1)
                                cv2.putText(frame,lbl,(x1+5,y1-8),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
                            video_ph.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), width='stretch')
                            time.sleep(0.08)
                        st.success("✅ Détection terminée !")
                    else:
                        st.warning("Aucune plaque détectée")

                plate   = cleaned['cleaned']
                valid   = cleaned['valid']
                ptype   = cleaned['plate_type']
                css_cls = "valid-plate" if valid else "invalid-plate"
                st.markdown(f"<div class='plate-display {css_cls}'>{plate if plate else '— non lisible —'}</div>",
                            unsafe_allow_html=True)
                st.markdown(f"""<div style='margin-top:10px;color:#94A3B8;font-size:0.9em;'>
                    Texte brut : <code style='color:#F59E0B;'>{raw}</code><br>
                    Type : <b style='color:#06B6D4;'>{ptype}</b> &nbsp;|&nbsp;
                    Confiance : <b style='color:{"#10B981" if conf > 0.7 else "#F59E0B"};'>{conf:.0%}</b>
                </div>""", unsafe_allow_html=True)
                st.markdown("---")

                brand    = vehicle['brand']
                veh_conf = vehicle['confidence']
                all_brands = vehicle.get('all_brands', [])
                st.markdown(
                    f"<div style='background:#0D2137;border:1px solid #10B981;"
                    f"border-radius:8px;padding:12px;margin-top:10px;'>"
                    f"<div style='color:#10B981;font-weight:bold;'>"
                    f"🚗 Marque : <span style='color:#06B6D4;'>{brand}</span> "
                    f"<span style='color:#F59E0B;'>({veh_conf:.0%})</span></div></div>",
                    unsafe_allow_html=True)
                if all_brands:
                    for bn, bc in all_brands:
                        c1, c2 = st.columns([3, 1])
                        with c1: st.progress(bc, text=bn)
                        with c2: st.markdown(f"<div style='color:#F59E0B;padding-top:8px;'>{bc:.0%}</div>",
                                             unsafe_allow_html=True)
                st.caption(matched['message'])

                existing = conn.execute(
                    "SELECT id, entry_time, spot FROM sessions WHERE plate=? AND exit_time IS NULL",
                    (plate,)).fetchone()

        with col_right:
            if uploaded:
                st.markdown("#### 🎬 Action")
                if not plate:
                    st.warning("Plaque non lisible — saisie manuelle")
                    plate = st.text_input("Saisir la plaque manuellement :")

                just_entered = st.session_state.get('just_entered_plate') == plate

                if existing and not just_entered:
                    entry_dt         = datetime.fromisoformat(existing[1])
                    duration_preview = (datetime.now() - entry_dt).total_seconds()
                    price_preview    = duration_preview * TARIF
                    st.markdown(f"""
                    <div style='background:#0D2137;border:2px solid #F59E0B;
                                border-radius:12px;padding:20px;text-align:center;'>
                        <div style='color:#F59E0B;font-weight:bold;font-size:1.2em;'>🚗 Véhicule reconnu — Sortie</div>
                        <div style='font-size:1.8em;font-family:monospace;color:#06B6D4;margin:10px 0;'>{plate}</div>
                        <div style='color:#E2E8F0;'>🅿️ Place {existing[2]} &nbsp;|&nbsp; ⏱ {duration_preview/60:.0f} min</div>
                        <div style='font-size:1.6em;font-weight:bold;color:#F59E0B;margin:10px 0;'>💰 {price_preview:.2f} MAD</div>
                        <div style='color:#94A3B8;font-size:0.85em;'>Entrée : {entry_dt.strftime('%d/%m/%Y %H:%M:%S')}</div>
                    </div>""", unsafe_allow_html=True)
                    entry_dt_f, now_f, dur_f, price_f = do_exit(existing[0], existing[1], plate)
                    st.session_state.last_exit = {
                        'plate': plate, 'brand': vehicle['brand'],
                        'entry_dt': entry_dt_f, 'exit_dt': now_f,
                        'duration': dur_f, 'price': price_f, 'spot': existing[2]}
                    st.session_state.pop('just_entered_plate', None)
                    st.rerun()
                else:
                    if not existing:
                        spot = get_next_spot()
                        if spot:
                            now = datetime.now()
                            conn.execute("INSERT INTO sessions (plate,plate_type,brand,entry_time,spot,confidence) VALUES (?,?,?,?,?,?)",
                                         (plate, ptype, vehicle['brand'], now.isoformat(), spot, conf))
                            conn.execute("INSERT INTO known_vehicles (plate,plate_type,brand,last_seen) VALUES (?,?,?,?) "
                                         "ON CONFLICT(plate) DO UPDATE SET visit_count=visit_count+1,"
                                         "last_seen=excluded.last_seen,brand=excluded.brand",
                                         (plate, ptype, vehicle['brand'], now.isoformat()))
                            conn.commit()
                            st.session_state['just_entered_plate'] = plate
                    spot_display = existing[2] if just_entered and existing else (spot if not existing else "?")
                    st.markdown(f"""
                    <div class='success-box' style='text-align:center;padding:20px;'>
                        <div style='font-size:1.5em;font-weight:bold;color:#10B981;'>✅ Entrée enregistrée automatiquement</div>
                        <div style='margin-top:10px;color:#E2E8F0;'>🅿️ Place <b>{spot_display}</b> &nbsp;|&nbsp; 🚗 {vehicle['brand']}</div>
                        <div style='margin-top:8px;font-size:1.3em;font-family:monospace;color:#06B6D4;'>{plate}</div>
                        <div style='margin-top:10px;color:#94A3B8;font-size:0.9em;'>📷 Uploadez une nouvelle photo pour enregistrer une sortie</div>
                    </div>""", unsafe_allow_html=True)
            else:
                st.markdown("<div style='color:#64748B;text-align:center;margin-top:100px;'>← Uploadez une photo pour commencer</div>",
                            unsafe_allow_html=True)

# ============================================================
# ONGLET 2
# ============================================================
with tab2:
    parked = get_parked_vehicles()
    st.markdown(f"#### 🚗 {len(parked)} véhicule(s) actuellement dans le parking")
    if not parked:
        st.info("Aucun véhicule dans le parking pour l'instant.")
    else:
        for p_plate, p_ptype, p_brand, p_entry, p_spot, p_conf in parked:
            entry_dt     = datetime.fromisoformat(p_entry)
            duration_sec = (datetime.now() - entry_dt).total_seconds()
            price_so_far = duration_sec * TARIF
            with st.container():
                col_a, col_b, col_c, col_d = st.columns([2, 2, 2, 2])
                with col_a: st.markdown(f"**`{p_plate}`**  \n*{p_ptype}* — 🚗 {p_brand}")
                with col_b: st.markdown(f"🅿️ Place **{p_spot}**  \n⏱ {duration_sec/60:.0f} min")
                with col_c: st.markdown(f"💰 **{price_so_far:.2f} MAD** en cours  \n📅 {entry_dt.strftime('%H:%M:%S')}")
                with col_d:
                    if st.button("🚪 Sortie", key=f"exit_{p_plate}_{p_spot}",
                                 use_container_width=True, type="primary"):
                        sid = conn.execute("SELECT id FROM sessions WHERE plate=? AND exit_time IS NULL",
                                           (p_plate,)).fetchone()[0]
                        entry_dt_f, now_f, dur_f, price_f = do_exit(sid, p_entry, p_plate)
                        st.success(f"✅ Sortie **{p_plate}** — {dur_f/60:.1f} min — **{price_f:.2f} MAD**")
                        st.rerun()
                st.divider()

# ============================================================
# ONGLET 3
# ============================================================
with tab3:
    st.markdown("#### 📊 Analyse OCR — Dataset complet")
    fig_path = Path("results/figures/compare_ocr.png")
    if fig_path.exists():
        st.image(str(fig_path), width='stretch')
    else:
        st.warning("Lance d'abord `python scripts/compare_ocr.py`")
    lc_path = Path("results/figures/lowcost.png")
    if lc_path.exists():
        st.markdown("#### 🍓 Axe Low-cost — Simulation Raspberry Pi")
        st.image(str(lc_path), width='stretch')
    st.markdown("#### 🅿️ Statistiques de la session")
    history = get_history(limit=100)
    if history:
        import pandas as pd
        df = pd.DataFrame(history, columns=["Plaque","Type","Entrée","Sortie","Durée (min)","Prix (MAD)"])
        df["Prix (MAD)"]  = df["Prix (MAD)"].apply(lambda x: f"{x:.2f}" if x else "—")
        df["Durée (min)"] = df["Durée (min)"].apply(lambda x: f"{x:.1f}" if x else "—")
        st.dataframe(df, use_container_width=True, hide_index=True)
        known = conn.execute("SELECT plate, plate_type, visit_count, last_seen "
                             "FROM known_vehicles ORDER BY visit_count DESC LIMIT 10").fetchall()
        if known:
            st.markdown("#### 🔁 Véhicules fréquents")
            st.dataframe(pd.DataFrame(known, columns=["Plaque","Type","Visites","Dernière visite"]),
                         use_container_width=True, hide_index=True)
    else:
        st.info("Aucune session terminée pour l'instant.")
