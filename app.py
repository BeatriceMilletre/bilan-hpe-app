# -*- coding: utf-8 -*-
import re
from io import BytesIO
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
from docx import Document

st.set_page_config(page_title="Bilan HPE – Passation & Scores", page_icon="🧠", layout="wide")

# =========================
# Outils parsing DOCX
# =========================
SECTION_HEADERS = [
    "SPQ 10", "EQ 10", "Q-R-10", "QA– 10", "QA- 10", "QA-10",
    "Echelle Rationnelle Expérientielle", "Échelle Rationnelle Expérientielle",
    "Codage inversé", "Habileté rationnelle", "Engagement rationnel",
    "Habileté expérientielle", "Engagement expérientiel"
]

# Map d’options lisibles -> valeur numérique selon les barèmes présents dans le DOCX
# NB : On détecte (3 2 1 0) etc pour chaque item, donc on stocke aussi un fallback Likert.
LIKERT_LABELS_FR = [
    "Tout à fait d’accord",
    "Plutôt d’accord",
    "Plutôt pas d’accord",
    "Pas du tout d’accord"
]

LIKERT_LABELS_FR4_REV = [
    "Tout à fait d’accord",
    "Plutôt d’accord",
    "Plutôt en désaccord",
    "Tout à fait en désaccord"
]

def load_docx(file_bytes: bytes) -> Document:
    return Document(BytesIO(file_bytes))

def paragraphs_text(doc: Document):
    for p in doc.paragraphs:
        txt = p.text.strip()
        if txt:
            yield txt

def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())

def detect_section_blocks(lines):
    # Renvoie dict {section_name: [lignes]}
    sections = {}
    current = None
    for ln in lines:
        line = normalize(ln)
        if any(line.startswith(h) for h in SECTION_HEADERS):
            current = line
            sections[current] = []
        elif current:
            sections[current].append(line)
    return sections

# Extraction des items “intitulé … (3 2 1 0)” etc
BAR_PATTERN = re.compile(r"\((?P<scores>(?:\d+\s+)+\d+)\)")
FOUR_COL_SET = set(LIKERT_LABELS_FR)
FOUR_COL_ALT_SET = set(LIKERT_LABELS_FR4_REV)

def parse_simple_scale(block_lines, default_labels):
    """
    Parse items sous forme:
    - Intitulé ... (3 2 1 0)  -> on lit le mapping spécifique
    - Entre les items, des lignes vides/consignes possibles
    Retourne: list[dict{id,text,labels,values}]
    """
    items = []
    idx = 1
    buf = []
    for line in block_lines:
        if not line:
            continue
        # Item si ligne contient un barème (nombres entre parenthèses)
        m = BAR_PATTERN.search(line)
        if m:
            scores = [int(x) for x in m.group("scores").split()]
            # Nettoyage intitulé sans la parenthèse
            text = normalize(BAR_PATTERN.sub("", line)).strip(" .:;-")
            # Construit labels (4 positions par défaut)
            labels = default_labels
            # Valeurs alignées à gauche (labels[0] => scores[0], etc.)
            # Si le doc a 4 valeurs, on mappe 4 labels. Si 2 valeurs répétées, on conserve tel quel.
            values = scores
            items.append({
                "id": f"Q{idx}",
                "text": text,
                "labels": labels,
                "values": values
            })
            idx += 1
        # On ignore les autres lignes (titres de colonnes déjà connus)
    return items

def parse_eq_block(block_lines):
    # EQ-10 présente souvent (2 1 0 0) ou (0 0 1 2) selon l’orientation.
    # On lit chaque item avec son barème explicite.
    return parse_simple_scale(block_lines, LIKERT_LABELS_FR)

def parse_spq_block(block_lines):
    # SPQ-10 avec (3 2 1 0)
    return parse_simple_scale(block_lines, LIKERT_LABELS_FR)

def parse_qr_block(block_lines):
    # Q-R-10 avec libellés “Plutôt en désaccord / Tout à fait en désaccord”
    return parse_simple_scale(block_lines, LIKERT_LABELS_FR4_REV)

def parse_qa_block(block_lines):
    # QA-10 mélange des (1 1 0 0), (0 0 1 1), etc.
    # On lit le barème propre à chaque item.
    # Libellés proches de Q-R-10, on garde la même variante 4 points.
    return parse_simple_scale(block_lines, LIKERT_LABELS_FR4_REV)

# Parsing Échelle Rationnelle/Expérientielle
# On détecte les items “HR/ER/HE/EE” + barème (5 4 3 2 1) ou inversé (1 2 3 4 5)
FIVE_LABELS = ["Tout à fait d’accord", "Plutôt d’accord", "NSP", "Plutôt pas d’accord", "Tout à fait d’accord"]  # juste indicatif; l’app affiche un slider 1..5

FIVE_PATTERN = re.compile(r"\((?P<a>\d)\s+(?P<b>\d)\s+(?P<c>\d)\s+(?P<d>\d)\s+(?P<e>\d)\)")
SCALE_TAG_PATTERN = re.compile(r"^(HR|ER|HE|EE)\s*(.*)$", re.IGNORECASE)

def parse_re_block(block_lines):
    items = []
    idx = 1
    for line in block_lines:
        m_tag = SCALE_TAG_PATTERN.match(line)
        m_bar = FIVE_PATTERN.search(line)
        if m_tag and m_bar:
            tag = m_tag.group(1).upper()
            text = normalize(m_tag.group(2)).strip(" .:;-")
            values = [int(m_bar.group(k)) for k in ["a","b","c","d","e"]]
            items.append({
                "id": f"R{idx}",
                "text": text,
                "labels": ["5","4","3","2","1"],  # on affichera 1..5 UI; values fait foi
                "values": values,
                "tag": tag
            })
            idx += 1
    return items

REV_LIST_PATTERN = re.compile(r"Codage inversé\s*[:：]?\s*(.*)$", re.IGNORECASE)

def extract_reverse_list(full_text_lines):
    # Cherche la ligne “Codage inversé : …” et récupère la liste des numéros pour l’échelle R/E
    joined = " ".join(full_text_lines)
    m = REV_LIST_PATTERN.search(joined)
    if not m:
        return set()
    nums = re.findall(r"\d+", m.group(1))
    return set(int(n) for n in nums)

# Formules 4 scores (moyenne de certains items)
# On repère la ligne “Habileté rationnelle = (1 + 4 + ...)/10” etc
FORMULA_PATTERN = re.compile(
    r"(Habileté rationnelle|Engagement rationnel|Habileté expérientielle|Engagement expérientiel)\s*=\s*\(([^)]+)\)\s*/\s*(\d+)",
    re.IGNORECASE
)

def extract_formulas(full_text):
    formulas = {}
    for m in FORMULA_PATTERN.finditer(full_text):
        label = m.group(1).strip().lower()
        nums = [int(x) for x in re.findall(r"\d+", m.group(2))]
        denom = int(m.group(3))
        formulas[label] = {"items": nums, "denom": denom}
    return formulas

# =========================
# UI – Upload & Parsing
# =========================
st.title("🧠 Bilan HPE – Passation & Interprétation automatique")
st.write("Charge **ton fichier DOCX** puis passe les questionnaires. Les scores et un compte-rendu seront générés.")

uploaded = st.file_uploader("Dépose le fichier DOCX (ex: Bilan HPE 10 avec codage.docx)", type=["docx"])
if not uploaded:
    st.info("En attente de ton fichier…")
    st.stop()

doc = load_docx(uploaded.read())
lines = [normalize(t) for t in paragraphs_text(doc)]
sections = detect_section_blocks(lines)
full_text = " ".join(lines)

# Reverse list + formules pour l’échelle R/E
reverse_set = extract_reverse_list(lines)
formulas = extract_formulas(full_text)

# Parse des 5 blocs principaux
def get_block(name_candidates):
    for key in sections.keys():
        for cand in name_candidates:
            if key.lower().startswith(cand.lower()):
                return sections[key]
    return []

spq_block = get_block(["SPQ 10"])
eq_block  = get_block(["EQ 10"])
qr_block  = get_block(["Q-R-10"])
qa_block  = get_block(["QA– 10","QA- 10","QA-10"])
re_block  = get_block(["Echelle Rationnelle Expérientielle","Échelle Rationnelle Expérientielle"])

spq_items = parse_spq_block(spq_block)
eq_items  = parse_eq_block(eq_block)
qr_items  = parse_qr_block(qr_block)
qa_items  = parse_qa_block(qa_block)
re_items  = parse_re_block(re_block)

if not any([spq_items, eq_items, qr_items, qa_items, re_items]):
    st.error("Je n’ai pas réussi à extraire des items. Vérifie le format du document.")
    st.stop()

# =========================
# Passation – Méta
# =========================
with st.sidebar:
    st.header("Infos répondant")
    name = st.text_input("Nom (optionnel)")
    age = st.text_input("Âge (optionnel)")
    st.caption("Les infos s’ajoutent au rapport. Non obligatoires.")

# =========================
# UI – Questionnaire
# =========================
def ask_block(title, items, scale_note=None):
    st.subheader(title)
    if scale_note:
        st.caption(scale_note)
    answers = {}
    for it in items:
        # valeurs et labels peuvent diverger (ex: (2 1 0 0))
        lbls = it["labels"]
        vals = it["values"]
        # UI: on affiche un select horizontal
        cols = st.columns(2)
        with cols[0]:
            st.write(f"**{it['id']}**. {it['text']}")
        with cols[1]:
            choice = st.radio(
                f"{title}-{it['id']}",
                options=list(range(len(vals))),
                index=0,
                horizontal=True,
                label_visibility="collapsed",
                format_func=lambda i: lbls[i] if i < len(lbls) else str(i)
            )
        answers[it["id"]] = vals[choice]
        st.divider()
    return answers

spq_answers = {}
eq_answers  = {}
qr_answers  = {}
qa_answers  = {}
re_answers  = {}

with st.expander("🟦 SPQ-10 (sensibilités sensorielles)"):
    spq_answers = ask_block(
        "SPQ-10",
        spq_items,
        "Barème itemisé tel que dans le document (ex: 3-2-1-0)."
    )

with st.expander("🟨 EQ-10 (empathie)"):
    eq_answers = ask_block(
        "EQ-10",
        eq_items,
        "Barème itemisé tel que dans le document (ex: 2-1-0-0 / 0-0-1-2)."
    )

with st.expander("🟩 Q-R-10"):
    qr_answers = ask_block(
        "Q-R-10",
        qr_items,
        "Barème itemisé (4 points)."
    )

with st.expander("🟧 QA-10"):
    qa_answers = ask_block(
        "QA-10",
        qa_items,
        "Barème itemisé (4 points)."
    )

with st.expander("🟥 Échelle Rationnelle / Expérientielle (HR, ER, HE, EE)"):
    st.caption("Barèmes 5 points itemisés; **codage inversé** appliqué aux numéros indiqués dans le document.")
    # On affiche un slider 1..5 mais on enregistrera la note selon le mapping values
    for i, it in enumerate(re_items, start=1):
        st.write(f"**{i}. [{it.get('tag','?')}]** {it['text']}")
        # On propose un choix 1..5 “acquiescement -> désaccord”
        val = st.slider(f"re-{i}", min_value=1, max_value=5, value=5, label_visibility="collapsed")
        # Convertit 1..5 en index 0..4 (gauche->droite)
        idx = 5 - val  # 5 -> 0, 1 -> 4 (car le doc parfois est (5 4 3 2 1))
        # Score brut via valeurs de la parenthèse
        raw_score = it["values"][idx]
        # Codage inversé si i dans reverse_set
        if i in reverse_set:
            # Inversion “linéaire” sur échelle 1..5: on permute index
            idx_inv = 4 - idx
            raw_score = it["values"][idx_inv]
        re_answers[i] = {"score": raw_score, "tag": it.get("tag","")}

# =========================
# Scoring
# =========================
def sum_dict(d):
    return sum(float(v) for v in d.values()) if d else 0.0

spq_total = sum_dict(spq_answers)
eq_total  = sum_dict(eq_answers)
qr_total  = sum_dict(qr_answers)
qa_total  = sum_dict(qa_answers)

# R/E – calcule 4 sous-échelles selon formules extraites
def compute_subscale(formula_label):
    f = formulas.get(formula_label, None)
    if not f:
        return None
    vals = []
    for n in f["items"]:
        if n in re_answers:
            vals.append(float(re_answers[n]["score"]))
    if not vals:
        return None
    return round(np.mean(vals), 3)

HR = compute_subscale("habileté rationnelle")
ER = compute_subscale("engagement rationnel")
HE = compute_subscale("habileté expérientielle")
EE = compute_subscale("engagement expérientiel")

# =========================
# Interprétation (minimale, à ajuster après validation)
# =========================
def band(value, low_mid_high=(0.33, 0.66)):
    if value is None:
        return "—"
    # Normalisation simple selon max théorique approximatif
    # SPQ-10: items typiques (3 max) * 10 => ~30; EQ ~20; QR ~20; QA ~10 (selon barèmes)
    return value

def interp_re(label, val):
    if val is None:
        return ""
    if label in ["HR","ER"]:
        # Rationnel
        if val >= 4.0:
            return f"{label} élevé : appétence/aisance pour l’analyse, le raisonnement explicite."
        elif val >= 3.0:
            return f"{label} modéré : ressources rationnelles présentes, variables selon contexte."
        else:
            return f"{label} faible : préférences possiblement moins ‘logiques’ déclarées (à contextualiser)."
    else:
        # Expérientiel
        if val >= 4.0:
            return f"{label} élevé : place importante de l’intuition/ressenti dans les décisions."
        elif val >= 3.0:
            return f"{label} modéré : l’intuition coexiste avec le rationnel."
        else:
            return f"{label} faible : faible recours auto-rapporté à l’intuition."
        
def format_score(val):
    return "—" if val is None else f"{val:.2f}"

# =========================
# Affichage résultats
# =========================
st.header("Résultats")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Synthèse questionnaires courts")
    df_short = pd.DataFrame([
        {"Échelle": "SPQ-10", "Total": spq_total},
        {"Échelle": "EQ-10",  "Total": eq_total},
        {"Échelle": "Q-R-10", "Total": qr_total},
        {"Échelle": "QA-10",  "Total": qa_total},
    ])
    st.dataframe(df_short, use_container_width=True)

with col2:
    st.subheader("Rationnel / Expérientiel")
    df_re = pd.DataFrame([
        {"Sous-échelle": "HR (Habileté rationnelle)", "Score (1-5)": HR, "Interprétation": interp_re("HR", HR)},
        {"Sous-échelle": "ER (Engagement rationnel)", "Score (1-5)": ER, "Interprétation": interp_re("ER", ER)},
        {"Sous-échelle": "HE (Habileté expérientielle)", "Score (1-5)": HE, "Interprétation": interp_re("HE", HE)},
        {"Sous-échelle": "EE (Engagement expérientiel)", "Score (1-5)": EE, "Interprétation": interp_re("EE", EE)},
    ])
    st.dataframe(df_re, use_container_width=True)

# =========================
# Rapport exportable
# =========================
def build_report():
    lines = []
    lines.append(f"# Bilan HPE – Compte-rendu automatique")
    lines.append(f"**Date** : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    meta = " – ".join([p for p in [f"Nom: {name}" if name else None, f"Âge: {age}" if age else None] if p])
    if meta: lines.append(meta)
    lines.append("")
    lines.append("## Scores – questionnaires courts")
    lines.append(f"- SPQ-10 : **{spq_total:.0f}**")
    lines.append(f"- EQ-10 : **{eq_total:.0f}**")
    lines.append(f"- Q-R-10 : **{qr_total:.0f}**")
    lines.append(f"- QA-10 : **{qa_total:.0f}**")
    lines.append("")
    lines.append("## Rationnel / Expérientiel (1–5)")
    lines.append(f"- HR : **{format_score(HR)}** – {interp_re('HR', HR)}")
    lines.append(f"- ER : **{format_score(ER)}** – {interp_re('ER', ER)}")
    lines.append(f"- HE : **{format_score(HE)}** – {interp_re('HE', HE)}")
    lines.append(f"- EE : **{format_score(EE)}** – {interp_re('EE', EE)}")
    lines.append("")
    lines.append("> *Note : seuils/interprétations à affiner après validation interne. Les vignettes A/B du document sont **illustratives** et non scorées dans cette app.*")
    return "\n".join(lines)

report_md = build_report()
st.download_button("💾 Télécharger le rapport (.md)", data=report_md, file_name="bilan_hpe_rapport.md", mime="text/markdown")

# Export CSV des réponses brutes
raw = {
    **{f"SPQ_{k}": v for k, v in spq_answers.items()},
    **{f"EQ_{k}": v for k, v in eq_answers.items()},
    **{f"QR_{k}": v for k, v in qr_answers.items()},
    **{f"QA_{k}": v for k, v in qa_answers.items()},
}
# R/E détaillé
for n, obj in re_answers.items():
    raw[f"RE_item_{n}"] = obj["score"]
raw.update({"HR": HR, "ER": ER, "HE": HE, "EE": EE})

raw_df = pd.DataFrame([raw])
st.download_button("⬇️ Exporter les réponses (.csv)", data=raw_df.to_csv(index=False).encode("utf-8"), file_name="bilan_hpe_reponses.csv", mime="text/csv")

st.success("Prêt. Tu peux passer les questionnaires, puis exporter le rapport et les données.")
