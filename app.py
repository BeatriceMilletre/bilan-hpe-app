# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

st.set_page_config(page_title="Bilan HPE – simplifié", page_icon="🧠", layout="centered")

# -------------------------------------------------------------------
# 1) CONFIG MINIMALE – ÉDITABLE DANS CE FICHIER
# -------------------------------------------------------------------
# Chaque item = {id, text, values}
# values = liste de 4 valeurs (gauche → droite) pour les réponses 4 points
#   Exemples : [3,2,1,0]  ou  [2,1,0,0]  ou  [0,0,1,2]
# Tu peux changer les textes "Item ..." par tes vrais intitulés.
SPQ_10 = [
    {"id":"SPQ1","text":"SPQ - Item 1","values":[3,2,1,0]},
    {"id":"SPQ2","text":"SPQ - Item 2","values":[3,2,1,0]},
    {"id":"SPQ3","text":"SPQ - Item 3","values":[3,2,1,0]},
    {"id":"SPQ4","text":"SPQ - Item 4","values":[3,2,1,0]},
    {"id":"SPQ5","text":"SPQ - Item 5","values":[3,2,1,0]},
    {"id":"SPQ6","text":"SPQ - Item 6","values":[3,2,1,0]},
    {"id":"SPQ7","text":"SPQ - Item 7","values":[3,2,1,0]},
    {"id":"SPQ8","text":"SPQ - Item 8","values":[3,2,1,0]},
    {"id":"SPQ9","text":"SPQ - Item 9","values":[3,2,1,0]},
    {"id":"SPQ10","text":"SPQ - Item 10","values":[3,2,1,0]},
]

EQ_10 = [
    {"id":"EQ1","text":"EQ - Item 1","values":[2,1,0,0]},
    {"id":"EQ2","text":"EQ - Item 2","values":[0,0,1,2]},
    {"id":"EQ3","text":"EQ - Item 3","values":[2,1,0,0]},
    {"id":"EQ4","text":"EQ - Item 4","values":[0,0,1,2]},
    {"id":"EQ5","text":"EQ - Item 5","values":[2,1,0,0]},
    {"id":"EQ6","text":"EQ - Item 6","values":[0,0,1,2]},
    {"id":"EQ7","text":"EQ - Item 7","values":[2,1,0,0]},
    {"id":"EQ8","text":"EQ - Item 8","values":[0,0,1,2]},
    {"id":"EQ9","text":"EQ - Item 9","values":[2,1,0,0]},
    {"id":"EQ10","text":"EQ - Item 10","values":[0,0,1,2]},
]

QR_10 = [
    {"id":"QR1","text":"Q-R - Item 1","values":[3,2,1,0]},
    {"id":"QR2","text":"Q-R - Item 2","values":[3,2,1,0]},
    {"id":"QR3","text":"Q-R - Item 3","values":[3,2,1,0]},
    {"id":"QR4","text":"Q-R - Item 4","values":[3,2,1,0]},
    {"id":"QR5","text":"Q-R - Item 5","values":[3,2,1,0]},
    {"id":"QR6","text":"Q-R - Item 6","values":[3,2,1,0]},
    {"id":"QR7","text":"Q-R - Item 7","values":[3,2,1,0]},
    {"id":"QR8","text":"Q-R - Item 8","values":[3,2,1,0]},
    {"id":"QR9","text":"Q-R - Item 9","values":[3,2,1,0]},
    {"id":"QR10","text":"Q-R - Item 10","values":[3,2,1,0]},
]

QA_10 = [
    {"id":"QA1","text":"QA - Item 1","values":[1,1,0,0]},
    {"id":"QA2","text":"QA - Item 2","values":[0,0,1,1]},
    {"id":"QA3","text":"QA - Item 3","values":[1,1,0,0]},
    {"id":"QA4","text":"QA - Item 4","values":[0,0,1,1]},
    {"id":"QA5","text":"QA - Item 5","values":[1,1,0,0]},
    {"id":"QA6","text":"QA - Item 6","values":[0,0,1,1]},
    {"id":"QA7","text":"QA - Item 7","values":[1,1,0,0]},
    {"id":"QA8","text":"QA - Item 8","values":[0,0,1,1]},
    {"id":"QA9","text":"QA - Item 9","values":[1,1,0,0]},
    {"id":"QA10","text":"QA - Item 10","values":[0,0,1,1]},
]

# Échelle R/E simplifiée : 20 items (exemple) tagués HR/ER/HE/EE.
# Chaque item est un slider 1..5 transformé en score 1..5 (reverse possible).
RE_ITEMS = [
    # tag: HR (Habileté rationnelle)
    {"id":"R1","tag":"HR","text":"R - HR Item 1","reverse":False},
    {"id":"R2","tag":"HR","text":"R - HR Item 2","reverse":True},
    {"id":"R3","tag":"HR","text":"R - HR Item 3","reverse":False},
    {"id":"R4","tag":"HR","text":"R - HR Item 4","reverse":False},
    {"id":"R5","tag":"HR","text":"R - HR Item 5","reverse":True},
    # ER
    {"id":"R6","tag":"ER","text":"R - ER Item 1","reverse":False},
    {"id":"R7","tag":"ER","text":"R - ER Item 2","reverse":False},
    {"id":"R8","tag":"ER","text":"R - ER Item 3","reverse":True},
    {"id":"R9","tag":"ER","text":"R - ER Item 4","reverse":False},
    {"id":"R10","tag":"ER","text":"R - ER Item 5","reverse":False},
    # HE
    {"id":"R11","tag":"HE","text":"R - HE Item 1","reverse":False},
    {"id":"R12","tag":"HE","text":"R - HE Item 2","reverse":True},
    {"id":"R13","tag":"HE","text":"R - HE Item 3","reverse":False},
    {"id":"R14","tag":"HE","text":"R - HE Item 4","reverse":False},
    {"id":"R15","tag":"HE","text":"R - HE Item 5","reverse":True},
    # EE
    {"id":"R16","tag":"EE","text":"R - EE Item 1","reverse":False},
    {"id":"R17","tag":"EE","text":"R - EE Item 2","reverse":False},
    {"id":"R18","tag":"EE","text":"R - EE Item 3","reverse":True},
    {"id":"R19","tag":"EE","text":"R - EE Item 4","reverse":False},
    {"id":"R20","tag":"EE","text":"R - EE Item 5","reverse":False},
]

LIKERT4 = ["Tout à fait d’accord","Plutôt d’accord","Plutôt pas d’accord","Pas du tout d’accord"]

# -------------------------------------------------------------------
# 2) UI – EN-TÊTE
# -------------------------------------------------------------------
st.title("🧠 Bilan HPE – Passation simplifiée (sans DOCX)")
st.caption("Réponds aux items ci-dessous. Les barèmes sont intégrés dans l’app (modifiables dans le code).")

with st.sidebar:
    st.header("Infos répondant")
    name = st.text_input("Nom (optionnel)")
    age = st.text_input("Âge (optionnel)")
    st.markdown("---")
    st.write("ℹ️ Pour modifier les items/barèmes, édite `app.py` (section CONFIG).")

# -------------------------------------------------------------------
# 3) FORMULAIRES – BLOCS COURTS
# -------------------------------------------------------------------
def ask_block_4pt(block_title, items):
    st.subheader(block_title)
    out = {}
    for it in items:
        col1, col2 = st.columns([3,2])
        with col1:
            st.write(f"**{it['id']}** — {it['text']}")
        with col2:
            choice = st.radio(
                it["id"], options=list(range(4)), index=0, horizontal=True,
                label_visibility="collapsed", format_func=lambda i: LIKERT4[i]
            )
        out[it["id"]] = it["values"][choice]
        st.divider()
    return out

with st.expander("🟦 SPQ-10"):
    spq_ans = ask_block_4pt("SPQ-10", SPQ_10)
with st.expander("🟨 EQ-10"):
    eq_ans = ask_block_4pt("EQ-10", EQ_10)
with st.expander("🟩 Q-R-10"):
    qr_ans = ask_block_4pt("Q-R-10", QR_10)
with st.expander("🟧 QA-10"):
    qa_ans = ask_block_4pt("QA-10", QA_10)

# -------------------------------------------------------------------
# 4) FORMULAIRE – R/E (HR, ER, HE, EE)
# -------------------------------------------------------------------
st.subheader("🟥 Échelle Rationnelle / Expérientielle")
st.caption("Score 1–5 par item (réversion appliquée si cochée). Les moyennes par tag donnent HR / ER / HE / EE.")
re_scores = {}
for it in RE_ITEMS:
    val = st.slider(f"{it['id']} – {it['tag']}: {it['text']}", 1, 5, 3)
    if it["reverse"]:
        val = 6 - val  # inversion 1↔5, 2↔4, 3↔3
    re_scores[it["id"]] = {"tag": it["tag"], "score": float(val)}

# -------------------------------------------------------------------
# 5) SCORING + AFFICHAGE
# -------------------------------------------------------------------
def total(d): return sum(d.values()) if d else 0.0

spq_total = total(spq_ans)
eq_total  = total(eq_ans)
qr_total  = total(qr_ans)
qa_total  = total(qa_ans)

# Moyenne par tag pour R/E
def mean_tag(tag):
    vals = [v["score"] for k, v in re_scores.items() if v["tag"] == tag]
    return round(float(np.mean(vals)), 2) if vals else None

HR = mean_tag("HR")
ER = mean_tag("ER")
HE = mean_tag("HE")
EE = mean_tag("EE")

def interp_re(label, val):
    if val is None: return ""
    if label in ["HR","ER"]:
        if val >= 4: return f"{label} élevé : aisance pour le raisonnement explicite et structuré."
        if val >= 3: return f"{label} modéré : ressources rationnelles présentes."
        return f"{label} faible : appui moindre sur l’analytique (à contextualiser)."
    else:
        if val >= 4: return f"{label} élevé : place importante de l’intuition/ressenti."
        if val >= 3: return f"{label} modéré : intuition et logique coexistent."
        return f"{label} faible : faible recours déclaré à l’intuition."

st.header("Résultats")
c1, c2 = st.columns(2)
with c1:
    st.subheader("Synthèse questionnaires courts")
    st.dataframe(pd.DataFrame([
        {"Échelle":"SPQ-10","Total":spq_total},
        {"Échelle":"EQ-10","Total":eq_total},
        {"Échelle":"Q-R-10","Total":qr_total},
        {"Échelle":"QA-10","Total":qa_total},
    ]), use_container_width=True)

with c2:
    st.subheader("Rationnel / Expérientiel (1–5)")
    df_re = pd.DataFrame([
        {"Sous-échelle":"HR","Score":HR,"Interprétation":interp_re("HR",HR)},
        {"Sous-échelle":"ER","Score":ER,"Interprétation":interp_re("ER",ER)},
        {"Sous-échelle":"HE","Score":HE,"Interprétation":interp_re("HE",HE)},
        {"Sous-échelle":"EE","Score":EE,"Interprétation":interp_re("EE",EE)},
    ])
    st.dataframe(df_re, use_container_width=True)

# -------------------------------------------------------------------
# 6) EXPORTS
# -------------------------------------------------------------------
def build_report():
    L = []
    L.append("# Bilan HPE – Rapport (simplifié)")
    L.append(f"**Date** : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    meta = " – ".join([x for x in [f"Nom: {name}" if name else None, f"Âge: {age}" if age else None] if x])
    if meta: L.append(meta)
    L.append("")
    L.append("## Scores – courts")
    L.append(f"- SPQ-10 : **{spq_total:.0f}**")
    L.append(f"- EQ-10 : **{eq_total:.0f}**")
    L.append(f"- Q-R-10 : **{qr_total:.0f}**")
    L.append(f"- QA-10 : **{qa_total:.0f}**")
    L.append("")
    L.append("## Rationnel / Expérientiel (1–5)")
    for lab, val in [("HR",HR),("ER",ER),("HE",HE),("EE",EE)]:
        txt = interp_re(lab, val)
        sval = "—" if val is None else f"{val:.2f}"
        L.append(f"- {lab} : **{sval}** – {txt}")
    L.append("")
    L.append("> Note : barèmes et textes d’items sont paramétrables dans le fichier `app.py`.")
    return "\n".join(L)

report = build_report()
st.download_button("💾 Télécharger le rapport (.md)", report, file_name="bilan_hpe_rapport.md", mime="text/markdown")

# Export CSV des réponses
raw = {}
raw.update({k:v for k,v in spq_ans.items()})
raw.update({k:v for k,v in eq_ans.items()})
raw.update({k:v for k,v in qr_ans.items()})
raw.update({k:v for k,v in qa_ans.items()})
for k,v in re_scores.items():
    raw[k] = v["score"]
raw.update({"HR":HR,"ER":ER,"HE":HE,"EE":EE})
st.download_button("⬇️ Exporter les réponses (.csv)", pd.DataFrame([raw]).to_csv(index=False).encode("utf-8"),
                   file_name="bilan_hpe_reponses.csv", mime="text/csv")

st.success("OK. Tu peux répondre, voir les totaux et exporter le rapport/CSV.")
