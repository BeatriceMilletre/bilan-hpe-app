# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import yaml
import matplotlib.pyplot as plt
import io
from datetime import datetime
from pathlib import Path

st.set_page_config(page_title="Bilan HPE – Passation (YAML)", page_icon="🧠", layout="wide")

# ---------- Charger le YAML ----------
YAML_PATH = Path("questionnaire.yml")
if not YAML_PATH.exists():
    st.error("Le fichier 'questionnaire.yml' est introuvable à la racine du dépôt.")
    st.stop()

with YAML_PATH.open("r", encoding="utf-8") as f:
    data = yaml.safe_load(f)

thresholds = data.get("thresholds", {})

# ---------- Fonctions utilitaires ----------
DEFAULT_LIKERT4 = ["Tout à fait d’accord","Plutôt d’accord","Plutôt pas d’accord","Pas du tout d’accord"]

def categorize(value, thres: dict):
    if value is None:
        return "—"
    if value >= thres.get("high", 9e9):
        return "élevé"
    if value >= thres.get("medium", -9e9):
        return "moyen"
    return "faible"

def synthese_re(hr, er, he, ee, thres_re: dict):
    parts = []
    for lab, val in [("HR", hr), ("ER", er), ("HE", he), ("EE", ee)]:
        cat = categorize(val, thres_re.get(lab, {}))
        if lab in ["HR", "ER"]:
            if cat == "élevé":
                parts.append(f"{lab} **élevé** → forte aisance/motivation pour le raisonnement analytique.")
            elif cat == "moyen":
                parts.append(f"{lab} **moyen** → ressources rationnelles présentes.")
            else:
                parts.append(f"{lab} **faible** → sentiment moindre d’habileté rationnelle.")
        else:
            if cat == "élevé":
                parts.append(f"{lab} **élevé** → recours fréquent à l’intuition/ressenti.")
            elif cat == "moyen":
                parts.append(f"{lab} **moyen** → équilibre entre intuition et logique.")
            else:
                parts.append(f"{lab} **faible** → intuition moins mobilisée.")
    return "\n".join(f"- {p}" for p in parts)

def interpretation_bmri(bmri_result):
    if bmri_result is None or not bmri_result.get("choices"):
        return "—"
    a_cnt = sum(1 for v in bmri_result["choices"].values() if v == "A")
    b_cnt = len(bmri_result["choices"]) - a_cnt
    if abs(a_cnt - b_cnt) <= 2:
        return f"Profil équilibré (A={a_cnt}, B={b_cnt}) : alternance entre raisonnement et intuition."
    elif a_cnt > b_cnt:
        return f"Tendance rationnelle (A={a_cnt} > B={b_cnt}) : préférence pour l’analyse structurée."
    else:
        return f"Tendance expérientielle (B={b_cnt} > A={a_cnt}) : préférence pour l’intuition et le ressenti."

def compte_rendu_auto(short_totals, re_cats, name=None, age=None, bmri_result=None):
    spq_cat = re_cats.get("_SPQ_cat", "—")
    eq_cat  = re_cats.get("_EQ_cat", "—")
    qr_cat  = re_cats.get("_QR_cat", "—")
    qa_cat  = re_cats.get("_QA_cat", "—")

    hr_s, hr_c = re_cats.get("HR", (None,"—"))
    er_s, er_c = re_cats.get("ER", (None,"—"))
    he_s, he_c = re_cats.get("HE", (None,"—"))
    ee_s, ee_c = re_cats.get("EE", (None,"—"))

    bmri_sentence = interpretation_bmri(bmri_result) if bmri_result else "—"

    lignes = []
    lignes.append("## Compte rendu automatique")
    sous = []
    if name: sous.append(f"Nom : **{name}**")
    if age:  sous.append(f"Âge : **{age}**")
    sous.append(f"Date : **{datetime.now().strftime('%Y-%m-%d %H:%M')}**")
    lignes.append(" — ".join(sous))
    lignes.append("")

    # Vue d’ensemble
    phrases = []
    if he_c == "élevé" and (er_c in ["élevé","moyen"]):
        phrases.append("Profil **intuitif engagé** : recours fréquent au ressenti, avec appétit pour l’analyse.")
    if (hr_c == "faible") and (er_c in ["élevé","moyen"]):
        phrases.append("Motivation pour raisonner présente, mais **sentiment d’habileté analytique plus bas**.")
    if hr_c == "élevé" and he_c == "élevé":
        phrases.append("Double appui **logique + intuition** : alternance flexible.")
    if not phrases:
        phrases.append("Répartition des préférences R/E **équilibrée**.")
    if bmri_sentence and bmri_sentence != "—":
        phrases.append(f"BMRI : {bmri_sentence}")
    lignes.append("### Vue d’ensemble")
    lignes.append("- " + " ".join(phrases))

    return "\n".join(lignes)

# ---------- Graphiques ----------
def fig_re_radar(hr, er, he, ee):
    import math
    labels = ["HR","ER","HE","EE"]
    values = [v if v is not None else 0 for v in [hr, er, he, ee]]
    values += values[:1]
    angles = [n/float(len(labels))*2*math.pi for n in range(len(labels))]
    angles += angles[:1]

    fig = plt.figure(figsize=(4,4))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(math.pi/2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_yticks([1,2,3,4,5])
    ax.set_ylim(0,5)

    ax.plot(angles, values)
    ax.fill(angles, values, alpha=0.2)
    ax.set_title("Profil R/E")
    return fig

def fig_short_bars(short_totals):
    items = list(short_totals.keys())
    vals = [short_totals[k] for k in items]
    fig, ax = plt.subplots(figsize=(6,3))
    ax.bar(items, vals)
    ax.set_title("Questionnaires courts – totaux")
    ax.set_ylabel("Score")
    return fig

def fig_bmri_ab(bmri_result):
    a_cnt = sum(1 for v in bmri_result["choices"].values() if v=="A")
    b_cnt = len(bmri_result["choices"]) - a_cnt
    fig, ax = plt.subplots(figsize=(4,3))
    ax.bar(["A","B"], [a_cnt, b_cnt])
    ax.set_title("BMRI – A vs B")
    ax.set_ylabel("Nombre de réponses")
    return fig

def download_fig_button(fig, filename):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    st.download_button("⬇️ Télécharger le graphique", data=buf.getvalue(),
                       file_name=filename, mime="image/png")

# ---------- Blocs ----------
def ask_block_likert(block: dict):
    labels = block.get("scale_labels", DEFAULT_LIKERT4)
    out = {}
    st.subheader(block.get("key", "Échelle"))
    for it in block.get("items", []):
        vals = it["values"]
        n = len(vals)
        local_labels = (labels + [f"Option {i+1}" for i in range(len(labels), n)])[:n]
        c1, c2 = st.columns([3,2])
        with c1: st.write(f"**{it['id']}** — {it.get('text','')}")
        with c2:
            choice = st.radio(it["id"], options=list(range(n)), index=0,
                              horizontal=True, label_visibility="collapsed",
                              format_func=lambda i: local_labels[i])
        out[it["id"]] = vals[choice]
        st.divider()
    return out

def ask_block_forced_choice(block: dict):
    st.subheader(block.get("key", "Forced-choice (A/B)"))
    choices = {}
    for idx, it in enumerate(block.get("items", []), 1):
        stem, a, b = it.get("stem",""), it.get("a",""), it.get("b","")
        st.markdown(f"**{it.get('id', f'FC{idx}')}** — {stem}")
        pick = st.radio(it.get("id", f"FC{idx}"),
                        options=["A","B"], index=0, horizontal=True,
                        format_func=lambda x: f"{x}) {a if x=='A' else b}")
        choices[it.get("id", f"FC{idx}")] = pick
        st.divider()
    return {"choices": choices}

# ---------- Rapport ----------
def build_report(short_totals, re_scores, HR, ER, HE, EE, name, age, bmri_result=None):
    short_cats = {k: categorize(v, thresholds.get("short_scales", {}).get(k, {}))
                  for k, v in short_totals.items()}
    re_cats = {
        "HR": (HR, categorize(HR, thresholds.get("re_scales", {}).get("HR", {}))),
        "ER": (ER, categorize(ER, thresholds.get("re_scales", {}).get("ER", {}))),
        "HE": (HE, categorize(HE, thresholds.get("re_scales", {}).get("HE", {}))),
        "EE": (EE, categorize(EE, thresholds.get("re_scales", {}).get("EE", {}))),
        "_SPQ_cat": short_cats.get("SPQ-10", "—"),
        "_EQ_cat": short_cats.get("EQ-10", "—"),
        "_QR_cat": short_cats.get("Q-R-10", "—"),
        "_QA_cat": short_cats.get("QA-10", "—"),
    }
    L = []
    L.append("# Bilan HPE – Rapport")
    L.append(f"**Date** : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    meta = " – ".join(x for x in [f"Nom: {name}" if name else None, f"Âge: {age}" if age else None] if x)
    if meta: L.append(meta)
    L.append("")
    L.append("## Scores courts")
    for k,v in short_totals.items():
        L.append(f"- {k} : {v} ({short_cats.get(k,'—')})")
    L.append("")
    L.append("## R/E (1–5)")
    for lab,val in [("HR",HR),("ER",ER),("HE",HE),("EE",EE)]:
        sval = "—" if val is None else f"{val:.2f}"
        L.append(f"- {lab} : {sval}")
    L.append("")
    if bmri_result: 
        L.append("## BMRI")
        L.append(interpretation_bmri(bmri_result))
    L.append("")
    L.append(compte_rendu_auto(short_totals, re_cats, name, age, bmri_result))
    return "\n".join(L)

# ---------- Passation ----------
short_totals, re_scores, bmri_result = {}, {}, None

def mean_tag(tag, re_scores):
    vals = [v["score"] for v in re_scores.values() if v["tag"]==tag]
    return round(float(np.mean(vals)),2) if vals else None

for block in data.get("blocks", []):
    btype = block.get("type")
    if btype == "re":
        st.subheader("Échelle R/E (1–5)")
        scores = {}
        for it in block.get("items", []):
            iid, tag, text = it.get("id"), it.get("tag"), it.get("text","")
            val = st.slider(f"{iid} – {tag} : {text}", 1, 5, 3, key=f"rei-{iid}")
            if it.get("reverse", False): val = 6-val
            scores[str(iid)] = {"tag": str(tag).upper(), "score": float(val)}
        re_scores = scores
    elif btype == "forced_choice":
        bmri_result = ask_block_forced_choice(block)
    else:
        answers = ask_block_likert(block)
        short_totals[block["key"]] = sum(answers.values())

HR,ER,HE,EE = mean_tag("HR",re_scores), mean_tag("ER",re_scores), mean_tag("HE",re_scores), mean_tag("EE",re_scores)

# ---------- Résultats ----------
st.header("Résultats")
c1,c2 = st.columns(2)
with c1:
    st.subheader("Questionnaires courts")
    rows = [{"Échelle":k,"Total":v,"Catégorie":categorize(v, thresholds.get("short_scales", {}).get(k, {}))}
            for k,v in short_totals.items()]
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
with c2:
    st.subheader("R/E (1–5)")
    rows = [{"Sous-échelle":lab,"Score":val,"Catégorie":categorize(val, thresholds.get("re_scales", {}).get(lab, {}))}
            for lab,val in [("HR",HR),("ER",ER),("HE",HE),("EE",EE)]]
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

if bmri_result:
    st.subheader("BMRI – Résumé")
    st.write(interpretation_bmri(bmri_result))

# ---------- Graphiques ----------
st.subheader("📊 Visualisations")
colA,colB = st.columns(2)
with colA:
    fig1=fig_short_bars(short_totals); st.pyplot(fig1); download_fig_button(fig1,"courts.png")
with colB:
    fig2=fig_re_radar(HR,ER,HE,EE); st.pyplot(fig2); download_fig_button(fig2,"rei.png")
if bmri_result:
    fig3=fig_bmri_ab(bmri_result); st.pyplot(fig3); download_fig_button(fig3,"bmri.png")

# ---------- Exports ----------
with st.sidebar:
    st.header("Infos répondant")
    name=st.text_input("Nom (optionnel)")
    age=st.text_input("Âge (optionnel)")

report=build_report(short_totals,re_scores,HR,ER,HE,EE,name,age,bmri_result)
st.download_button("💾 Rapport (.md)", report, file_name="bilan_hpe_rapport.md", mime="text/markdown")

raw={k:v for k,v in short_totals.items()}
for k,v in re_scores.items(): raw[k]=v["score"]
raw.update({"HR":HR,"ER":ER,"HE":HE,"EE":EE})
if bmri_result: 
    for k,v in bmri_result["choices"].items(): raw[k]=v
st.download_button("⬇️ Réponses (.csv)", pd.DataFrame([raw]).to_csv(index=False).encode("utf-8"),
                   file_name="bilan_hpe_reponses.csv", mime="text/csv")
