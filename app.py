# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import yaml
from datetime import datetime
from pathlib import Path

st.set_page_config(page_title="Bilan HPE – Passation (YAML)", page_icon="🧠", layout="centered")

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
                parts.append(f"{lab} **faible** → sentiment moindre d’habileté rationnelle (à contextualiser).")
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

# ---------- UI blocs ----------
def ask_block_likert(block: dict):
    labels = block.get("scale_labels", DEFAULT_LIKERT4)
    out = {}
    st.subheader(block.get("key", "Échelle"))
    for it in block.get("items", []):
        vals = it["values"]
        n = len(vals)
        local_labels = (labels + [f"Option {i+1}" for i in range(len(labels), n)])[:n]
        c1, c2 = st.columns([3, 2])
        with c1:
            st.write(f"**{it['id']}** — {it.get('text','')}")
        with c2:
            choice = st.radio(
                it["id"], options=list(range(n)), index=0, horizontal=True,
                label_visibility="collapsed", format_func=lambda i: local_labels[i]
            )
        out[it["id"]] = vals[choice]
        st.divider()
    return out

def ask_block_forced_choice(block: dict):
    st.subheader(block.get("key", "Forced-choice (A/B)"))
    choices = {}
    for idx, it in enumerate(block.get("items", []), 1):
        stem = it.get("stem", "")
        a = it.get("a", "")
        b = it.get("b", "")
        st.markdown(f"**{it.get('id', f'FC{idx}')}** — {stem}")
        pick = st.radio(
            it.get("id", f"FC{idx}"),
            options=["A","B"],
            index=0,
            horizontal=True,
            format_func=lambda x: f"{x}) {a if x=='A' else b}"
        )
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
    }

    L = []
    L.append("# Bilan HPE – Rapport (YAML)")
    L.append(f"**Date** : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    meta = " – ".join(x for x in [f"Nom: {name}" if name else None, f"Âge: {age}" if age else None] if x)
    if meta: L.append(meta)
    L.append("")
    L.append("## Scores – courts")
    for k, v in short_totals.items():
        cat = short_cats.get(k, "—")
        L.append(f"- {k} : **{v:.0f}** ({cat})")
    L.append("")
    L.append("## Rationnel / Expérientiel (1–5)")
    for lab, val in [("HR",HR),("ER",ER),("HE",HE),("EE",EE)]:
        cat = re_cats[lab][1]
        sval = "—" if val is None else f"{val:.2f}"
        L.append(f"- {lab} : **{sval}** ({cat})")
    L.append("")
    L.append("## Synthèse R/E")
    L.append(synthese_re(HR, ER, HE, EE, thresholds.get("re_scales", {})))
    if bmri_result is not None:
        L.append("")
        L.append("## BMRI (28 items)")
        L.append(interpretation_bmri(bmri_result))
    return "\n".join(L)

# ---------- Passation ----------
short_totals = {}
re_scores = {}
bmri_result = None

def mean_tag(tag: str, re_scores):
    vals = [v["score"] for v in re_scores.values() if v["tag"] == tag]
    return round(float(np.mean(vals)), 2) if vals else None

for block in data.get("blocks", []):
    btype = block.get("type")
    if btype == "re":
        # REI (garde ta fonction existante de sliders RE)
        st.subheader("Échelle Rationnelle / Expérientielle (1–5)")
        scores = {}
        for it in block.get("items", []):
            iid, tag, text = it.get("id"), it.get("tag"), it.get("text","")
            val = st.slider(f"{iid} – {tag} : {text}", 1, 5, 3, key=f"rei-{iid}")
            if it.get("reverse", False):
                val = 6 - val
            scores[str(iid)] = {"tag": str(tag).upper(), "score": float(val)}
        re_scores = scores
    elif btype == "forced_choice":
        bmri_result = ask_block_forced_choice(block)
    else:
        answers = ask_block_likert(block)
        short_totals[block["key"]] = sum(answers.values())

HR = mean_tag("HR", re_scores); ER = mean_tag("ER", re_scores)
HE = mean_tag("HE", re_scores); EE = mean_tag("EE", re_scores)

# ---------- Résultats ----------
st.header("Résultats")
c1, c2 = st.columns(2)
with c1:
    st.subheader("Questionnaires courts")
    rows = []
    for k, v in short_totals.items():
        cat = categorize(v, thresholds.get("short_scales", {}).get(k, {}))
        rows.append({"Échelle": k, "Total": v, "Catégorie": cat})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
with c2:
    st.subheader("Rationnel / Expérientiel (1–5)")
    rows = []
    for lab, val in [("HR",HR),("ER",ER),("HE",HE),("EE",EE)]:
        cat = categorize(val, thresholds.get("re_scales", {}).get(lab, {}))
        rows.append({"Sous-échelle": lab, "Score": val, "Catégorie": cat})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

if bmri_result is not None:
    st.subheader("BMRI – Résumé")
    st.write(interpretation_bmri(bmri_result))

# ---------- Exports ----------
with st.sidebar:
    st.header("Infos répondant")
    name = st.text_input("Nom (optionnel)")
    age = st.text_input("Âge (optionnel)")

report = build_report(short_totals, re_scores, HR, ER, HE, EE, name, age, bmri_result=bmri_result)
st.download_button("💾 Télécharger le rapport (.md)", report,
                   file_name="bilan_hpe_rapport.md", mime="text/markdown")

raw = {}
for k, v in short_totals.items():
    raw[k] = v
for k, v in re_scores.items():
    raw[k] = v["score"]
raw.update({"HR": HR, "ER": ER, "HE": HE, "EE": EE})
if bmri_result is not None:
    for k, v in bmri_result["choices"].items():
        raw[k] = v
st.download_button("⬇️ Exporter les réponses (.csv)",
                   pd.DataFrame([raw]).to_csv(index=False).encode("utf-8"),
                   file_name="bilan_hpe_reponses.csv", mime="text/csv")
