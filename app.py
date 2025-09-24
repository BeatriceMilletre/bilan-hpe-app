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

thresholds = data.get("thresholds", {})  # voir section thresholds dans questionnaire.yml

st.title("🧠 Bilan HPE – Passation (YAML)")
st.caption("Les items et barèmes viennent de `questionnaire.yml` (modifiez-le pour faire évoluer le test).")

with st.sidebar:
    st.header("Infos répondant")
    name = st.text_input("Nom (optionnel)")
    age = st.text_input("Âge (optionnel)")
    st.markdown("---")
    st.write("✏️ Pour changer les questions/barèmes/seuils : éditez `questionnaire.yml`.")

# ---------- Outils ----------
DEFAULT_LIKERT4 = ["Tout à fait d’accord","Plutôt d’accord","Plutôt pas d’accord","Pas du tout d’accord"]

def categorize(value, thres: dict):
    """Renvoie 'faible' / 'moyen' / 'élevé' selon les seuils du YAML."""
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

# ---------- UI blocs ----------
def ask_block_4pt(block: dict):
    labels = block.get("scale_labels", DEFAULT_LIKERT4)
    out = {}
    st.subheader(block.get("key", "Échelle"))
    for it in block.get("items", []):
        c1, c2 = st.columns([3, 2])
        with c1:
            st.write(f"**{it['id']}** — {it.get('text','')}")
        with c2:
            choice = st.radio(
                it["id"], options=list(range(4)), index=0, horizontal=True,
                label_visibility="collapsed", format_func=lambda i: labels[i]
            )
        out[it["id"]] = it["values"][choice]
        st.divider()
    return out

def ask_block_re(block: dict):
    st.subheader("Échelle Rationnelle / Expérientielle (1–5)")
    st.caption("Reverse appliqué si `reverse: true` dans questionnaire.yml (1↔5).")
    scores, bad = {}, []
    for idx, it in enumerate(block.get("items", []), start=1):
        if not isinstance(it, dict):
            bad.append((idx, f"type={type(it).__name__} -> {it!r}"))
            continue
        iid = it.get("id"); tag = it.get("tag"); text = it.get("text", "")
        if not iid or not tag:
            bad.append((idx, f"id={iid!r}, tag={tag!r}, text={text[:40]!r}"))
            continue
        val = st.slider(f"{iid} – {tag} : {text}", 1, 5, 3, key=f"rei-{iid}")
        if it.get("reverse", False):
            val = 6 - val
        scores[str(iid)] = {"tag": str(tag).upper(), "score": float(val)}
    if bad:
        st.warning(f"Certains items RE-40 ont été ignorés (id/tag manquants) : {len(bad)}. "
                   f"Exemple de problème : {bad[0]}")
    return scores

# ---------- Passation ----------
short_totals = {}
re_scores = {}
for block in data.get("blocks", []):
    if block.get("type") == "re":
        re_scores = ask_block_re(block)
    else:
        answers = ask_block_4pt(block)
        short_totals[block["key"]] = sum(answers.values())

# ---------- Scores R/E ----------
def mean_tag(tag: str):
    vals = [v["score"] for v in re_scores.values() if v["tag"] == tag]
    return round(float(np.mean(vals)), 2) if vals else None

HR = mean_tag("HR"); ER = mean_tag("ER"); HE = mean_tag("HE"); EE = mean_tag("EE")

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
    for lab, val in [("HR", HR), ("ER", ER), ("HE", HE), ("EE", EE)]:
        cat = categorize(val, thresholds.get("re_scales", {}).get(lab, {}))
        rows.append({"Sous-échelle": lab, "Score": val, "Catégorie": cat})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

# ---------- Rapport & Exports ----------
def build_report():
    L = []
    L.append("# Bilan HPE – Rapport (YAML)")
    L.append(f"**Date** : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    meta = " – ".join(
        x for x in [f"Nom: {name}" if name else None, f"Âge: {age}" if age else None] if x
    )
    if meta:
        L.append(meta)
    L.append("")
    L.append("## Scores – courts")
    for k, v in short_totals.items():
        cat = categorize(v, thresholds.get("short_scales", {}).get(k, {}))
        L.append(f"- {k} : **{v:.0f}** ({cat})")
    L.append("")
    L.append("## Rationnel / Expérientiel (1–5)")
    for lab, val in [("HR", HR), ("ER", ER), ("HE", HE), ("EE", EE)]:
        cat = categorize(val, thresholds.get("re_scales", {}).get(lab, {}))
        sval = "—" if val is None else f"{val:.2f}"
        L.append(f"- {lab} : **{sval}** ({cat})")
    L.append("")
    L.append("## Synthèse R/E")
    L.append(synthese_re(HR, ER, HE, EE, thresholds.get("re_scales", {})))
    L.append("")
    L.append("> Note : seuils configurables dans `questionnaire.yml` → `thresholds`.")
    return "\n".join(L)

report = build_report()
st.download_button("💾 Télécharger le rapport (.md)", report,
                   file_name="bilan_hpe_rapport.md", mime="text/markdown")

raw = {}
for k, v in short_totals.items():
    raw[k] = v
for k, v in re_scores.items():
    raw[k] = v["score"]
raw.update({"HR": HR, "ER": ER, "HE": HE, "EE": EE})
st.download_button("⬇️ Exporter les réponses (.csv)",
                   pd.DataFrame([raw]).to_csv(index=False).encode("utf-8"),
                   file_name="bilan_hpe_reponses.csv", mime="text/csv")

st.success("OK. Classement faible/moyen/élevé ajouté + synthèse automatique dans le rapport.")
