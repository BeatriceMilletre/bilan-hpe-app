# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import yaml
from datetime import datetime
from pathlib import Path

st.set_page_config(page_title="Bilan HPE – Passation (YAML)", page_icon="🧠", layout="centered")

YAML_PATH = Path("questionnaire.yml")

st.title("🧠 Bilan HPE – Passation (YAML)")
st.caption("Les items et barèmes viennent du fichier questionnaire.yml à la racine du dépôt.")

# -------- Charger le YAML --------
if not YAML_PATH.exists():
    st.error("Le fichier 'questionnaire.yml' est introuvable à la racine du dépôt.")
    st.stop()
with YAML_PATH.open("r", encoding="utf-8") as f:
    data = yaml.safe_load(f)
# Pré-contrôle du bloc RE-40
for block in data.get("blocks", []):
    if block.get("key") == "RE-40":
        bads = []
        for i, it in enumerate(block.get("items", []), 1):
            if not isinstance(it, dict) or "id" not in it or "tag" not in it:
                bads.append((i, it))
        if bads:
            st.info(f"Diagnostic RE-40: {len(bads)} entrée(s) à corriger. Exemple: {bads[0]}")
        break

with st.sidebar:
    st.header("Infos répondant")
    name = st.text_input("Nom (optionnel)")
    age = st.text_input("Âge (optionnel)")
    st.markdown("---")
    st.write("✏️ Pour modifier les questions, éditez `questionnaire.yml` dans GitHub.")

DEFAULT_LIKERT4 = ["Tout à fait d’accord","Plutôt d’accord","Plutôt pas d’accord","Pas du tout d’accord"]

def ask_block_4pt(block):
    labels = block.get("scale_labels", DEFAULT_LIKERT4)
    out = {}
    st.subheader(block["key"])
    for it in block["items"]:
        c1, c2 = st.columns([3,2])
        with c1:
            st.write(f"**{it['id']}** — {it['text']}")
        with c2:
            choice = st.radio(it["id"], options=list(range(4)), index=0,
                              horizontal=True, label_visibility="collapsed",
                              format_func=lambda i: labels[i])
        out[it["id"]] = it["values"][choice]
        st.divider()
    return out

def ask_block_re(block):
    st.subheader("Échelle Rationnelle / Expérientielle (1–5)")
    st.caption("Reverse appliqué si `reverse: true` dans questionnaire.yml (1↔5).")
    scores, bad = {}, []
    for idx, it in enumerate(block.get("items", []), start=1):
        # robustesse: chaque entrée doit être un dict avec id/tag/text
        if not isinstance(it, dict):
            bad.append((idx, f"type={type(it).__name__} -> {it!r}"))
            continue
        iid = it.get("id"); tag = it.get("tag"); text = it.get("text", "")
        if not iid or not tag:
            bad.append((idx, f"id={iid!r}, tag={tag!r}, text={text[:40]!r}"))
            continue
        # clé unique pour Streamlit
        widget_key = f"rei-{iid}"
        val = st.slider(f"{iid} – {tag} : {text}", 1, 5, 3, key=widget_key)
        if it.get("reverse", False):
            val = 6 - val
        scores[str(iid)] = {"tag": str(tag).upper(), "score": float(val)}
    if bad:
        st.warning(
            "⚠️ Certains items RE-40 ont été ignorés car il manque `id` ou `tag` "
            f"({len(bad)} au total). Premier problème repéré : " + str(bad[0])
        )
    return scores


# -------- Passation --------
short_totals = {}
re_scores = {}
for block in data.get("blocks", []):
    if block.get("type") == "re":
        re_scores = ask_block_re(block)
    else:
        answers = ask_block_4pt(block)
        short_totals[block["key"]] = sum(answers.values())

# -------- Scores R/E --------
def mean_tag(tag):
    vals = [v["score"] for v in re_scores.values() if v["tag"] == tag]
    return round(float(np.mean(vals)), 2) if vals else None

HR = mean_tag("HR"); ER = mean_tag("ER"); HE = mean_tag("HE"); EE = mean_tag("EE")

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

# -------- Résultats --------
st.header("Résultats")
c1, c2 = st.columns(2)

with c1:
    st.subheader("Synthèse questionnaires courts")
    df_short = pd.DataFrame([{"Échelle": k, "Total": v} for k, v in short_totals.items()])
    st.dataframe(df_short, use_container_width=True)

with c2:
    st.subheader("Rationnel / Expérientiel (1–5)")
    df_re = pd.DataFrame([
        {"Sous-échelle":"HR", "Score":HR, "Interprétation":interp_re("HR",HR)},
        {"Sous-échelle":"ER", "Score":ER, "Interprétation":interp_re("ER",ER)},
        {"Sous-échelle":"HE", "Score":HE, "Interprétation":interp_re("HE",HE)},
        {"Sous-échelle":"EE", "Score":EE, "Interprétation":interp_re("EE",EE)},
    ])
    st.dataframe(df_re, use_container_width=True)

# -------- Exports --------
def build_report():
    L = []
    L.append("# Bilan HPE – Rapport (YAML)")
    L.append(f"**Date** : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    meta = " – ".join([x for x in [f'Nom: {name}' if name else None, f'Âge: {age}' if age else None] if x])
    if meta: L.append(meta)
    L.append("")
    L.append("## Scores – courts")
    for k, v in short_totals.items():
        L.append(f"- {k} : **{v:.0f}**")
    L.append("")
    L.append("## Rationnel / Expérientiel (1–5)")
    for lab, val in [("HR",HR),("ER",ER),("HE",HE),("EE",EE)]:
        sval = "—" if val is None else f"{val:.2f}"
        L.append(f"- {lab} : **{sval}** – {interp_re(lab, val)}")
    return "\n".join(L)

report = build_report()
st.download_button("💾 Télécharger le rapport (.md)", report, file_name="bilan_hpe_rapport.md", mime="text/markdown")

raw = {}
for k, v in short_totals.items(): raw[k] = v
for k, v in re_scores.items(): raw[k] = v["score"]
raw.update({"HR":HR,"ER":ER,"HE":HE,"EE":EE})
st.download_button(
    "⬇️ Exporter les réponses (.csv)",
    pd.DataFrame([raw]).to_csv(index=False).encode("utf-8"),
    file_name="bilan_hpe_reponses.csv",
    mime="text/csv"
)

st.success("OK. Les questions proviennent de `questionnaire.yml`.")
