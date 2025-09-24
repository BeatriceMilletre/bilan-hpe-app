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

# ---------- Outils ----------
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

def compte_rendu_auto(short_totals, re_cats, name=None, age=None):
    """
    Génère un récit en français à partir :
    - short_totals: dict { 'SPQ-10': int, 'EQ-10': int, 'Q-R-10': int, 'QA-10': int }
    - re_cats: dict {'HR': ('score','cat'), ...}
    """
    # Catégories courtes
    spq_cat = re_cats.get("_SPQ_cat", "—")
    eq_cat  = re_cats.get("_EQ_cat", "—")
    qr_cat  = re_cats.get("_QR_cat", "—")
    qa_cat  = re_cats.get("_QA_cat", "—")

    hr_s, hr_c = re_cats.get("HR", (None,"—"))
    er_s, er_c = re_cats.get("ER", (None,"—"))
    he_s, he_c = re_cats.get("HE", (None,"—"))
    ee_s, ee_c = re_cats.get("EE", (None,"—"))

    lignes = []
    # En-tête
    entete = "## Compte rendu automatique"
    lignes.append(entete)
    sous = []
    if name: sous.append(f"Nom : **{name}**")
    if age:  sous.append(f"Âge : **{age}**")
    sous.append(f"Date : **{datetime.now().strftime('%Y-%m-%d %H:%M')}**")
    lignes.append(" — ".join(sous))
    lignes.append("")

    # 1) Vue d’ensemble
    lignes.append("### 1) Vue d’ensemble")
    phrases = []
    # Profil R/E global
    if he_c == "élevé" and (er_c in ["élevé","moyen"]):
        phrases.append("Profil **intuitif engagé** : recours fréquent au ressenti, avec un bon appétit pour l’analyse.")
    if (hr_c == "faible") and (er_c in ["élevé","moyen"]):
        phrases.append("Motivation pour raisonner présente, mais **sentiment d’habileté analytique plus bas**.")
    if hr_c == "élevé" and he_c == "élevé":
        phrases.append("Double appui **logique + intuition** : alternance flexible selon les contextes.")
    if not phrases:
        phrases.append("Répartition des préférences R/E **équilibrée** ou variable selon les situations.")
    lignes.append("- " + " ".join(phrases))

    # 2) Détails par domaines (courts)
    lignes.append("")
    lignes.append("### 2) Indicateurs spécifiques (questionnaires courts)")
    # SPQ
    if spq_cat == "élevé":
        lignes.append("- **SPQ-10 élevé** : sensibilité sensorielle marquée ; soigner l’hygiène des environnements (lumières, bruit, odeurs).")
    elif spq_cat == "moyen":
        lignes.append("- **SPQ-10 moyen** : sensibilité présente mais gérable selon les contextes.")
    elif spq_cat == "faible":
        lignes.append("- **SPQ-10 faible** : peu d’interférences sensorielles rapportées.")
    # EQ
    if eq_cat == "élevé":
        lignes.append("- **EQ-10 élevé** : compréhension fine d’autrui ; atout pour la communication et l’accompagnement.")
    elif eq_cat == "moyen":
        lignes.append("- **EQ-10 moyen** : empathie adéquate ; adapter au contexte social.")
    else:
        lignes.append("- **EQ-10 faible** : repérage émotionnel plus difficile ; expliciter les signaux sociaux peut aider.")
    # Q-R
    if qr_cat == "élevé":
        lignes.append("- **Q-R-10 élevé** : intérêt fort pour les structures, théories et explications détaillées.")
    elif qr_cat == "moyen":
        lignes.append("- **Q-R-10 moyen** : alternance entre vision d’ensemble et détails selon l’intérêt.")
    else:
        lignes.append("- **Q-R-10 faible** : préférence pour le concret et le pratico-pratique.")
    # QA
    if qa_cat == "élevé":
        lignes.append("- **QA-10 élevé** : accords fréquents avec les items ciblés ; surveiller la charge attentionnelle.")
    elif qa_cat == "moyen":
        lignes.append("- **QA-10 moyen** : attention globalement fonctionnelle, variable selon l’environnement.")
    else:
        lignes.append("- **QA-10 faible** : peu de difficultés auto-rapportées sur ces items spécifiques.")

    # 3) Forces & points de vigilance (R/E)
    lignes.append("")
    lignes.append("### 3) Rationnel / Expérientiel – forces & points de vigilance")
    def phrase_cat(lab, s, c):
        if s is None: return None
        if c == "élevé":
            if lab in ["HR","ER"]:
                return f"- **{lab} élevé ({s:.2f})** : structuration, analyse, appétence pour le raisonnement."
            else:
                return f"- **{lab} élevé ({s:.2f})** : intuition vive, premières impressions utiles."
        if c == "moyen":
            return f"- **{lab} moyen ({s:.2f})** : équilibre, style adaptable."
        return f"- **{lab} faible ({s:.2f})** : à soutenir/structurer selon les tâches."
    for lab, (s,c) in {"HR":(hr_s,hr_c),"ER":(er_s,er_c),"HE":(he_s,he_c),"EE":(ee_s,ee_c)}.items():
        p = phrase_cat(lab, s, c)
        if p: lignes.append(p)

    # 4) Recommandations opérationnelles
    lignes.append("")
    lignes.append("### 4) Pistes concrètes")
    recos = []
    if he_c == "élevé":
        recos.append("- Utiliser l’intuition comme **hypothèse initiale** puis la valider avec 2–3 critères objectifs.")
    if hr_c in ["faible","moyen"]:
        recos.append("- Systématiser des **mini-cadres** (5-Pourquoi, tableau Avantages/Risques, checklist de décision).")
    if spq_cat == "élevé":
        recos.append("- **Hygiène sensorielle** : réduire bruit/néons/odeurs lors de tâches d’analyse.")
    if er_c == "élevé":
        recos.append("- Canaliser l’appétence pour les problèmes complexes en **blocs de travail** structurés (objectif, durée, livrable).")
    if not recos:
        recos.append("- Ajuster l’effort entre analyse et intuition selon l’enjeu et le temps disponible.")
    lignes.extend(recos)

    # 5) À discuter / limites
    lignes.append("")
    lignes.append("### 5) À discuter")
    lignes.append("- Les résultats sont **auto-rapportés** ; à croiser avec des observations et objectifs réels.")
    lignes.append("- Les catégories (faible/moyen/élevé) dépendent des **seuils configurés** ; ajustables.")

    return "\n".join(lignes)

def build_report(short_totals, re_scores, HR, ER, HE, EE, name, age):
    # Catégorisation
    short_cats = {
        k: categorize(v, thresholds.get("short_scales", {}).get(k, {}))
        for k, v in short_totals.items()
    }
    re_cats = {
        "HR": (HR, categorize(HR, thresholds.get("re_scales", {}).get("HR", {}))),
        "ER": (ER, categorize(ER, thresholds.get("re_scales", {}).get("ER", {}))),
        "HE": (HE, categorize(HE, thresholds.get("re_scales", {}).get("HE", {}))),
        "EE": (EE, categorize(EE, thresholds.get("re_scales", {}).get("EE", {}))),
        "_SPQ_cat": short_cats.get("SPQ-10", "—"),
        "_EQ_cat":  short_cats.get("EQ-10", "—"),
        "_QR_cat":  short_cats.get("Q-R-10", "—"),
        "_QA_cat":  short_cats.get("QA-10", "—"),
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
    L.append("")
    # Nouveau : Compte rendu automatique
    L.append(compte_rendu_auto(short_totals, re_cats, name=name, age=age))
    L.append("")
    L.append("> Note : seuils configurables dans `questionnaire.yml` → `thresholds`.")
    return "\n".join(L)

# ---------- Mode sélection ----------
mode = st.radio("Choisir le mode :", ["Passer le test", "Téléverser un fichier existant"])

if mode == "Passer le test":
    st.title("🧠 Passation du Bilan HPE")

    with st.sidebar:
        st.header("Infos répondant")
        name = st.text_input("Nom (optionnel)")
        age = st.text_input("Âge (optionnel)")

    # ---- Passation des blocs ----
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
        scores = {}
        for it in block.get("items", []):
            iid, tag, text = it.get("id"), it.get("tag"), it.get("text","")
            val = st.slider(f"{iid} – {tag} : {text}", 1, 5, 3, key=f"rei-{iid}")
            if it.get("reverse", False):
                val = 6 - val
            scores[str(iid)] = {"tag": str(tag).upper(), "score": float(val)}
        return scores

    short_totals = {}
    re_scores = {}
    for block in data.get("blocks", []):
        if block.get("type") == "re":
            re_scores = ask_block_re(block)
        else:
            answers = ask_block_4pt(block)
            short_totals[block["key"]] = sum(answers.values())

    def mean_tag(tag: str):
        vals = [v["score"] for v in re_scores.values() if v["tag"] == tag]
        return round(float(np.mean(vals)), 2) if vals else None

    HR = mean_tag("HR"); ER = mean_tag("ER"); HE = mean_tag("HE"); EE = mean_tag("EE")

    # ---- Résultats ----
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

    # ---- Compte rendu automatique (aperçu) ----
    # Catégories pour l'algorithme narratif
    re_cats = {
        "HR": (HR, categorize(HR, thresholds.get("re_scales", {}).get("HR", {}))),
        "ER": (ER, categorize(ER, thresholds.get("re_scales", {}).get("ER", {}))),
        "HE": (HE, categorize(HE, thresholds.get("re_scales", {}).get("HE", {}))),
        "EE": (EE, categorize(EE, thresholds.get("re_scales", {}).get("EE", {}))),
        "_SPQ_cat": categorize(short_totals.get("SPQ-10", 0), thresholds.get("short_scales", {}).get("SPQ-10", {})),
        "_EQ_cat":  categorize(short_totals.get("EQ-10", 0),  thresholds.get("short_scales", {}).get("EQ-10",  {})),
        "_QR_cat":  categorize(short_totals.get("Q-R-10", 0), thresholds.get("short_scales", {}).get("Q-R-10", {})),
        "_QA_cat":  categorize(short_totals.get("QA-10", 0),  thresholds.get("short_scales", {}).get("QA-10",  {})),
    }
    cr_auto = compte_rendu_auto(short_totals, re_cats, name=name, age=age)

    st.subheader("📝 Compte rendu automatique – aperçu")
    st.markdown(cr_auto)

    # ---- Exports ----
    report = build_report(short_totals, re_scores, HR, ER, HE, EE, name, age)
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

elif mode == "Téléverser un fichier existant":
    st.title("📂 Importer un rapport ou des réponses")
    uploaded = st.file_uploader("Déposez un fichier .csv (réponses brutes) ou .md (rapport)")
    if uploaded:
        if uploaded.name.endswith(".csv"):
            df = pd.read_csv(uploaded)
            st.subheader("Réponses importées")
            st.dataframe(df, use_container_width=True)
        elif uploaded.name.endswith(".md"):
            content = uploaded.read().decode("utf-8")
            st.subheader("Rapport importé")
            st.markdown(content)
        else:
            st.error("Format non reconnu. Utilisez .csv ou .md")
