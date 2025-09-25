# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import yaml
import io
import os
import random
import smtplib
from email.message import EmailMessage
from datetime import datetime
from pathlib import Path

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
st.set_page_config(page_title="Bilan HPE – Questionnaire", page_icon="🧭", layout="wide")

YAML_PATH = Path("questionnaire.yml")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

DEFAULT_LIKERT4 = ["Tout à fait d’accord","Plutôt d’accord","Plutôt pas d’accord","Pas du tout d’accord"]

# ------------------------------------------------------------
# Chargement YAML
# ------------------------------------------------------------
if not YAML_PATH.exists():
    st.error("Le fichier 'questionnaire.yml' est introuvable à la racine du dépôt.")
    st.stop()

with YAML_PATH.open("r", encoding="utf-8") as f:
    data = yaml.safe_load(f)

thresholds = data.get("thresholds", {})

# ------------------------------------------------------------
# Utilitaires
# ------------------------------------------------------------
def make_recovery_code(n=6):
    # Code lisible type A1BF01
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    return "".join(random.choice(alphabet) for _ in range(n))

def categorize(value, thres: dict):
    if value is None: return "—"
    if value >= thres.get("high", 9e9): return "élevé"
    if value >= thres.get("medium", -9e9): return "moyen"
    return "faible"

def save_results(code: str, report_md: str, csv_bytes: bytes):
    folder = RESULTS_DIR / code
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "rapport.md").write_text(report_md, encoding="utf-8")
    (folder / "reponses.csv").write_bytes(csv_bytes)
    return folder

def send_code_to_practitioner(code: str, practitioner_email: str):
    sender = st.secrets.get("EMAIL_SENDER")
    app_pwd = st.secrets.get("EMAIL_APP_PASSWORD")
    if not sender or not app_pwd:
        st.warning("EMAIL_SENDER / EMAIL_APP_PASSWORD non configurés dans Secrets : l’e-mail ne sera pas envoyé.")
        return False

    msg = EmailMessage()
    msg["From"] = sender
    msg["To"] = practitioner_email
    msg["Subject"] = "Nouveau résultat – Code de récupération"
    msg.set_content(
        f"Bonjour,\n\nUn répondant vient de terminer le questionnaire.\n"
        f"Code de récupération : {code}\n\n"
        "Pour récupérer les fichiers, utilisez l’onglet « Accès praticien » et saisissez ce code.\n\n"
        "Bien cordialement,\nBilan HPE"
    )
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(sender, app_pwd)
            smtp.send_message(msg)
        return True
    except Exception as e:
        st.error(f"Échec d’envoi de l’e-mail praticien : {e}")
        return False

def synthese_re(hr, er, he, ee, thres_re: dict):
    parts = []
    for lab, val in [("HR",hr),("ER",er),("HE",he),("EE",ee)]:
        cat = categorize(val, thres_re.get(lab, {}))
        txt = (
            {"élevé":"forte aisance/motivation analytique",
             "moyen":"ressources rationnelles présentes",
             "faible":"habileté analytique perçue plus basse"} if lab in ["HR","ER"]
            else
            {"élevé":"recours fréquent à l’intuition/ressenti",
             "moyen":"équilibre intuition/logique",
             "faible":"intuition moins mobilisée"}
        ).get(cat, "—")
        sval = "—" if val is None else f"{val:.2f}"
        parts.append(f"- {lab} **{sval}** ({cat}) : {txt}.")
    return "\n".join(parts)

# ------------------------------------------------------------
# UI widgets (Likert / REI / Forced-choice)
# ------------------------------------------------------------
def ask_block_likert(block: dict):
    labels = block.get("scale_labels", DEFAULT_LIKERT4)
    out = {}
    st.subheader(block.get("key", "Échelle"))
    for it in block.get("items", []):
        vals = it["values"]; n = len(vals)
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

def ask_block_re(block: dict):
    st.subheader("Échelle R/E (1–5)")
    scores = {}
    for it in block.get("items", []):
        iid, tag, text = it.get("id"), it.get("tag"), it.get("text","")
        val = st.slider(f"{iid} – {tag} : {text}", 1, 5, 3, key=f"rei-{iid}")
        if it.get("reverse", False): val = 6 - val
        scores[str(iid)] = {"tag": str(tag).upper(), "score": float(val)}
    return scores

def ask_block_forced_choice(block: dict):
    st.subheader(block.get("key","BMRI (A/B)"))
    choices = {}
    for idx, it in enumerate(block.get("items", []), 1):
        stem, a, b = it.get("stem",""), it.get("a",""), it.get("b","")
        st.markdown(f"**{it.get('id', f'FC{idx}')}** — {stem}")
        pick = st.radio(it.get("id", f"FC{idx}"), options=["A","B"], index=0, horizontal=True,
                        format_func=lambda x: f"{x}) {a if x=='A' else b}")
        choices[it.get("id", f"FC{idx}")] = pick
        st.divider()
    return {"choices": choices}

# ------------------------------------------------------------
# Rapport
# ------------------------------------------------------------
def build_report(short_totals, re_scores, HR, ER, HE, EE, name=None, age=None, bmri_result=None):
    short_cats = {k: categorize(v, thresholds.get("short_scales", {}).get(k, {}))
                  for k, v in short_totals.items()}
    L = []
    L.append("# Bilan HPE – Rapport")
    L.append(f"**Date** : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    meta = " – ".join(x for x in [f"Nom: {name}" if name else None, f"Âge: {age}" if age else None] if x)
    if meta: L.append(meta)
    L.append("")
    if short_totals:
        L.append("## Scores – courts")
        for k,v in short_totals.items():
            L.append(f"- {k} : **{v:.0f}** ({short_cats.get(k,'—')})")
        L.append("")
    L.append("## Rationnel / Expérientiel (1–5)")
    for lab, val in [("HR",HR),("ER",ER),("HE",HE),("EE",EE)]:
        sval = "—" if val is None else f"{val:.2f}"
        L.append(f"- {lab} : **{sval}**")
    L.append("")
    L.append("## Synthèse R/E")
    L.append(synthese_re(HR,ER,HE,EE, thresholds.get("re_scales", {})))
    return "\n".join(L)

def mean_tag(tag: str, re_scores):
    vals = [v["score"] for v in re_scores.values() if v["tag"] == tag]
    return round(float(np.mean(vals)), 2) if vals else None

# ------------------------------------------------------------
# Deux parcours : Passer le test / Accès praticien
# ------------------------------------------------------------
tab_test, tab_pro = st.tabs(["📝 Passer le test", "🔑 Accès praticien"])

# --------- 📝 Passer le test ----------
with tab_test:
    st.title("Questionnaire – Passer le test")

    short_totals, re_scores, bmri_result = {}, {}, None

    for block in data.get("blocks", []):
        btype = block.get("type")
        if btype == "re":
            re_scores = ask_block_re(block)
        elif btype == "forced_choice":
            bmri_result = ask_block_forced_choice(block)
        else:
            answers = ask_block_likert(block)
            short_totals[block["key"]] = sum(answers.values())

    HR = mean_tag("HR", re_scores); ER = mean_tag("ER", re_scores)
    HE = mean_tag("HE", re_scores); EE = mean_tag("EE", re_scores)

    st.header("Aperçu des résultats")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Questionnaires courts")
        if short_totals:
            rows = [{"Échelle":k, "Total":v, "Catégorie":categorize(v, thresholds.get("short_scales", {}).get(k, {}))}
                    for k,v in short_totals.items()]
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else:
            st.caption("Aucun questionnaire court dans ce YAML.")
    with c2:
        st.subheader("R/E (1–5)")
        rows = [{"Sous-échelle":lab, "Score":val} for lab, val in [("HR",HR),("ER",ER),("HE",HE),("EE",EE)]]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # Identité (optionnel)
    with st.sidebar:
        st.header("Infos répondant")
        name = st.text_input("Nom (optionnel)")
        age = st.text_input("Âge (optionnel)")

    # Prépare les fichiers (mais NE LES PROPOSE PAS au téléchargement ici)
    report = build_report(short_totals, re_scores, HR, ER, HE, EE, name=name, age=age, bmri_result=bmri_result)
    raw = {}
    for k, v in short_totals.items(): raw[k] = v
    for k, v in re_scores.items():    raw[k] = v["score"]
    raw.update({"HR":HR, "ER":ER, "HE":HE, "EE":EE})
    if bmri_result:
        for k, v in bmri_result["choices"].items(): raw[k] = v
    csv_bytes = pd.DataFrame([raw]).to_csv(index=False).encode("utf-8")

    # ---------- Envoi final (automatique vers PRACTITIONER_EMAIL) ----------
    st.markdown("---")
    st.subheader("🧾 Validation & envoi au praticien")

    target_email = st.secrets.get("PRACTITIONER_EMAIL")
    if not target_email:
        st.warning(
            "Adresse praticien non configurée : ajoute `PRACTITIONER_EMAIL` dans les Secrets.\n"
            "L’email ne sera pas envoyé mais le code sera affiché et les fichiers seront sauvegardés."
        )

    if st.button("Envoyer"):
        code = make_recovery_code()
        save_results(code, report, csv_bytes)
        sent = False
        if target_email:
            sent = send_code_to_practitioner(code, target_email)

        st.session_state["last_recovery_code"] = code
        if sent:
            st.info("📧 Un email a été envoyé automatiquement au praticien avec le code de récupération.")
        st.success(f"Résultats enregistrés avec le code **{code}** (à communiquer au praticien).")

# --------- 🔑 Accès praticien ----------
with tab_pro:
    st.title("Accès praticien")
    st.write("Entrez le code communiqué par le répondant pour récupérer les fichiers.")
    code_in = st.text_input("Code praticien (ex: A1BF01)").strip().upper()
    if st.button("Récupérer"):
        if not code_in:
            st.warning("Saisissez un code.")
        else:
            folder = RESULTS_DIR / code_in
            md_path = folder / "rapport.md"
            csv_path = folder / "reponses.csv"
            if md_path.exists() or csv_path.exists():
                st.success(f"Résultats trouvés pour le code **{code_in}**.")
                col1, col2 = st.columns(2)
                if md_path.exists():
                    st.download_button("💾 Télécharger le rapport (.md)",
                                       data=md_path.read_text(encoding="utf-8"),
                                       file_name=f"rapport_{code_in}.md",
                                       mime="text/markdown")
                if csv_path.exists():
                    st.download_button("⬇️ Télécharger les réponses (.csv)",
                                       data=csv_path.read_bytes(),
                                       file_name=f"reponses_{code_in}.csv",
                                       mime="text/csv")
            else:
                st.error("Aucun résultat pour ce code. Vérifiez l’orthographe ou réessayez plus tard.")
