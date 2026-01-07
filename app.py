# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import yaml
import os
import random
import smtplib
import ssl
import json
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

DEFAULT_LIKERT4 = ["Tout à fait d’accord", "Plutôt d’accord", "Plutôt pas d’accord", "Pas du tout d’accord"]

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
    # Code lisible type A1BF01 (sans O/0/I/1 ambigu)
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    return "".join(random.choice(alphabet) for _ in range(n))

def categorize(value, thres: dict):
    if value is None:
        return "—"
    if value >= thres.get("high", 9e9):
        return "élevé"
    if value >= thres.get("medium", -9e9):
        return "moyen"
    return "faible"

def save_results(code: str, report_md: str, csv_bytes: bytes):
    folder = RESULTS_DIR / code
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "rapport.md").write_text(report_md, encoding="utf-8")
    (folder / "reponses.csv").write_bytes(csv_bytes)
    return folder

def synthese_re(hr, er, he, ee, thres_re: dict):
    parts = []
    for lab, val in [("HR", hr), ("ER", er), ("HE", he), ("EE", ee)]:
        cat = categorize(val, thres_re.get(lab, {}))
        txt = (
            {"élevé": "forte aisance/motivation analytique",
             "moyen": "ressources rationnelles présentes",
             "faible": "habileté analytique perçue plus basse"} if lab in ["HR", "ER"]
            else
            {"élevé": "recours fréquent à l’intuition/ressenti",
             "moyen": "équilibre intuition/logique",
             "faible": "intuition moins mobilisée"}
        ).get(cat, "—")
        sval = "—" if val is None else f"{val:.2f}"
        parts.append(f"- {lab} **{sval}** ({cat}) : {txt}.")
    return "\n".join(parts)

def mean_tag(tag: str, re_scores):
    vals = [v["score"] for v in re_scores.values() if v["tag"] == tag]
    return round(float(np.mean(vals)), 2) if vals else None

# ------------------------------------------------------------
# EMAIL : JSON joint (+ optionnel CSV/MD)
# ------------------------------------------------------------
def send_email_with_attachments(code: str, payload: dict, report_md: str, csv_bytes: bytes) -> tuple[bool, str]:
    """
    Envoi email au praticien avec pièce jointe JSON (et optionnellement MD/CSV).
    Secrets attendus :
      - PRACTITIONER_EMAIL
      - EMAIL_SENDER
      - EMAIL_APP_PASSWORD
    Envoie via Gmail SMTP SSL 465 (comme ton code initial).
    """
    sender = st.secrets.get("EMAIL_SENDER")
    app_pwd = st.secrets.get("EMAIL_APP_PASSWORD")
    practitioner_email = st.secrets.get("PRACTITIONER_EMAIL")

    if not practitioner_email:
        return False, "PRACTITIONER_EMAIL manquant dans Secrets."
    if not sender or not app_pwd:
        return False, "EMAIL_SENDER / EMAIL_APP_PASSWORD manquants dans Secrets."

    try:
        msg = EmailMessage()
        msg["From"] = sender
        msg["To"] = practitioner_email
        msg["Subject"] = f"[Bilan HPE] Nouvelle passation ({code})"

        submitted_at = payload.get("meta", {}).get("submitted_at", "")
        name = payload.get("respondent", {}).get("name", "")
        age = payload.get("respondent", {}).get("age", "")

        msg.set_content(
            "Une nouvelle passation du questionnaire Bilan HPE a été complétée.\n\n"
            f"Code : {code}\n"
            f"Date : {submitted_at}\n"
            f"Nom : {name}\n"
            f"Âge : {age}\n\n"
            "Les données complètes (avec énoncés) sont jointes au format JSON.\n"
            "Vous trouverez aussi, si disponible, le rapport Markdown et le CSV en pièces jointes.\n"
        )

        # JSON joint (source de vérité)
        json_bytes = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        msg.add_attachment(
            json_bytes,
            maintype="application",
            subtype="json",
            filename=f"bilan_hpe_{code}.json",
        )

        # Optionnels mais pratiques
        if report_md:
            msg.add_attachment(
                report_md.encode("utf-8"),
                maintype="text",
                subtype="markdown",
                filename=f"rapport_{code}.md",
            )

        if csv_bytes:
            msg.add_attachment(
                csv_bytes,
                maintype="text",
                subtype="csv",
                filename=f"reponses_{code}.csv",
            )

        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=20) as smtp:
            smtp.login(sender, app_pwd)
            smtp.send_message(msg)

        return True, "Email envoyé (JSON joint)."
    except Exception as e:
        return False, f"Échec d’envoi email : {e}"

# ------------------------------------------------------------
# UI widgets (Likert / REI / Forced-choice)
# ------------------------------------------------------------
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
                it["id"],
                options=list(range(n)),
                index=0,
                horizontal=True,
                label_visibility="collapsed",
                format_func=lambda i: local_labels[i],
                key=f"likert-{block.get('key','')}-{it['id']}",
            )
        out[it["id"]] = vals[choice]
        st.divider()
    return out

def ask_block_re(block: dict):
    st.subheader("Échelle R/E (1–5)")
    scores = {}
    for it in block.get("items", []):
        iid, tag, text = it.get("id"), it.get("tag"), it.get("text", "")
        val = st.slider(f"{iid} – {tag} : {text}", 1, 5, 3, key=f"rei-{iid}")
        if it.get("reverse", False):
            val = 6 - val
        scores[str(iid)] = {"tag": str(tag).upper(), "score": float(val)}
    return scores

def ask_block_forced_choice(block: dict):
    st.subheader(block.get("key", "BMRI (A/B)"))
    choices = {}
    for idx, it in enumerate(block.get("items", []), 1):
        stem, a, b = it.get("stem", ""), it.get("a", ""), it.get("b", "")
        st.markdown(f"**{it.get('id', f'FC{idx}')}** — {stem}")
        pick = st.radio(
            it.get("id", f"FC{idx}"),
            options=["A", "B"],
            index=0,
            horizontal=True,
            format_func=lambda x: f"{x}) {a if x=='A' else b}",
            key=f"fc-{it.get('id', f'FC{idx}')}",
        )
        choices[it.get("id", f"FC{idx}")] = pick
        st.divider()
    return {"choices": choices}

# ------------------------------------------------------------
# Rapport
# ------------------------------------------------------------
def build_report(short_totals, re_scores, HR, ER, HE, EE, name=None, age=None, bmri_result=None):
    short_cats = {
        k: categorize(v, thresholds.get("short_scales", {}).get(k, {}))
        for k, v in short_totals.items()
    }
    L = []
    L.append("# Bilan HPE – Rapport")
    L.append(f"**Date** : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    meta = " – ".join(x for x in [f"Nom: {name}" if name else None, f"Âge: {age}" if age else None] if x)
    if meta:
        L.append(meta)
    L.append("")
    if short_totals:
        L.append("## Scores – courts")
        for k, v in short_totals.items():
            L.append(f"- {k} : **{v:.0f}** ({short_cats.get(k, '—')})")
        L.append("")
    L.append("## Rationnel / Expérientiel (1–5)")
    for lab, val in [("HR", HR), ("ER", ER), ("HE", HE), ("EE", EE)]:
        sval = "—" if val is None else f"{val:.2f}"
        L.append(f"- {lab} : **{sval}**")
    L.append("")
    L.append("## Synthèse R/E")
    L.append(synthese_re(HR, ER, HE, EE, thresholds.get("re_scales", {})))

    if bmri_result and bmri_result.get("choices"):
        L.append("")
        L.append("## Forced-choice (BMRI) – réponses")
        for k, v in bmri_result["choices"].items():
            L.append(f"- {k} : {v}")

    L.append("")
    L.append("_Outil d’orientation : ce questionnaire n’est pas un diagnostic._")
    return "\n".join(L)

# ------------------------------------------------------------
# Construire payload JSON COMPLET (avec énoncés)
# ------------------------------------------------------------
def build_payload(code: str,
                  name: str,
                  age: str,
                  short_totals: dict,
                  re_scores: dict,
                  HR, ER, HE, EE,
                  bmri_result: dict | None):
    """
    JSON complet, incluant :
    - YAML brut (blocks) => énoncés / items
    - réponses structurées par bloc
    - scores calculés
    - seuils utilisés
    """
    submitted_at = datetime.now().isoformat()

    # réponses détaillées Likert : on reconstruit un dict item_id -> valeur brute (celle du YAML 'values')
    # et un dict totals courts.
    short_cats = {
        k: categorize(v, thresholds.get("short_scales", {}).get(k, {}))
        for k, v in short_totals.items()
    }

    payload = {
        "questionnaire": "Bilan HPE",
        "meta": {
            "code": code,
            "submitted_at": submitted_at,
            "source_yaml": str(YAML_PATH),
        },
        "respondent": {
            "name": name or "",
            "age": age or "",
        },
        "thresholds": thresholds,          # seuils utilisés
        "structure": {
            "blocks": data.get("blocks", [])  # contient les énoncés (items)
        },
        "results": {
            "short_totals": short_totals,
            "short_categories": short_cats,
            "re_scores_items": re_scores,  # par item (tag/score)
            "HR": HR, "ER": ER, "HE": HE, "EE": EE,
            "bmri": bmri_result or {},
        }
    }
    return payload

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
            # total "court" = somme des valeurs codées du YAML
            short_totals[block["key"]] = sum(answers.values())

    HR = mean_tag("HR", re_scores)
    ER = mean_tag("ER", re_scores)
    HE = mean_tag("HE", re_scores)
    EE = mean_tag("EE", re_scores)

    st.header("Aperçu des résultats")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Questionnaires courts")
        if short_totals:
            rows = [{
                "Échelle": k,
                "Total": v,
                "Catégorie": categorize(v, thresholds.get("short_scales", {}).get(k, {}))
            } for k, v in short_totals.items()]
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else:
            st.caption("Aucun questionnaire court dans ce YAML.")
    with c2:
        st.subheader("R/E (1–5)")
        rows = [{"Sous-échelle": lab, "Score": val} for lab, val in [("HR", HR), ("ER", ER), ("HE", HE), ("EE", EE)]]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # Identité (optionnel)
    with st.sidebar:
        st.header("Infos répondant")
        name = st.text_input("Nom (optionnel)")
        age = st.text_input("Âge (optionnel)")

    # Prépare fichiers
    report = build_report(short_totals, re_scores, HR, ER, HE, EE, name=name, age=age, bmri_result=bmri_result)

    # CSV plat (comme avant) : 1 ligne
    raw = {}
    for k, v in short_totals.items():
        raw[k] = v
    for k, v in re_scores.items():
        raw[k] = v["score"]
    raw.update({"HR": HR, "ER": ER, "HE": HE, "EE": EE})
    if bmri_result and bmri_result.get("choices"):
        for k, v in bmri_result["choices"].items():
            raw[k] = v
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

        # 1) Sauvegarde best-effort sur disque
        save_results(code, report, csv_bytes)

        # 2) Payload JSON complet (incluant TOUS les items du YAML)
        payload = build_payload(
            code=code,
            name=name,
            age=age,
            short_totals=short_totals,
            re_scores=re_scores,
            HR=HR, ER=ER, HE=HE, EE=EE,
            bmri_result=bmri_result
        )

        # 3) Email avec JSON joint (+ MD/CSV)
        sent = False
        if target_email:
            sent, msg = send_email_with_attachments(code, payload, report_md=report, csv_bytes=csv_bytes)
            if sent:
                st.info("📧 Un email a été envoyé automatiquement au praticien avec le JSON en pièce jointe.")
            else:
                st.warning("Email non envoyé automatiquement.")
                st.caption(msg)
                with st.expander("Voir le JSON (à copier si besoin)", expanded=False):
                    st.json(payload)

        st.session_state["last_recovery_code"] = code
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
                    st.download_button(
                        "💾 Télécharger le rapport (.md)",
                        data=md_path.read_text(encoding="utf-8"),
                        file_name=f"rapport_{code_in}.md",
                        mime="text/markdown"
                    )
                if csv_path.exists():
                    st.download_button(
                        "⬇️ Télécharger les réponses (.csv)",
                        data=csv_path.read_bytes(),
                        file_name=f"reponses_{code_in}.csv",
                        mime="text/csv"
                    )
            else:
                st.error("Aucun résultat pour ce code. Vérifiez l’orthographe ou réessayez plus tard.")

st.markdown("---")
st.caption(
    "⚠️ Confidentialité : ce prototype enregistre des résultats sur le disque (dossier 'results/'). "
    "Sur Streamlit Cloud, ce stockage est temporaire. "
    "Le canal fiable est l’e-mail (JSON joint)."
)
