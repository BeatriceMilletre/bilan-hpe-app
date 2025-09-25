# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import yaml
import matplotlib.pyplot as plt
import io
import smtplib
from email.message import EmailMessage
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
DEFAULT_LIKERT4 = ["Tout à fait d’accord","Plutôt d’accord","Plutôt pas d’accord","Pas du tout d’accord"]

# ---------- Utils ----------
def categorize(value, thres: dict):
    if value is None: return "—"
    if value >= thres.get("high", 9e9): return "élevé"
    if value >= thres.get("medium", -9e9): return "moyen"
    return "faible"

def synthese_re(hr, er, he, ee, thres_re: dict):
    parts = []
    for lab, val in [("HR",hr),("ER",er),("HE",he),("EE",ee)]:
        cat = categorize(val, thres_re.get(lab, {}))
        if lab in ["HR","ER"]:
            txt = {"élevé":"forte aisance/motivation analytique",
                   "moyen":"ressources rationnelles présentes",
                   "faible":"habileté analytique perçue plus basse"}.get(cat,"—")
        else:
            txt = {"élevé":"recours fréquent à l’intuition/ressenti",
                   "moyen":"équilibre intuition/logique",
                   "faible":"intuition moins mobilisée"}.get(cat,"—")
        sval = "—" if val is None else f"{val:.2f}"
        parts.append(f"- {lab} **{sval}** ({cat}) : {txt}.")
    return "\n".join(parts)

def interpretation_bmri(bmri_result):
    if bmri_result is None or not bmri_result.get("choices"): return "—"
    a_cnt = sum(1 for v in bmri_result["choices"].values() if v=="A")
    b_cnt = len(bmri_result["choices"]) - a_cnt
    if abs(a_cnt-b_cnt) <= 2:
        return f"Profil équilibré (A={a_cnt}, B={b_cnt}) : alternance entre raisonnement et intuition."
    return (f"Tendance rationnelle (A={a_cnt} > B={b_cnt}) : préférence pour l’analyse structurée."
            if a_cnt>b_cnt else
            f"Tendance expérientielle (B={b_cnt} > A={a_cnt}) : préférence pour l’intuition et le ressenti.")

def compte_rendu_auto(short_totals, re_cats, name=None, age=None, bmri_result=None):
    spq_cat = re_cats.get("_SPQ_cat","—"); eq_cat = re_cats.get("_EQ_cat","—")
    qr_cat = re_cats.get("_QR_cat","—");  qa_cat = re_cats.get("_QA_cat","—")
    hr_s,hr_c = re_cats.get("HR",(None,"—")); er_s,er_c = re_cats.get("ER",(None,"—"))
    he_s,he_c = re_cats.get("HE",(None,"—")); ee_s,ee_c = re_cats.get("EE",(None,"—"))
    bmri_sentence = interpretation_bmri(bmri_result) if bmri_result else "—"

    L=[]
    L.append("## Compte rendu automatique")
    meta=[]
    if name: meta.append(f"Nom : **{name}**")
    if age:  meta.append(f"Âge : **{age}**")
    meta.append(f"Date : **{datetime.now().strftime('%Y-%m-%d %H:%M')}**")
    L.append(" — ".join(meta)); L.append("")
    L.append("### Vue d’ensemble")
    phrases=[]
    if he_c=="élevé" and er_c in ["élevé","moyen"]:
        phrases.append("Profil **intuitif engagé** (intuition + appétence pour l’analyse).")
    if hr_c=="faible" and er_c in ["élevé","moyen"]:
        phrases.append("Motivation pour raisonner présente, mais **habileté analytique perçue plus basse**.")
    if hr_c=="élevé" and he_c=="élevé":
        phrases.append("Double appui **logique + intuition**.")
    if not phrases: phrases.append("Préférences R/E **équilibrées**.")
    if bmri_sentence and bmri_sentence!="—": phrases.append(f"BMRI : {bmri_sentence}")
    L.append("- " + " ".join(phrases)); L.append("")
    L.append("### Rationnel / Expérientiel")
    for lab,(s,c) in {"HR":(hr_s,hr_c),"ER":(er_s,er_c),"HE":(he_s,he_c),"EE":(ee_s,ee_c)}.items():
        if s is None: continue
        L.append(f"- {lab} : **{s:.2f}** ({c}).")
    L.append("")
    L.append("### Indicateurs courts")
    L.append(f"- SPQ-10 : **{spq_cat}**.")
    L.append(f"- EQ-10  : **{eq_cat}**.")
    L.append(f"- Q-R-10 : **{qr_cat}**.")
    L.append(f"- QA-10  : **{qa_cat}**.")
    L.append("")
    L.append("### Limites")
    L.append("- Auto-rapporté ; à croiser avec l’observation clinique et le contexte.")
    L.append("- Seuils et clé BMRI ajustables dans `questionnaire.yml`.")
    return "\n".join(L)

# ---------- Graphiques ----------
def fig_re_radar(hr, er, he, ee):
    import math
    labels = ["HR","ER","HE","EE"]
    values = [v if v is not None else 0 for v in [hr,er,he,ee]]
    values += values[:1]
    angles = [n/float(len(labels))*2*math.pi for n in range(len(labels))]
    angles += angles[:1]
    fig = plt.figure(figsize=(4,4))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(math.pi/2); ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels)
    ax.set_yticks([1,2,3,4,5]); ax.set_ylim(0,5)
    ax.plot(angles, values); ax.fill(angles, values, alpha=0.2)
    ax.set_title("Profil R/E (1–5)")
    return fig

def fig_short_bars(short_totals):
    items = list(short_totals.keys()); vals=[short_totals[k] for k in items]
    fig, ax = plt.subplots(figsize=(6,3))
    ax.bar(items, vals); ax.set_title("Questionnaires courts – totaux"); ax.set_ylabel("Score")
    ax.set_xticklabels(items, rotation=15, ha="right")
    return fig

def fig_bmri_ab(bmri_result):
    a_cnt = sum(1 for v in bmri_result["choices"].values() if v=="A")
    b_cnt = len(bmri_result["choices"]) - a_cnt
    fig, ax = plt.subplots(figsize=(4,3))
    ax.bar(["A","B"], [a_cnt, b_cnt]); ax.set_title("BMRI – A vs B"); ax.set_ylabel("Nombre de réponses")
    return fig

def download_fig_button(fig, filename):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    st.download_button("⬇️ Télécharger le graphique", data=buf.getvalue(),
                       file_name=filename, mime="image/png")

# ---------- E-mail ----------
def send_email_smtp(to_addr: str, subject: str, body_text: str, attachments: list):
    """
    Envoie un e-mail via SMTP Gmail avec pièces jointes.
    attachments: [(filename:str, content_bytes:bytes, mime:str)]
    Secrets requis : EMAIL_SENDER, EMAIL_APP_PASSWORD
    """
    sender = st.secrets.get("EMAIL_SENDER")
    app_pwd = st.secrets.get("EMAIL_APP_PASSWORD")
    if not sender or not app_pwd:
        st.error("Secrets manquants : définis EMAIL_SENDER et EMAIL_APP_PASSWORD dans .streamlit/secrets.toml")
        return False

    msg = EmailMessage()
    msg["From"] = sender
    msg["To"] = to_addr
    msg["Subject"] = subject
    msg.set_content(body_text)

    for fname, content, mime in attachments:
        maintype, subtype = (mime.split("/", 1) if "/" in mime else ("application","octet-stream"))
        msg.add_attachment(content, maintype=maintype, subtype=subtype, filename=fname)

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(sender, app_pwd)
            smtp.send_message(msg)
        return True
    except Exception as e:
        st.error(f"Échec d’envoi SMTP : {e}")
        return False

def send_email_sendgrid(to_addr: str, subject: str, body_text: str, attachments: list):
    """ Variante via SendGrid (nécessite SENDGRID_API_KEY dans les secrets et la lib `sendgrid`). """
    try:
        from sendgrid import SendGridAPIClient
        from sendgrid.helpers.mail import Mail, Attachment, FileContent, FileName, FileType, Disposition
    except Exception:
        st.error("SendGrid non disponible : ajoute `sendgrid` à requirements.txt.")
        return False

    sender = st.secrets.get("EMAIL_SENDER")
    api_key = st.secrets.get("SENDGRID_API_KEY")
    if not sender or not api_key:
        st.error("Secrets manquants : définis SENDGRID_API_KEY et EMAIL_SENDER.")
        return False

    message = Mail(from_email=sender, to_emails=to_addr, subject=subject, plain_text_content=body_text)

    import base64
    for fname, content, mime in attachments:
        att = Attachment()
        att.file_content = FileContent(base64.b64encode(content).decode())
        att.file_type = FileType(mime or "application/octet-stream")
        att.file_name = FileName(fname)
        att.disposition = Disposition("attachment")
        message.add_attachment(att)

    try:
        sg = SendGridAPIClient(api_key)
        resp = sg.send(message)
        if 200 <= resp.status_code < 300:
            return True
        st.error(f"SendGrid a répondu {resp.status_code}.")
        return False
    except Exception as e:
        st.error(f"Échec d’envoi SendGrid : {e}")
        return False

# ---------- Widgets de passation ----------
def ask_block_likert(block: dict):
    labels = block.get("scale_labels", DEFAULT_LIKERT4)
    out = {}
    st.subheader(block.get("key", "Échelle"))
    for it in block.get("items", []):
        vals = it["values"]; n=len(vals)
        local_labels = (labels + [f"Option {i+1}" for i in range(len(labels), n)])[:n]
        c1, c2 = st.columns([3,2])
        with c1: st.write(f"**{it['id']}** — {it.get('text','')}")
        with c2:
            choice = st.radio(it["id"], options=list(range(n)), index=0,
                              horizontal=True, label_visibility="collapsed",
                              format_func=lambda i: local_labels[i])
        out[it["id"]] = vals[choice]; st.divider()
    return out

def ask_block_forced_choice(block: dict):
    st.subheader(block.get("key","BMRI (A/B)"))
    choices={}
    for idx,it in enumerate(block.get("items",[]),1):
        stem,a,b = it.get("stem",""), it.get("a",""), it.get("b","")
        st.markdown(f"**{it.get('id', f'FC{idx}')}** — {stem}")
        pick = st.radio(it.get("id",f"FC{idx}"), options=["A","B"], index=0, horizontal=True,
                        format_func=lambda x: f"{x}) {a if x=='A' else b}")
        choices[it.get("id",f"FC{idx}")] = pick
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
        "_SPQ_cat": short_cats.get("SPQ-10","—"),
        "_EQ_cat":  short_cats.get("EQ-10","—"),
        "_QR_cat":  short_cats.get("Q-R-10","—"),
        "_QA_cat":  short_cats.get("QA-10","—"),
    }
    L=[]
    L.append("# Bilan HPE – Rapport (YAML)")
    L.append(f"**Date** : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    meta=" – ".join(x for x in [f"Nom: {name}" if name else None, f"Âge: {age}" if age else None] if x)
    if meta: L.append(meta)
    L.append("")
    L.append("## Scores – courts")
    for k,v in short_totals.items():
        L.append(f"- {k} : **{v:.0f}** ({short_cats.get(k,'—')})")
    L.append("")
    L.append("## Rationnel / Expérientiel (1–5)")
    for lab,val in [("HR",HR),("ER",ER),("HE",HE),("EE",EE)]:
        sval="—" if val is None else f"{val:.2f}"
        cat = re_cats[lab][1]
        L.append(f"- {lab} : **{sval}** ({cat})")
    L.append("")
    L.append("## Synthèse R/E")
    L.append(synthese_re(HR,ER,HE,EE, thresholds.get("re_scales", {})))
    if bmri_result:
        L.append("")
        L.append("## BMRI (28 items)")
        L.append(interpretation_bmri(bmri_result))
    L.append("")
    L.append(compte_rendu_auto(short_totals, re_cats, name=name, age=age, bmri_result=bmri_result))
    return "\n".join(L)

# ---------- MODE : 3 options ----------
mode = st.radio(
    "Choisir le mode :",
    ["Passer le test", "Téléverser un fichier existant", "Télécharger des fichiers de résultats"]
)

if mode == "Passer le test":
    # ---- Passation
    short_totals, re_scores, bmri_result = {}, {}, None

    for block in data.get("blocks", []):
        btype = block.get("type")
        if btype == "re":
            st.subheader("Échelle R/E (1–5)")
            scores={}
            for it in block.get("items", []):
                iid,tag,text = it.get("id"), it.get("tag"), it.get("text","")
                val = st.slider(f"{iid} – {tag} : {text}", 1, 5, 3, key=f"rei-{iid}")
                if it.get("reverse", False): val = 6 - val
                scores[str(iid)] = {"tag": str(tag).upper(), "score": float(val)}
            re_scores = scores
        elif btype == "forced_choice":
            bmri_result = ask_block_forced_choice(block)
        else:
            answers = ask_block_likert(block)
            short_totals[block["key"]] = sum(answers.values())

    def mean_tag(tag, re_scores):
        vals = [v["score"] for v in re_scores.values() if v["tag"]==tag]
        return round(float(np.mean(vals)),2) if vals else None
    HR,ER,HE,EE = (mean_tag("HR",re_scores), mean_tag("ER",re_scores),
                   mean_tag("HE",re_scores), mean_tag("EE",re_scores))

    # ---- Résultats
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

    # ---- Graphiques
    st.subheader("📊 Visualisations")
    colA,colB = st.columns(2)
    with colA:
        fig1 = fig_short_bars(short_totals); st.pyplot(fig1); download_fig_button(fig1,"courts.png")
    with colB:
        fig2 = fig_re_radar(HR,ER,HE,EE);    st.pyplot(fig2); download_fig_button(fig2,"rei.png")
    if bmri_result:
        fig3 = fig_bmri_ab(bmri_result);     st.pyplot(fig3); download_fig_button(fig3,"bmri.png")

    # ---- Exports
    with st.sidebar:
        st.header("Infos répondant")
        name = st.text_input("Nom (optionnel)")
        age  = st.text_input("Âge (optionnel)")

    report = build_report(short_totals, re_scores, HR, ER, HE, EE, name, age, bmri_result=bmri_result)
    st.download_button("💾 Rapport (.md)", report, file_name="bilan_hpe_rapport.md", mime="text/markdown")

    raw = {k:v for k,v in short_totals.items()}
    for k,v in re_scores.items(): raw[k]=v["score"]
    raw.update({"HR":HR,"ER":ER,"HE":HE,"EE":EE})
    if bmri_result:
        for k,v in bmri_result["choices"].items(): raw[k]=v
    csv_bytes = pd.DataFrame([raw]).to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Réponses (.csv)", csv_bytes, file_name="bilan_hpe_reponses.csv", mime="text/csv")

    # --- mémoriser dans la session (pour l’onglet Télécharger des fichiers)
    st.session_state["last_report_md"] = report
    st.session_state["last_raw_csv"]   = csv_bytes

    # ---- Envoi e-mail
    st.markdown("---")
    st.subheader("✉️ Envoi par e-mail")
    col1, col2 = st.columns([2,1])
    with col1:
        to_addr = st.text_input("Destinataire", value="Beatricemilletre@gmail.com")
        subject = st.text_input("Objet", value="Bilan HPE – Résultats de passation")
    with col2:
        use_sendgrid = st.toggle("Utiliser SendGrid (sinon Gmail SMTP)", value=False)
    body = st.text_area(
        "Message",
        value=("Bonjour,\n\nVeuillez trouver ci-joints le rapport (.md) et le fichier de réponses (.csv)."
               "\n\nBien cordialement,\nBilan HPE"),
        height=140
    )
    attachments = [
        ("bilan_hpe_rapport.md", report.encode("utf-8"), "text/markdown"),
        ("bilan_hpe_reponses.csv", csv_bytes, "text/csv"),
    ]
    if st.button("Envoyer maintenant ✉️"):
        if not to_addr.strip():
            st.warning("Renseigne une adresse destinataire.")
        else:
            ok = (send_email_sendgrid(to_addr, subject, body, attachments) if use_sendgrid
                  else send_email_smtp(to_addr, subject, body, attachments))
            if ok:
                st.success(f"E-mail envoyé à {to_addr}.")
                st.session_state["last_report_md"] = report
                st.session_state["last_raw_csv"]   = csv_bytes

elif mode == "Téléverser un fichier existant":
    st.title("📂 Importer un rapport ou des réponses")
    uploaded = st.file_uploader("Déposez un fichier .csv (réponses brutes) ou .md (rapport)")
    if uploaded:
        if uploaded.name.endswith(".csv"):
            df = pd.read_csv(uploaded)
            st.subheader("Prévisualisation du CSV importé")
            st.dataframe(df, use_container_width=True)
            st.download_button("⬇️ Télécharger une copie du CSV",
                               data=df.to_csv(index=False).encode("utf-8"),
                               file_name="bilan_hpe_reponses_copie.csv", mime="text/csv")
            name = st.text_input("Nom (optionnel)", key="up_name")
            age  = st.text_input("Âge (optionnel)", key="up_age")
            short_totals = {}
            for key in ["SPQ-10","EQ-10","Q-R-10","QA-10","BMRI-20"]:
                if key in df.columns:
                    try: short_totals[key] = float(df.iloc[0][key])
                    except: pass
            HR = float(df.iloc[0]["HR"]) if "HR" in df.columns else None
            ER = float(df.iloc[0]["ER"]) if "ER" in df.columns else None
            HE = float(df.iloc[0]["HE"]) if "HE" in df.columns else None
            EE = float(df.iloc[0]["EE"]) if "EE" in df.columns else None
            bmri_cols = [c for c in df.columns if c.upper().startswith("BMRI")]
            bmri_result = {"choices": {c: str(df.iloc[0][c]) for c in bmri_cols}} if bmri_cols else None
            report = build_report(short_totals, {}, HR, ER, HE, EE, name, age, bmri_result=bmri_result)
            st.subheader("📄 Générer un rapport depuis le CSV")
            st.download_button("💾 Rapport (.md)", report, file_name="bilan_hpe_rapport.md", mime="text/markdown")
        elif uploaded.name.endswith(".md"):
            content = uploaded.read().decode("utf-8")
            st.subheader("Rapport importé")
            st.markdown(content)
            st.download_button("⬇️ Télécharger une copie du rapport",
                               data=content.encode("utf-8"),
                               file_name="bilan_hpe_rapport_copie.md",
                               mime="text/markdown")
        else:
            st.error("Format non reconnu. Utilisez .csv ou .md")

elif mode == "Télécharger des fichiers de résultats":
    st.title("⬇️ Télécharger des fichiers de résultats")
    st.subheader("Derniers fichiers générés (cette session)")
    if "last_report_md" in st.session_state or "last_raw_csv" in st.session_state:
        col1, col2 = st.columns(2)
        with col1:
            if "last_report_md" in st.session_state:
                st.download_button(
                    "💾 Télécharger le dernier rapport (.md)",
                    data=st.session_state["last_report_md"],
                    file_name="bilan_hpe_rapport.md",
                    mime="text/markdown",
                )
            else:
                st.caption("Aucun rapport généré encore.")
        with col2:
            if "last_raw_csv" in st.session_state:
                st.download_button(
                    "⬇️ Télécharger le dernier CSV de réponses",
                    data=st.session_state["last_raw_csv"],
                    file_name="bilan_hpe_reponses.csv",
                    mime="text/csv",
                )
            else:
                st.caption("Aucun CSV généré encore.")
    else:
        st.info("Aucun export généré dans cette session pour l’instant. Passe le test une fois pour alimenter ici.")

    st.markdown("---")
    st.subheader("Modèles vierges (gabarits)")
    template_csv = pd.DataFrame([{
        "SPQ-10": "", "EQ-10": "", "Q-R-10": "", "QA-10": "",
        "HR": "", "ER": "", "HE": "", "EE": "",
        "BMRI1": "", "BMRI2": "", "BMRI3": "", "BMRI4": "", "BMRI5": "", "BMRI6": "",
        "BMRI7": "", "BMRI8": "", "BMRI9": "", "BMRI10": "", "BMRI11": "", "BMRI12": "",
        "BMRI13": "", "BMRI14": "", "BMRI15": "", "BMRI16": "", "BMRI17": "", "BMRI18": "",
        "BMRI19": "", "BMRI20": "", "BMRI21": "", "BMRI22": "", "BMRI23": "", "BMRI24": "",
        "BMRI25": "", "BMRI26": "", "BMRI27": "", "BMRI28": "",
    }]).to_csv(index=False).encode("utf-8")
    colA, colB = st.columns(2)
    with colA:
        st.download_button(
            "📄 Télécharger le modèle de CSV (vierge)",
            data=template_csv, file_name="modele_bilan_hpe_reponses.csv", mime="text/csv"
        )
    with colB:
        template_md = """# Bilan HPE – Rapport (modèle)
**Date** : AAAA-MM-JJ HH:MM
Nom: —  –  Âge: —

## Scores – courts
- SPQ-10 : —
- EQ-10  : —
- Q-R-10 : —
- QA-10  : —

## Rationnel / Expérientiel (1–5)
- HR : —
- ER : —
- HE : —
- EE : —

## BMRI (28 items)
- A vs B : — (ex. A=14, B=14)

## Compte rendu automatique
(Renseigner après passation ou import CSV)
"""
        st.download_button(
            "📝 Télécharger le modèle de rapport (.md)",
            data=template_md, file_name="modele_bilan_hpe_rapport.md", mime="text/markdown"
        )
