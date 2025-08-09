import base64
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Tableau de Bord - Suivi des Articles", layout="wide")
st.title("Tableau de Bord - Suivi des Articles")

# ---------------------- Helpers: sheet/column detection ----------------------
SYNONYMS = {
    "date": {
        "date", "jour", "day", "date transaction", "date bs", "date reception",
        "date consommation", "date de reception", "date de consommation",
        "date_entree", "date_sortie"
    },
    "article": {
        "article", "code", "code article", "sku", "item", "produit",
        "reference", "référence", "designation", "libelle", "libellé"
    },
    "qty": {
        "qty", "quantity", "quantite", "quantité", "qte", "qté", "qte recu",
        "qte reçue", "qte consommee", "qte consommée", "entree", "sortie",
        "input", "output"
    },
}

def _excel_engine_for(name: str) -> str | None:
    name = name.lower()
    if name.endswith(".xlsb"):
        return "pyxlsb"
    if name.endswith((".xlsx", ".xls")):
        return "openpyxl"
    return None

def _best_match(colnames, kind):
    lower_map = {c.lower(): c for c in colnames}
    # exact match on known synonyms
    for low, original in lower_map.items():
        if low.strip() in SYNONYMS[kind]:
            return original
    # relaxed contains search
    for low, original in lower_map.items():
        if any(tok in low for tok in SYNONYMS[kind]):
            return original
    return None

def _read_sheet(uploaded_file, sheet_name):
    engine = _excel_engine_for(uploaded_file.name)
    return pd.read_excel(uploaded_file, sheet_name=sheet_name, engine=engine)

# ---------------------- Upload: one file, two sheets ----------------------
uploaded_file = st.file_uploader(
    "Importer un fichier Excel contenant deux feuilles (Réception et Consommation)",
    type=["xlsx", "xls", "xlsb"]
)

if not uploaded_file:
    st.info("Importez le fichier Excel pour continuer.")
    st.stop()

try:
    xls = pd.ExcelFile(uploaded_file, engine=_excel_engine_for(uploaded_file.name))
    sheets = xls.sheet_names
except Exception as e:
    st.error(f"Impossible de lire le fichier: {e}")
    st.stop()

st.write("Feuilles détectées:", ", ".join(sheets))

def _guess_sheet(role):
    targets = ["reception", "réception"] if role == "reception" else ["consommation", "consumption"]
    for i, s in enumerate(sheets):
        if any(t in s.lower() for t in targets):
            return i
    return 0 if role == "reception" else (1 if len(sheets) > 1 else 0)

sheet_rec = st.selectbox("Feuille Réception", options=sheets, index=_guess_sheet("reception"))
sheet_conso = st.selectbox("Feuille Consommation", options=sheets, index=_guess_sheet("consommation"))

try:
    rec_raw = _read_sheet(uploaded_file, sheet_rec)
    conso_raw = _read_sheet(uploaded_file, sheet_conso)
except Exception as e:
    st.error(f"Erreur de lecture des feuilles: {e}")
    st.stop()

with st.expander("Aperçu Réception (10 premières lignes)"):
    st.dataframe(rec_raw.head(10), use_container_width=True)
with st.expander("Aperçu Consommation (10 premières lignes)"):
    st.dataframe(conso_raw.head(10), use_container_width=True)

# ---------------------- Column mapping UI ----------------------
def mapping_ui(df, label):
    cols = list(df.columns)
    guess_date = _best_match(cols, "date") or "(aucune)"
    guess_art  = _best_match(cols, "article") or "(aucune)"
    guess_qty  = _best_match(cols, "qty") or "(aucune)"
    choices = ["(aucune)"] + cols
    st.caption(f"Associer les colonnes pour {label}")
    m_date = st.selectbox("Colonne Date", choices, index=choices.index(guess_date) if guess_date in choices else 0, key=f"{label}_date")
    m_art  = st.selectbox("Colonne Article", choices, index=choices.index(guess_art)  if guess_art  in choices else 0, key=f"{label}_article")
    m_qty  = st.selectbox("Colonne Quantité", choices, index=choices.index(guess_qty) if guess_qty in choices else 0, key=f"{label}_qty")
    return {"Date": m_date, "Article": m_art, "Quantity": m_qty}

st.subheader("Associer les colonnes")
c1, c2 = st.columns(2)
with c1:
    map_rec = mapping_ui(rec_raw, "Réception")
with c2:
    map_conso = mapping_ui(conso_raw, "Consommation")

if "(aucune)" in map_rec.values() or "(aucune)" in map_conso.values():
    st.warning("Veuillez mapper Date, Article et Quantité pour les deux feuilles.")
    st.stop()

@st.cache_data(show_spinner=False)
def normalize_frames(rec_raw, conso_raw, map_rec, map_conso):
    rec = rec_raw.rename(columns={
        map_rec["Date"]: "Date", map_rec["Article"]: "Article", map_rec["Quantity"]: "Quantity"
    })[["Date", "Article", "Quantity"]].copy()
    con = conso_raw.rename(columns={
        map_conso["Date"]: "Date", map_conso["Article"]: "Article", map_conso["Quantity"]: "Quantity"
    })[["Date", "Article", "Quantity"]].copy()

    rec["Date"] = pd.to_datetime(rec["Date"], errors="coerce")
    con["Date"] = pd.to_datetime(con["Date"], errors="coerce")
    rec["Quantity"] = pd.to_numeric(rec["Quantity"], errors="coerce")
    con["Quantity"] = pd.to_numeric(con["Quantity"], errors="coerce")

    rec = rec.dropna(subset=["Date", "Article", "Quantity"])
    con = con.dropna(subset=["Date", "Article", "Quantity"])
    return rec, con

reception_df, consommation_df = normalize_frames(rec_raw, conso_raw, map_rec, map_conso)

# ---------------------- Business logic ----------------------
def construire_flux(reception, consommation):
    reception = reception.sort_values(["Article", "Date"])
    consommation = consommation.sort_values(["Article", "Date"])

    reception_agg = (
        reception.groupby(["Date", "Article"], as_index=False)
        .agg(Quantite_Recue=("Quantity", "sum"))
    )
    consommation_agg = (
        consommation.groupby(["Date", "Article"], as_index=False)
        .agg(Quantite_Consommee=("Quantity", "sum"))
    )

    flux = pd.merge(reception_agg, consommation_agg, on=["Date", "Article"], how="outer")
    flux = flux.sort_values(["Article", "Date"]).fillna({"Quantite_Recue": 0, "Quantite_Consommee": 0})
    flux["Cumul_Recu"] = flux.groupby("Article")["Quantite_Recue"].cumsum()
    flux["Cumul_Consomme"] = flux.groupby("Article")["Quantite_Consommee"].cumsum()
    flux["Stock_Courant"] = flux["Cumul_Recu"] - flux["Cumul_Consomme"]
    return flux

def afficher_graphique(flux, article):
    data = flux[flux["Article"] == article]
    if data.empty:
        st.warning("Aucune donnée pour cet article.")
        return None
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(data["Date"], data["Cumul_Recu"], label="Cumul Réception")
    ax.plot(data["Date"], data["Cumul_Consomme"], label="Cumul Consommation")
    ax.plot(data["Date"], data["Stock_Courant"], label="Stock Courant", linestyle="--")
    ax.set_xlabel("Date")
    ax.set_ylabel("Quantité")
    ax.set_title(f"Flux de Stock - {article}")
    ax.legend()
    ax.grid(True)
    return fig

def calculer_kpis(flux):
    if flux.empty:
        return pd.DataFrame(columns=[
            "Article","Quantite_Recue_Totale","Quantite_Consommee_Totale",
            "Taux_Rotation","Conso_Moyenne_Journaliere","Stock_Actuel","Jours_Couverture"
        ])
    kpi = (
        flux.groupby("Article", as_index=False)
        .agg(
            Quantite_Recue_Totale=("Quantite_Recue", "sum"),
            Quantite_Consommee_Totale=("Quantite_Consommee", "sum"),
        )
    )
    if flux["Date"].notna().any():
        jours = max(int((flux["Date"].max() - flux["Date"].min()).days) + 1, 1)
    else:
        jours = 1
    kpi["Taux_Rotation"] = kpi["Quantite_Consommee_Totale"] / kpi["Quantite_Recue_Totale"].replace(0, pd.NA)
    kpi["Conso_Moyenne_Journaliere"] = kpi["Quantite_Consommee_Totale"] / jours

    stock_actuel = (
        flux.sort_values(["Article", "Date"])
            .groupby("Article")
            .tail(1)[["Article", "Stock_Courant"]]
            .rename(columns={"Stock_Courant": "Stock_Actuel"})
    )
    kpi = kpi.merge(stock_actuel, on="Article", how="left")
    kpi["Jours_Couverture"] = kpi["Stock_Actuel"] / kpi["Conso_Moyenne_Journaliere"].replace(0, pd.NA)
    return kpi

def recommander_approvisionnement(kpis, couverture_jours=30):
    if kpis.empty:
        return kpis
    kpis = kpis.copy()
    kpis["Stock_Cible"] = kpis["Conso_Moyenne_Journaliere"] * couverture_jours
    kpis["Approvisionnement_Recommande"] = (kpis["Stock_Cible"] - kpis["Stock_Actuel"]).clip(lower=0)
    return kpis[["Article", "Stock_Actuel", "Stock_Cible", "Approvisionnement_Recommande"]]

# ---------------------- Compute and display ----------------------
flux_df = construire_flux(reception_df, consommation_df)

articles = sorted(flux_df["Article"].dropna().unique())
article_choisi = st.selectbox("Sélectionnez un code article", ["(Tous)"] + articles)

st.subheader("Graphique de flux")
if article_choisi and article_choisi != "(Tous)":
    fig = afficher_graphique(flux_df, article_choisi)
    if fig:
        st.pyplot(fig)
else:
    st.caption("Sélectionnez un article pour afficher le graphique.")

st.subheader("Indicateurs de Performance (KPI)")
kpis = calculer_kpis(flux_df)
st.dataframe(kpis, use_container_width=True)

st.subheader("Recommandation d’Approvisionnement (30 jours de couverture)")
recommandation = recommander_approvisionnement(kpis, 30)
st.dataframe(recommandation, use_container_width=True)

# ---------------------- Optional background and dark text ----------------------
def add_background_image(local_image_path):
    try:
        with open(local_image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        css = f"""
        <style>
        .stApp {{
            background: linear-gradient(rgba(255,255,255,0.7), rgba(255,255,255,0.7)),
                        url("data:image/jpg;base64,{encoded_string}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)
    except FileNotFoundError:
        st.caption("Ajoutez un fichier 'background.jpg' à la racine si vous souhaitez un fond personnalisé.")

def darken_text():
    css = """
    <style>
    /* Darken all text globally for readability over background */
    html, body, .stApp, [class*="css"] {
        color: #000000 !important;
    }
    /* Make labels and captions darker/bolder */
    label, .stCaption, .stMarkdown p, .st-emotion-cache, .stSelectbox label {
        color: #111111 !important;
        font-weight: 600;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

add_background_image("background.jpg")
darken_text()
