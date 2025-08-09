import io
import base64
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Tableau de Bord - Suivi des Articles", layout="wide")
st.title("Tableau de Bord - Suivi des Articles")

# ---------------------- Utilitaires lecture ----------------------
def _excel_engine_for(name: str) -> str | None:
    name = name.lower()
    if name.endswith(".xlsb"):
        return "pyxlsb"
    if name.endswith((".xlsx", ".xls")):
        return "openpyxl"
    return None

def read_excel_sheet(uploaded_file, sheet_name):
    engine = _excel_engine_for(uploaded_file.name)
    return pd.read_excel(uploaded_file, sheet_name=sheet_name, engine=engine)

# ---------------------- Construction des données ----------------------
def construire_flux(reception: pd.DataFrame, consommation: pd.DataFrame) -> pd.DataFrame:
    # Nettoyage et typage
    for df in (reception, consommation):
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        if "Quantity" in df.columns:
            df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    reception = reception.dropna(subset=["Date", "Article", "Quantity"])
    consommation = consommation.dropna(subset=["Date", "Article", "Quantity"])

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

    flux = (
        pd.merge(reception_agg, consommation_agg, on=["Date", "Article"], how="outer")
        .sort_values(["Article", "Date"])
    )

    flux["Quantite_Recue"] = flux["Quantite_Recue"].fillna(0)
    flux["Quantite_Consommee"] = flux["Quantite_Consommee"].fillna(0)
    flux["Cumul_Recu"] = flux.groupby("Article")["Quantite_Recue"].cumsum()
    flux["Cumul_Consomme"] = flux.groupby("Article")["Quantite_Consommee"].cumsum()
    flux["Stock_Courant"] = flux["Cumul_Recu"] - flux["Cumul_Consomme"]
    return flux

def afficher_graphique(flux: pd.DataFrame, article: str):
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

def calculer_kpis(flux: pd.DataFrame) -> pd.DataFrame:
    if flux.empty:
        return pd.DataFrame(columns=[
            "Article","Quantite_Recue_Totale","Quantite_Consommee_Totale",
            "Taux_Rotation","Conso_Moyenne_Journaliere","Stock_Actuel","Jours_Couverture"
        ])

    kpi = (
        flux.groupby("Article", as_index=False)
        .agg(
            Quantite_Recue_Totale=("Quantite_Recue", "sum"),
            Quantite_Consommee_Totale=("Quantite_Consommee", "sum")
        )
    )

    if flux["Date"].notna().any():
        jours = int((flux["Date"].max() - flux["Date"].min()).days) + 1
        jours = max(jours, 1)
    else:
        jours = 1

    kpi["Taux_Rotation"] = (
        kpi["Quantite_Consommee_Totale"] / kpi["Quantite_Recue_Totale"].replace(0, pd.NA)
    )
    kpi["Conso_Moyenne_Journaliere"] = kpi["Quantite_Consommee_Totale"] / jours

    stock_actuel = (
        flux.sort_values(["Article", "Date"])
            .groupby("Article")
            .tail(1)[["Article", "Stock_Courant"]]
            .rename(columns={"Stock_Courant": "Stock_Actuel"})
    )
    kpi = kpi.merge(stock_actuel, on="Article", how="left")
    kpi["Jours_Couverture"] = (
        kpi["Stock_Actuel"] / kpi["Conso_Moyenne_Journaliere"].replace(0, pd.NA)
    )
    return kpi

def recommander_approvisionnement(kpis: pd.DataFrame, couverture_jours=30) -> pd.DataFrame:
    if kpis.empty:
        return kpis
    kpis = kpis.copy()
    kpis["Stock_Cible"] = kpis["Conso_Moyenne_Journaliere"] * couverture_jours
    kpis["Approvisionnement_Recommande"] = (
        kpis["Stock_Cible"] - kpis["Stock_Actuel"]
    ).clip(lower=0)
    return kpis[["Article", "Stock_Actuel", "Stock_Cible", "Approvisionnement_Recommande"]]

# ---------------------- UI: un fichier, deux feuilles ----------------------
uploaded_file = st.file_uploader(
    "Importez un fichier Excel contenant deux feuilles: Réception et Consommation",
    type=["xlsx", "xls", "xlsb"]
)

if not uploaded_file:
    st.info("Importez le fichier Excel pour continuer.")
    st.stop()

try:
    xls = pd.ExcelFile(uploaded_file, engine=_excel_engine_for(uploaded_file.name))
    sheets = xls.sheet_names
    st.write("Feuilles détectées :", sheets)

    sheet_rec = st.selectbox("Feuille Réception", options=sheets, index=0)
    sheet_conso = st.selectbox("Feuille Consommation", options=sheets, index=1 if len(sheets) > 1 else 0)

    reception_df = read_excel_sheet(uploaded_file, sheet_rec)
    consommation_df = read_excel_sheet(uploaded_file, sheet_conso)
except Exception as e:
    st.error(f"Erreur de lecture du fichier Excel: {e}")
    st.stop()

# Validation minimale des colonnes attendues
required_cols = {"Date", "Article", "Quantity"}
missing_rec = required_cols - set(reception_df.columns)
missing_con = required_cols - set(consommation_df.columns)
if missing_rec or missing_con:
    st.error(
        "Colonnes attendues: Date, Article, Quantity. "
        f"Manquantes dans Réception: {sorted(missing_rec)}; "
        f"Manquantes dans Consommation: {sorted(missing_con)}"
    )
    st.stop()

# ---------------------- Calculs et affichages ----------------------
flux_df = construire_flux(reception_df, consommation_df)

articles = sorted(flux_df["Article"].dropna().unique())
article_choisi = st.selectbox("Sélectionnez un code article", ["(Tous)"] + articles)

st.subheader("Graphique de flux")
if article_choisi and article_choisi != "(Tous)":
    fig = afficher_graphique(flux_df, article_choisi)
    if fig:
        st.pyplot(fig)
else:
    st.caption("Sélectionnez un article pour afficher le graphique détaillé.")

st.subheader("Indicateurs de Performance (KPI)")
kpis = calculer_kpis(flux_df)
st.dataframe(kpis, use_container_width=True)

st.subheader("Recommandation d’Approvisionnement (30 jours de couverture)")
recommandation = recommander_approvisionnement(kpis, 30)
st.dataframe(recommandation, use_container_width=True)

# ---------------------- Fond d'écran optionnel ----------------------
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

add_background_image("background.jpg")
