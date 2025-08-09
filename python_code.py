import io
import base64
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Tableau de Bord - Suivi des Articles", layout="wide")
st.title("Tableau de Bord - Suivi des Articles")

# ---------- Chargement des données (depuis l'interface) ----------
def _read_any_table(uploaded_file, parse_dates=("Date",)):
    """Lit CSV/XLSX/XLS/XLSB en détectant le séparateur pour CSV."""
    name = uploaded_file.name.lower()
    raw = uploaded_file.getvalue()
    bio = io.BytesIO(raw)

    if name.endswith(".csv"):
        for sep in [",", ";", "\t", "|"]:
            bio.seek(0)
            try:
                df = pd.read_csv(bio, sep=sep)
                if df.shape[1] > 1:  # séparateur plausible
                    break
            except Exception:
                continue
        else:
            bio.seek(0)
            df = pd.read_csv(bio)  # dernier essai
    elif name.endswith((".xlsx", ".xls")):
        bio.seek(0)
        df = pd.read_excel(bio, engine="openpyxl")
    elif name.endswith(".xlsb"):
        bio.seek(0)
        df = pd.read_excel(bio, engine="pyxlsb")
    else:
        raise ValueError("Type de fichier non supporté. Utilisez CSV, XLSX, XLS ou XLSB.")

    # Normaliser les dates attendues si présentes
    for col in parse_dates:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

@st.cache_data(show_spinner=False)
def charger_donnees_uploader(rec_file, conso_file, mapping_rec, mapping_conso):
    """Lit, renomme selon mapping, garde colonnes essentielles."""
    rec = _read_any_table(rec_file)
    con = _read_any_table(conso_file)

    # Renommer selon mapping choisi dans l'UI
    rec = rec.rename(columns={mapping_rec["Date"]: "Date",
                              mapping_rec["Article"]: "Article",
                              mapping_rec["Quantity"]: "Quantity"})
    con = con.rename(columns={mapping_conso["Date"]: "Date",
                              mapping_conso["Article"]: "Article",
                              mapping_conso["Quantity"]: "Quantity"})

    # Garder colonnes utiles et typer
    rec = rec[["Date", "Article", "Quantity"]].copy()
    con = con[["Date", "Article", "Quantity"]].copy()

    rec["Date"] = pd.to_datetime(rec["Date"], errors="coerce")
    con["Date"] = pd.to_datetime(con["Date"], errors="coerce")
    rec["Quantity"] = pd.to_numeric(rec["Quantity"], errors="coerce")
    con["Quantity"] = pd.to_numeric(con["Quantity"], errors="coerce")

    # Drop lignes vides
    rec = rec.dropna(subset=["Date", "Article", "Quantity"])
    con = con.dropna(subset=["Date", "Article", "Quantity"])

    return rec, con

# ---------- Construction des indicateurs ----------
def construire_flux(reception, consommation):
    # S'assurer de l'ordre par Article + Date pour cumsum cohérents
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
    flux = pd.merge(
        reception_agg, consommation_agg, on=["Date", "Article"], how="outer"
    ).sort_values(["Article", "Date"])

    flux["Quantite_Recue"] = flux["Quantite_Recue"].fillna(0)
    flux["Quantite_Consommee"] = flux["Quantite_Consommee"].fillna(0)

    # cumsum par article
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
            Quantite_Consommee_Totale=("Quantite_Consommee", "sum")
        )
    )
    # garde-fou si dates manquantes
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

def recommander_approvisionnement(kpis, couverture_jours=30):
    if kpis.empty:
        return kpis
    kpis = kpis.copy()
    kpis["Stock_Cible"] = kpis["Conso_Moyenne_Journaliere"] * couverture_jours
    kpis["Approvisionnement_Recommande"] = (
        kpis["Stock_Cible"] - kpis["Stock_Actuel"]
    ).clip(lower=0)
    return kpis[["Article", "Stock_Actuel", "Stock_Cible", "Approvisionnement_Recommande"]]

# ---------- UI: Upload + mapping colonnes ----------
st.subheader("1) Importez vos fichiers")
col1, col2 = st.columns(2)
with col1:
    rec_file = st.file_uploader("Réception (CSV/XLSX/XLS/XLSB)", type=["csv","xlsx","xls","xlsb"], key="rec")
with col2:
    conso_file = st.file_uploader("Consommation (CSV/XLSX/XLS/XLSB)", type=["csv","xlsx","xls","xlsb"], key="conso")

mapping_rec = {"Date": None, "Article": None, "Quantity": None}
mapping_conso = {"Date": None, "Article": None, "Quantity": None}

def _mapping_box(df, label):
    st.caption(f"Mapping colonnes — {label}")
    cols = ["(aucune)"] + list(df.columns)
    def pick(name, default):
        idx = cols.index(default) if default in cols else 0
        return st.selectbox(name, cols, index=idx, key=f"{label}_{name}")
    # heuristique de présélection
    def guess(colnames):
        low = [c.lower() for c in colnames]
        g = {
            "Date": next((c for c in colnames if c.lower() in ("date","jour","day")), "(aucune)"),
            "Article": next((c for c in colnames if c.lower() in ("article","code","sku","item","produit")), "(aucune)"),
            "Quantity": next((c for c in colnames if c.lower() in ("qty","quantity","quantite","qte","qté")), "(aucune)"),
        }
        return g
    guesses = guess(df.columns)
    return {
        "Date": pick("Colonne Date", guesses["Date"]),
        "Article": pick("Colonne Article", guesses["Article"]),
        "Quantity": pick("Colonne Quantité", guesses["Quantity"]),
    }

if rec_file and conso_file:
    # Aperçu + mapping
    try:
        rec_preview = _read_any_table(rec_file).head(200)
        con_preview = _read_any_table(conso_file).head(200)
        with st.expander("Aperçu Réception (top 200 lignes)"):
            st.dataframe(rec_preview, use_container_width=True)
        with st.expander("Aperçu Consommation (top 200 lignes)"):
            st.dataframe(con_preview, use_container_width=True)

        st.subheader("2) Associez les colonnes")
        c1, c2 = st.columns(2)
        with c1:
            mapping_rec = _mapping_box(rec_preview, "Réception")
        with c2:
            mapping_conso = _mapping_box(con_preview, "Consommation")

        # Validation de mapping
        if "(aucune)" in mapping_rec.values() or "(aucune)" in mapping_conso.values():
            st.warning("Veuillez mapper Date, Article et Quantité pour les deux fichiers.")
            st.stop()

        # Charger définitivement
        reception_df, consommation_df = charger_donnees_uploader(
            rec_file, conso_file, mapping_rec, mapping_conso
        )
    except Exception as e:
        st.error(f"Erreur de chargement: {e}")
        st.stop()
else:
    st.info("Importez **les deux** fichiers pour continuer.")
    st.stop()

# ---------- Calculs & affichages ----------
flux_df = construire_flux(reception_df, consommation_df)

articles = sorted(flux_df['Article'].dropna().unique())
article_choisi = st.selectbox("Sélectionnez un code article :", ["(Tous)"] + articles)

st.subheader("Graphique de flux")
if article_choisi and article_choisi != "(Tous)":
    fig = afficher_graphique(flux_df, article_choisi)
    if fig:
        st.pyplot(fig)
else:
    st.caption("Astuce : sélectionnez un article pour voir le détail graphique.")

st.subheader("Indicateurs de Performance (KPI)")
kpis = calculer_kpis(flux_df)
st.dataframe(kpis, use_container_width=True)

st.subheader("Recommandation d’Approvisionnement (30 jours de couverture)")
recommandation = recommander_approvisionnement(kpis, 30)
st.dataframe(recommandation, use_container_width=True)

# ---------- Fond d'écran (optionnel) ----------
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
        st.caption("💡 Ajoutez un fichier 'background.jpg' à la racine si vous souhaitez un fond personnalisé.")

add_background_image("background.jpg")
