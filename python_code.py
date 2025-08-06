import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

@st.cache_data
def charger_donnees():
    reception = pd.read_csv("reception_cleaned.csv", parse_dates=["Date"])
    consommation = pd.read_csv("consumption_cleaned.csv", parse_dates=["Date"])
    return reception, consommation

def construire_flux(reception, consommation):
    reception_agg = (
        reception.groupby(["Date", "Article"])
        .agg(Quantite_Recue=("Quantity", "sum"))
        .reset_index()
    )
    consommation_agg = (
        consommation.groupby(["Date", "Article"])
        .agg(Quantite_Consommee=("Quantity", "sum"))
        .reset_index()
    )
    flux = pd.merge(reception_agg, consommation_agg, on=["Date", "Article"], how="outer").sort_values("Date")
    flux["Quantite_Recue"] = flux["Quantite_Recue"].fillna(0)
    flux["Quantite_Consommee"] = flux["Quantite_Consommee"].fillna(0)
    flux["Cumul_Recu"] = flux.groupby("Article")["Quantite_Recue"].cumsum()
    flux["Cumul_Consomme"] = flux.groupby("Article")["Quantite_Consommee"].cumsum()
    flux["Stock_Courant"] = flux["Cumul_Recu"] - flux["Cumul_Consomme"]
    return flux

def afficher_graphique(flux, article):
    data = flux[flux["Article"] == article]
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
    kpi = (
        flux.groupby("Article")
        .agg(
            Quantite_Recue_Totale=("Quantite_Recue", "sum"),
            Quantite_Consommee_Totale=("Quantite_Consommee", "sum")
        )
        .reset_index()
    )
    jours = (flux["Date"].max() - flux["Date"].min()).days + 1
    kpi["Taux_Rotation"] = kpi["Quantite_Consommee_Totale"] / kpi["Quantite_Recue_Totale"]
    kpi["Conso_Moyenne_Journaliere"] = kpi["Quantite_Consommee_Totale"] / jours

    stock_actuel = flux.groupby("Article").apply(lambda g: g.iloc[-1]["Stock_Courant"]).reset_index()
    stock_actuel.columns = ["Article", "Stock_Actuel"]

    kpi = kpi.merge(stock_actuel, on="Article", how="left")
    kpi["Jours_Couverture"] = kpi["Stock_Actuel"] / kpi["Conso_Moyenne_Journaliere"]
    return kpi

def recommander_approvisionnement(kpis, couverture_jours=30):
    kpis["Stock_Cible"] = kpis["Conso_Moyenne_Journaliere"] * couverture_jours
    kpis["Approvisionnement_Recommande"] = kpis["Stock_Cible"] - kpis["Stock_Actuel"]
    kpis["Approvisionnement_Recommande"] = kpis["Approvisionnement_Recommande"].apply(lambda x: max(x, 0))
    return kpis[["Article", "Stock_Actuel", "Stock_Cible", "Approvisionnement_Recommande"]]

# Interface Streamlit
st.title("📦 Tableau de Bord - Suivi des Articles")

reception_df, consommation_df = charger_donnees()
flux_df = construire_flux(reception_df, consommation_df)

articles = sorted(flux_df['Article'].unique())
article_choisi = st.selectbox("Sélectionnez un code article :", articles)

if article_choisi:
    st.subheader("📈 Graphique de flux")
    fig = afficher_graphique(flux_df, article_choisi)
    st.pyplot(fig)

    st.subheader("📊 Indicateurs de Performance (KPI)")
    kpis = calculer_kpis(flux_df)
    st.dataframe(kpis)

    st.subheader("📦 Recommandation d’Approvisionnement (30 jours de couverture)")
    recommandation = recommander_approvisionnement(kpis, 30)
    st.dataframe(recommandation)
