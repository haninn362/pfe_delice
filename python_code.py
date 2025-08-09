# --- Upload unique fichier ---
uploaded_file = st.file_uploader(
    "📂 Importez votre fichier Excel (avec 2 feuilles : Réception & Consommation)",
    type=["xlsx", "xls", "xlsb"]
)

if not uploaded_file:
    st.info("Importez le fichier Excel pour continuer.")
    st.stop()

# --- Sélection des feuilles ---
try:
    # Lister les noms de feuilles
    xls = pd.ExcelFile(uploaded_file)
    sheets = xls.sheet_names
    st.write("Feuilles détectées :", sheets)

    sheet_rec = st.selectbox("Feuille Réception :", sheets, index=0)
    sheet_conso = st.selectbox("Feuille Consommation :", sheets, index=1 if len(sheets) > 1 else 0)

    # Lire les 2 feuilles
    reception_df = pd.read_excel(uploaded_file, sheet_name=sheet_rec, engine="openpyxl")
    consommation_df = pd.read_excel(uploaded_file, sheet_name=sheet_conso, engine="openpyxl")

except Exception as e:
    st.error(f"Erreur de lecture du fichier Excel : {e}")
    st.stop()

# --- Normaliser colonnes si besoin ---
for df in [reception_df, consommation_df]:
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if "Quantity" in df.columns:
        df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")

# Maintenant, vous pouvez continuer avec :
flux_df = construire_flux(reception_df, consommation_df)
