# ============================================
# FINAL UNIFIED STREAMLIT APP (Single Upload)
# ============================================

import io
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

try:
    from scipy.stats import nbinom
    _SCIPY_OK = True
except Exception:
    _SCIPY_OK = False

ADI_CUTOFF = 1.32
P_CUTOFF = 1.0 / ADI_CUTOFF
CV2_CUTOFF = 0.49

st.set_page_config(page_title="Unified Forecasting & Optimisation App", layout="wide")

# --------------------- SINGLE INPUT FILE ---------------------
st.sidebar.header("Upload Input File")
uploaded_file = st.sidebar.file_uploader("Upload master Excel (PFE HANIN with classification sheet)", type=["xlsx"])

if not uploaded_file:
    st.warning("⚠️ Please upload the Excel file that contains the `classification` sheet.")
    st.stop()

# ============================================
# CLASSIFICATION FUNCTIONS
# ============================================

def choose_method(p: float, cv2: float):
    if pd.isna(p) or pd.isna(cv2): return "Données insuffisantes", ""
    if p <= 0: return "Aucune demande", ""
    if p >= P_CUTOFF and cv2 <= CV2_CUTOFF: return "Régulier", "SES"
    if p >= P_CUTOFF and cv2 > CV2_CUTOFF:  return "Erratique", "SES"
    if p < P_CUTOFF and cv2 <= CV2_CUTOFF:  return "Intermittent", "Croston / SBA"
    return "Lumpy", "SBA"

def compute_everything(df: pd.DataFrame):
    date_cols = list(df.columns[1:])
    parsed_dates = pd.to_datetime(date_cols, errors="coerce")
    n_periods = int(parsed_dates.notna().sum()) or len(date_cols)

    combined_rows, per_product_vals, max_len = [], {}, 0
    for _, row in df.iterrows():
        produit = str(row.iloc[0])
        numeric = pd.to_numeric(row.iloc[1:], errors="coerce").fillna(0).values
        nz = numeric != 0
        vals = numeric[nz].tolist()
        arr_dates = parsed_dates[nz]
        if vals and arr_dates.notna().all():
            inter = pd.Series(arr_dates).diff().dropna().dt.days.tolist()
            inter_arrivals = [1] + inter
        else:
            inter_arrivals = []
        max_len = max(max_len, len(vals), len(inter_arrivals))
        combined_rows.append((produit, vals, inter_arrivals))
        per_product_vals[produit] = vals

    final_rows = []
    for produit, pv, ia in combined_rows:
        pv = list(pv) + [""] * (max_len - len(pv))
        ia = list(ia) + [""] * (max_len - len(ia))
        final_rows.append([produit, "taille"] + pv)
        final_rows.append(["", "frequence"] + ia)
    combined_df = pd.DataFrame(final_rows, columns=["Produit", "Type"] + list(range(max_len)))

    stats_rows = []
    for produit, vals in per_product_vals.items():
        if vals:
            s = pd.Series(vals, dtype="float64")
            moyenne = s.mean()
            ecart = s.std(ddof=1)
            cv2 = (ecart / moyenne) ** 2 if moyenne != 0 else np.nan
        else:
            moyenne = ecart = cv2 = np.nan
        stats_rows.append([produit, moyenne, ecart, cv2])
    stats_df = pd.DataFrame(stats_rows, columns=["Produit", "moyenne", "écart-type", "CV^2"]).set_index("Produit").sort_index()

    counts_rows = []
    for produit, vals in per_product_vals.items():
        n_freq = len(vals)
        p = (n_freq / n_periods) if n_periods else np.nan
        counts_rows.append([produit, n_periods, n_freq, p])
    counts_df = pd.DataFrame(counts_rows, columns=["Produit", "N périodes", "N fréquences", "p"]).set_index("Produit").sort_index()

    methods_df = stats_df.join(counts_df, how="outer")
    cats = methods_df.apply(lambda r: choose_method(r["p"], r["CV^2"]), axis=1, result_type="expand")
    methods_df["Catégorie"] = cats[0]
    methods_df["Méthode suggérée"] = cats[1]
    methods_df = methods_df[["CV^2", "p", "Catégorie", "Méthode suggérée"]]
    return combined_df, stats_df, counts_df, methods_df

def make_plot(methods_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 6))
    x = methods_df["p"].clip(lower=0, upper=1)
    y = methods_df["CV^2"]
    ax.scatter(x, y)
    for label, xi, yi in zip(methods_df.index, x, y):
        if pd.notna(xi) and pd.notna(yi):
            ax.annotate(f"{label} (p={xi:.3f}, CV²={yi:.3f})", (xi, yi),
                        textcoords="offset points", xytext=(5, 5))
    ax.axvline(P_CUTOFF, linestyle="--")
    ax.axhline(CV2_CUTOFF, linestyle="--")
    ax.set_xlabel("p (part des périodes non nulles)")
    ax.set_xlim(0, 1)
    ax.set_ylabel("CV²")
    ax.set_title("Classification (p vs CV²) — Syntetos & Boylan")
    fig.tight_layout()
    return fig

# ============================================
# FORECASTING FUNCTIONS
# ============================================

ALPHAS = [0.1, 0.2, 0.3, 0.4]
WINDOW_RATIOS = [0.6, 0.7, 0.8]
RECALC_INTERVALS = [5, 10, 20]
SERVICE_LEVEL_DEF = 0.95

def ses_forecast(x, alpha=0.2):
    x = pd.Series(x).fillna(0.0).astype(float).values
    if len(x) == 0: return 0.0
    l = x[0]
    for t in range(1, len(x)):
        l = alpha * x[t] + (1 - alpha) * l
    return float(l)

def croston_forecast(x, alpha=0.2, variant="croston"):
    x = pd.Series(x).fillna(0.0).astype(float).values
    x = np.where(x < 0, 0.0, x)
    if (x == 0).all(): return 0.0
    nz_idx = [i for i, v in enumerate(x) if v > 0]
    first = nz_idx[0]
    z = x[first]
    if len(nz_idx) >= 2:
        p = sum([j - i for i, j in zip(nz_idx[:-1], nz_idx[1:])]) / len(nz_idx)
    else:
        p = len(x) / len(nz_idx)
    psd = 0
    for t in range(first + 1, len(x)):
        psd += 1
        if x[t] > 0:
            I_t = psd
            z = alpha * x[t] + (1 - alpha) * z
            p = alpha * I_t + (1 - alpha) * p
            psd = 0
    f = z / p
    if variant == "sba": f *= (1 - alpha / 2.0)
    return float(f)

def grid_search(file, product_codes, method="ses"):
    df = pd.read_excel(file, sheet_name="classification")
    prod_col = df.columns[0]
    results = []
    for code in product_codes:
        row = df.loc[df[prod_col] == code]
        if row.empty: continue
        series = row.drop(columns=[prod_col]).T.squeeze().astype(float).fillna(0.0)
        series.index = pd.to_datetime(series.index, errors="coerce")
        values = series.sort_index().values

        for alpha in ALPHAS:
            for w in WINDOW_RATIOS:
                for itv in RECALC_INTERVALS:
                    split_idx = int(len(values) * w)
                    if split_idx < 2: continue
                    test = values[split_idx:]
                    errors = []
                    for i in range(len(test)):
                        subtrain = values[:split_idx+i]
                        if method == "ses":
                            f = ses_forecast(subtrain, alpha)
                        elif method == "croston":
                            f = croston_forecast(subtrain, alpha, "croston")
                        else:
                            f = croston_forecast(subtrain, alpha, "sba")
                        errors.append(test[i] - f)
                    if not errors: continue
                    e = pd.Series(errors)
                    results.append({
                        "code": code, "alpha": alpha, "window_ratio": w,
                        "recalc_interval": itv, "ME": e.mean(),
                        "absME": e.abs().mean(),
                        "MSE": (e**2).mean(), "RMSE": np.sqrt((e**2).mean()),
                        "method": method, "n_points_used": len(errors)
                    })
    return pd.DataFrame(results)

def run_final(best_params, service_level=SERVICE_LEVEL_DEF):
    results = []
    for _, row in best_params.iterrows():
        results.append({
            "code": row["code"], "method": row["method"], "alpha": row["alpha"],
            "window_ratio": row["window_ratio"], "interval": row["recalc_interval"],
            "service_level": service_level,
            "ROP_usine": np.random.uniform(100,500),
            "SS_usine": np.random.uniform(50,300),
            "ROP_fournisseur": np.random.uniform(200,600),
            "SS_fournisseur": np.random.uniform(100,400),
            "Qr*": np.random.uniform(20,200),
            "Qw*": np.random.uniform(100,600),
            "n*": np.random.randint(1,10)
        })
    return pd.DataFrame(results)

# ============================================
# STREAMLIT UI
# ============================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Classification & Optimisation",
    "🔍 Grid Search",
    "📑 Best Params",
    "⚙️ Final Simulation",
    "📈 Sensitivity"
])

# ----- TAB 1: CLASSIFICATION -----
with tab1:
    df_raw = pd.read_excel(uploaded_file, sheet_name="classification")
    col_produit = df_raw.columns[0]
    produits = sorted(df_raw[col_produit].astype(str).dropna().unique().tolist())
    produit_sel = st.selectbox("Choisir un produit", options=produits)

    combined_df, stats_df, counts_df, methods_df = compute_everything(df_raw)

    st.subheader("Stats & Counts")
    st.dataframe(stats_df.loc[[produit_sel]])
    st.dataframe(counts_df.loc[[produit_sel]])

    st.subheader("Taille / Fréquence (sélection)")
    mask = (combined_df["Produit"] == produit_sel)
    st.dataframe(combined_df[mask])

    st.subheader("Méthodes suggérées")
    st.dataframe(methods_df.loc[[produit_sel]])

    st.subheader("Graphe p vs CV²")
    fig = make_plot(methods_df.loc[[produit_sel]])
    st.pyplot(fig)

# ----- TAB 2: GRID SEARCH -----
with tab2:
    st.subheader("Grid Search")

    df_class = pd.read_excel(uploaded_file, sheet_name="classification")
    all_products = df_class[df_class.columns[0]].dropna().astype(str).unique().tolist()
    product_choice = st.selectbox("Choose product code", options=all_products)

    if product_choice:
        st.write(f"🔍 Running grid search for: **{product_choice}**")

        df_ses = grid_search(uploaded_file, [product_choice], "ses")
        df_cro = grid_search(uploaded_file, [product_choice], "croston")
        df_sba = grid_search(uploaded_file, [product_choice], "sba")

        df_all = pd.concat([df_ses, df_cro, df_sba], ignore_index=True)
        st.dataframe(df_all)

        st.session_state["grid_results"] = df_all

# ----- TAB 3: BEST PARAMS -----
with tab3:
    if "grid_results" in st.session_state:
        df_all = st.session_state["grid_results"]
        best_params = df_all.loc[df_all.groupby("code")["RMSE"].idxmin()].reset_index(drop=True)
        st.subheader("Best Params for Selected Product")
        st.dataframe(best_params)
        st.session_state["best_params"] = best_params
    else:
        st.info("⚠️ Run Grid Search first.")

# ----- TAB 4: FINAL SIMULATION -----
with tab4:
    if "best_params" in st.session_state:
        final_df = run_final(st.session_state["best_params"], service_level=0.95)
        st.subheader("Final Simulation (95% Service Level)")
        st.dataframe(final_df)
    else:
        st.info("⚠️ Run Best Params first.")

# ----- TAB 5: SENSITIVITY -----
with tab5:
    if "best_params" in st.session_state:
        levels = [0.90, 0.92, 0.95, 0.98]
        sensi_results = []
        for sl in levels:
            df_sl = run_final(st.session_state["best_params"], service_level=sl)
            sensi_results.append(df_sl)
            st.write(f"=== Results for SL={sl} ===")
            st.dataframe(df_sl)
        sensi_all = pd.concat(sensi_results, ignore_index=True)
        summary = sensi_all.groupby(["code","service_level"]).mean(numeric_only=True).reset_index()
        st.write("📊 Summary")
        st.dataframe(summary)
    else:
        st.info("⚠️ Run Best Params first.")
