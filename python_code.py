# ==================================================
# APPLICATION STREAMLIT - PFE HANIN
# Base Stock + Prévisions (SES / Croston / SBA)
# Sélection meilleure méthode + Simulation commandes
# + Analyse de sensibilité
# ==================================================

import numpy as np
import pandas as pd
from scipy.stats import nbinom
import streamlit as st
import matplotlib.pyplot as plt

# ---------- PARAMÈTRES ----------
st.set_page_config(page_title="PFE HANIN - Base Stock", layout="wide")

st.title("📦 Application PFE HANIN")
st.markdown("Méthode **Base Stock** + Prévisions (SES / Croston / SBA)")

# Sidebar : paramètres
st.sidebar.header("⚙️ Paramètres")
uploaded_file = st.sidebar.file_uploader("Chargez le fichier Excel", type=["xlsx"])

default_products = ["EM0400","EM1499","EM1091","EM1523","EM0392","EM1526"]
PRODUCT_CODE = st.sidebar.selectbox("Choisir le produit", default_products, index=0)

# Fixed parameters
NB_SIM = 1000
RNG_SEED = 42
LEAD_TIME = 10
LEAD_TIME_SUPPLIER = 3
SERVICE_LEVEL = 0.95

ALPHAS = [0.1, 0.2, 0.3, 0.4]
WINDOW_RATIOS = [0.6, 0.7, 0.8]
RECALC_INTERVALS = [5, 10, 20]
SERVICE_LEVELS = [0.90, 0.92, 0.95, 0.98]

# ==================================================
# PARTIE 1 : Qr*, Qw* et n* (Base Stock)
# ==================================================
def _find_product_sheet(excel_path, code: str) -> str:
    xls = pd.ExcelFile(excel_path)
    sheets = [s.strip().lower() for s in xls.sheet_names]
    targets = [f"time serie {code}".lower(), f"time series {code}".lower(), code.lower()]
    for t in targets:
        if t in sheets:
            return xls.sheet_names[sheets.index(t)]
    for s in sheets:
        if code.lower() in s:
            return xls.sheet_names[sheets.index(s)]
    raise ValueError(f"[Sheet] Onglet pour '{code}' introuvable.")

def compute_qstars(file_path, product_code):
    df_conso = pd.read_excel(file_path, sheet_name="consommation depots externe")
    df_conso = df_conso.groupby('Code Produit')['Quantite STIAL'].sum()
    qr_map, qw_map, n_map = {}, {}, {}
    for code in [product_code]:
        sheet = _find_product_sheet(file_path, code)
        df = pd.read_excel(file_path, sheet_name=sheet)
        C_r = df['Cr : cout stockage/article '].iloc[0]
        C_w = df['Cw : cout stockage\nchez F'].iloc[0]
        A_w = df['Aw : cout de\nlancement chez U'].iloc[0]
        A_r = df['Ar : cout de \nlancement chez F'].iloc[0]

        n = (A_w * C_r) / (A_r * C_w)
        n = 1 if n < 1 else round(n)
        n1, n2 = int(n), int(n) + 1
        F_n1 = (A_r + A_w / n1) * (n1 * C_w + C_r)
        F_n2 = (A_r + A_w / n2) * (n2 * C_w + C_r)
        n_star = n1 if F_n1 <= F_n2 else n2

        D = df_conso.get(code, 0)
        tau = 1
        Qr_star = ((2 * (A_r + A_w / n_star) * D) / (n_star * C_w + C_r * tau)) ** 0.5
        Qw_star = n_star * Qr_star

        qr_map[code] = round(Qr_star, 2)
        qw_map[code] = round(Qw_star, 2)
        n_map[code] = n_star

    return qr_map, qw_map, n_map

# ==================================================
# PARTIE 2 : Prévisions (SES / Croston / SBA)
# ==================================================
def croston_or_sba_forecast(x, alpha=0.2, variant="croston"):
    x = pd.Series(x).fillna(0.0).astype(float).values
    if (x == 0).all():
        return 0.0
    nz_idx = [i for i, v in enumerate(x) if v > 0]
    z, p = x[nz_idx[0]], len(x)/len(nz_idx)
    psd = 0
    for t in range(nz_idx[0]+1, len(x)):
        psd += 1
        if x[t] > 0:
            I_t = psd
            z = alpha * x[t] + (1-alpha) * z
            p = alpha * I_t + (1-alpha) * p
            psd = 0
    f = z / p
    if variant == "sba":
        f *= (1 - alpha/2.0)
    return f

def ses_forecast(x, alpha=0.2):
    x = pd.Series(x).fillna(0.0).astype(float).values
    if len(x) == 0:
        return 0.0
    l = x[0]
    for t in range(1, len(x)):
        l = alpha * x[t] + (1 - alpha) * l
    return l

# ==================================================
# PARTIE 3 : Simulation finale avec ROP
# ==================================================
def _interval_sum_next_days(daily: pd.Series, start_idx: int, interval: int) -> float:
    s, e = start_idx + 1, start_idx + 1 + interval
    return float(pd.Series(daily).iloc[s:e].sum())

def simulate_orders(file_path, best_per_code, qr_map, service_level=SERVICE_LEVEL):
    results = []
    rng = np.random.default_rng(RNG_SEED)
    for _, row in best_per_code.iterrows():
        code = row["code"]; method = row["method"]
        alpha = row["alpha"]; window_ratio = row["window_ratio"]; interval = int(row["recalc_interval"])
        sheet = _find_product_sheet(file_path, code)
        df = pd.read_excel(file_path, sheet_name=sheet)
        dates = pd.to_datetime(df.iloc[:,0], errors="coerce")
        stock_col = pd.to_numeric(df.iloc[:,1], errors="coerce").astype(float)
        cons_col = pd.to_numeric(df.iloc[:,2], errors="coerce").fillna(0.0).astype(float)
        ts_cons = pd.DataFrame({"d":dates,"q":cons_col}).dropna().sort_values("d").set_index("d")["q"]
        ts_stock = pd.DataFrame({"d":dates,"s":stock_col}).dropna().sort_values("d").set_index("d")["s"]
        full_idx = pd.date_range(ts_cons.index.min(), ts_cons.index.max(), freq="D")
        cons_daily = ts_cons.reindex(full_idx, fill_value=0.0)
        stock_daily = ts_stock.reindex(full_idx).ffill().fillna(0.0)
        vals = cons_daily.values
        split_index = int(len(vals) * window_ratio)
        if split_index < 2: continue
        stock_after_interval = 0.0
        for i in range(split_index, len(vals)):
            if (i - split_index) % interval == 0:
                train = vals[:i]
                if method == "ses": f = ses_forecast(train, alpha)
                elif method == "croston": f = croston_or_sba_forecast(train, alpha, "croston")
                else: f = croston_or_sba_forecast(train, alpha, "sba")
                sigma_period = float(pd.Series(train).std(ddof=1)) if i > 1 else 0.0
                sigma_period = sigma_period if np.isfinite(sigma_period) else 0.0
                X_Lt = LEAD_TIME * f
                sigma_Lt = sigma_period * np.sqrt(max(LEAD_TIME, 1e-9))
                var_u = sigma_Lt**2 if sigma_Lt**2 > X_Lt else X_Lt+1e-5
                p_nb = min(max(X_Lt/var_u, 1e-12),1-1e-12)
                r_nb = X_Lt**2/(var_u - X_Lt) if var_u > X_Lt else 1e6
                ROP_u = float(np.percentile(nbinom.rvs(r_nb, p_nb, size=NB_SIM, random_state=rng), 100*service_level))
                real_demand = _interval_sum_next_days(cons_daily, i, interval)
                stock_on_hand_running = _interval_sum_next_days(stock_daily, i, interval)
                stock_after_interval = stock_after_interval + stock_on_hand_running - real_demand
                stock_status = "rupture" if real_demand > ROP_u else "holding"
                results.append({
                    "code": code, "method": method,
                    "holding_pct": (stock_status=="holding")*100,
                    "rupture_pct": (stock_status=="rupture")*100,
                    "service_level": service_level
                })
    return pd.DataFrame(results)

# ==================================================
# PARTIE 4 : Sensibilité & Plot
# ==================================================
def run_sensitivity_with_methods(file_path, qr_map):
    all_results = []
    for method in ["ses","croston","sba"]:
        best_per_code = pd.DataFrame([{"code": PRODUCT_CODE, "method": method,
                                       "alpha": 0.2, "window_ratio": 0.7, "recalc_interval": 10}])
        for sl in SERVICE_LEVELS:
            df_run = simulate_orders(file_path, best_per_code, qr_map, service_level=sl)
            if not df_run.empty:
                summary = df_run.groupby(["code","method"]).agg(
                    holding_pct=("holding_pct","mean"),
                    rupture_pct=("rupture_pct","mean")
                ).reset_index()
                summary["service_level"] = sl
                all_results.append(summary)
    return pd.concat(all_results, ignore_index=True)

def plot_tradeoff(df_summary):
    if df_summary.empty:
        st.warning("Pas de résultats pour tracer la sensibilité.")
        return
    
    plt.figure(figsize=(8,6))
    colors = {"ses":"blue", "croston":"green", "sba":"red"}
    
    for method, subset in df_summary.groupby("method"):
        subset = subset.sort_values("service_level")
        distances = np.sqrt(subset["holding_pct"]**2 + subset["rupture_pct"]**2)
        norm = (distances - distances.min()) / (distances.max() - distances.min() + 1e-9)
        alphas = 1 - norm
        
        plt.plot(subset["holding_pct"], subset["rupture_pct"],
                 color=colors[method], alpha=0.7, linewidth=2, label=method)
        
        for i, row in subset.iterrows():
            plt.scatter(row["holding_pct"], row["rupture_pct"],
                        color=colors[method], s=80, alpha=alphas.loc[i])
            plt.text(row["holding_pct"]+0.5, row["rupture_pct"]+0.5,
                     f"{row['service_level']:.2f}", fontsize=8, color=colors[method])
    
    plt.xlabel("Holding %")
    plt.ylabel("Rupture %")
    plt.title("Trade-off Holding vs Rupture (%) – 3 Methods, 4 SL")
    plt.legend()
    st.pyplot(plt)

# ==================================================
# MAIN STREAMLIT
# ==================================================
if uploaded_file is not None:
    with st.spinner("⏳ Calcul en cours..."):
        qr_map, qw_map, n_map = compute_qstars(uploaded_file, PRODUCT_CODE)
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Base Stock", "📈 Sensibilité", "📦 Simulation", "🔮 Prévisions"])

        with tab1:
            st.subheader("Qr*, Qw* et n* (Base Stock)")
            df_base = pd.DataFrame({
                "Produit": list(qr_map.keys()),
                "Qr*": list(qr_map.values()),
                "Qw*": list(qw_map.values()),
                "n*": list(n_map.values())
            })
            st.dataframe(df_base)

        with tab2:
            st.subheader("Analyse de sensibilité")
            sensitivity_summary = run_sensitivity_with_methods(uploaded_file, qr_map)
            st.dataframe(sensitivity_summary)
            plot_tradeoff(sensitivity_summary)

        with tab3:
            st.subheader("Simulation (exemple SL fixe)")
            st.write("À implémenter selon besoin…")

        with tab4:
            st.subheader("Prévisions (SES / Croston / SBA)")
            st.write("À implémenter selon besoin…")
else:
    st.info("📥 Veuillez charger un fichier Excel pour commencer.")
