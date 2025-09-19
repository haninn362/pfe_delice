# ============================================================
# Unified Streamlit App
# - Tab 1: Classification (CV², p, Syntetos-Boylan, Optimisation)
# - Tab 2: Forecasting + Grid Search + Best Params + Simulation
# ============================================================

import io
import re
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

try:
    from scipy.stats import nbinom
    _SCIPY_OK = True
except Exception:
    _SCIPY_OK = False

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="Unified Supply Chain App", layout="wide")

# ============================================================
# TAB 1 — Classification + Optimisation
# ============================================================

ADI_CUTOFF = 1.32
P_CUTOFF = 1.0 / ADI_CUTOFF       # ≈ 0.757576
CV2_CUTOFF = 0.49

def choose_method(p: float, cv2: float) -> Tuple[str, str]:
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

    stats_df = (
        pd.DataFrame(stats_rows, columns=["Produit", "moyenne", "écart-type", "CV^2"])
        .set_index("Produit").sort_index()
    )

    counts_rows = []
    for produit, vals in per_product_vals.items():
        n_freq = len(vals)
        p = (n_freq / n_periods) if n_periods else np.nan
        counts_rows.append([produit, n_periods, n_freq, p])

    counts_df = (
        pd.DataFrame(counts_rows, columns=["Produit", "N périodes", "N fréquences", "p"])
        .set_index("Produit").sort_index()
    )

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

def excel_bytes(combined_df, stats_df, counts_df, methods_df) -> io.BytesIO:
    buf = io.BytesIO()
    for engine in ("openpyxl", "xlsxwriter", None):
        try:
            writer = pd.ExcelWriter(buf, engine=engine) if engine else pd.ExcelWriter(buf)
            with writer:
                sheet = "Résultats"
                stats_df.reset_index().to_excel(writer, index=False, sheet_name=sheet, startrow=0, startcol=0)
                r2 = len(stats_df) + 3
                counts_df.reset_index().to_excel(writer, index=False, sheet_name=sheet, startrow=r2, startcol=0)
                r3 = r2 + len(counts_df) + 3
                combined_df.to_excel(writer, index=False, sheet_name=sheet, startrow=r3, startcol=0)
                methods_df.reset_index().to_excel(writer, index=False, sheet_name="Méthodes")
            break
        except ModuleNotFoundError:
            buf = io.BytesIO()
            continue
    buf.seek(0)
    return buf

def _norm(s: str) -> str: return re.sub(r"\s+", " ", str(s).strip().lower())

def _get_excel_bytes(file_like) -> bytes:
    if file_like is None: return b""
    if hasattr(file_like, "getvalue"):
        try: return file_like.getvalue()
        except Exception: pass
    try:
        data = file_like.read()
        return data
    finally:
        try: file_like.seek(0)
        except Exception: pass

def _find_first_col(df: pd.DataFrame, starts_with: str = None, contains: str = None):
    for c in df.columns:
        cn = _norm(c)
        if starts_with and cn.startswith(starts_with): return c
        if contains and contains in cn: return c
    return None

def compute_qr_qw_from_workbook(file_like, conso_sheet_hint: str = "consommation depots externe",
                                time_series_prefix: str = "time seri"):
    info_msgs, warn_msgs = [], []
    if file_like is None:
        return pd.DataFrame(columns=["Code Produit", "n*", "Qr*", "Qw*"]), info_msgs, warn_msgs

    data_bytes = _get_excel_bytes(file_like)
    if not data_bytes:
        warn_msgs.append("Classeur d’optimisation vide ou illisible.")
        return pd.DataFrame(columns=["Code Produit", "n*", "Qr*", "Qw*"]), info_msgs, warn_msgs

    xls = pd.ExcelFile(io.BytesIO(data_bytes))
    sheet_names_norm = {_norm(s): s for s in xls.sheet_names}
    conso_sheet = sheet_names_norm.get(_norm(conso_sheet_hint))
    if not conso_sheet:
        cands = [s for s in xls.sheet_names if _norm(conso_sheet_hint) in _norm(s)]
        if cands: conso_sheet = cands[0]
    if not conso_sheet:
        warn_msgs.append("Feuille 'consommation depots externe' introuvable.")
        return pd.DataFrame(columns=["Code Produit", "n*", "Qr*", "Qw*"]), info_msgs, warn_msgs

    df_conso = pd.read_excel(io.BytesIO(data_bytes), sheet_name=conso_sheet)
    code_col = next((c for c in df_conso.columns if "code produit" in _norm(c)), None) or "Code Produit"
    qty_col = None
    for c in df_conso.columns:
        nc = _norm(c)
        if nc in ("quantite stial", "quantité stial"): qty_col = c; break
    if qty_col is None:
        for c in df_conso.columns:
            nc = _norm(c)
            if "quantite stial" in nc or "quantité stial" in nc: qty_col = c; break
    if qty_col is None:
        for key in ["quantite", "quantité", "qte"]:
            cand = next((c for c in df_conso.columns if key in _norm(c)), None)
            if cand: qty_col = cand; break

    if code_col is None or qty_col is None:
        warn_msgs.append("Colonnes 'Code Produit' et/ou 'Quantite STIAL' introuvables.")
        return pd.DataFrame(columns=["Code Produit", "n*", "Qr*", "Qw*"]), info_msgs, warn_msgs

    conso_series = df_conso.groupby(code_col, dropna=False)[qty_col].sum(numeric_only=True)
    info_msgs.append(f"Feuille de consommation : '{conso_sheet}' (lignes : {len(df_conso)})")
    info_msgs.append(f"Colonne quantité utilisée : '{qty_col}'")

    ts_sheets = [s for s in xls.sheet_names if _norm(s).startswith(_norm(time_series_prefix))]
    if not ts_sheets:
        warn_msgs.append("Aucune feuille 'time serie*' trouvée (ex. 'time serie EM0400').")
        return pd.DataFrame(columns=["Code Produit", "n*", "Qr*", "Qw*"]), info_msgs, warn_msgs

    rows = []
    for sheet in ts_sheets:
        try:
            df = pd.read_excel(io.BytesIO(data_bytes), sheet_name=sheet)
            code_produit = sheet.split()[-1]

            cr_col = _find_first_col(df, starts_with="cr")
            cw_col = _find_first_col(df, starts_with="cw")
            aw_col = _find_first_col(df, starts_with="aw")
            ar_col = _find_first_col(df, starts_with="ar")
            if not all([cr_col, cw_col, aw_col, ar_col]):
                warn_msgs.append(f"[{sheet}] Paramètres CR/CW/AW/AR manquants — ignoré.")
                continue

            C_r = pd.to_numeric(df[cr_col].iloc[0], errors="coerce")
            C_w = pd.to_numeric(df[cw_col].iloc[0], errors="coerce")
            A_w = pd.to_numeric(df[aw_col].iloc[0], errors="coerce")
            A_r = pd.to_numeric(df[ar_col].iloc[0], errors="coerce")
            if any(pd.isna(v) for v in [C_r, C_w, A_w, A_r]) or any(v == 0 for v in [C_w, A_r]):
                warn_msgs.append(f"[{sheet}] Valeurs de paramètres invalides — ignoré.")
                continue

            n = (A_w * C_r) / (A_r * C_w)
            n = 1 if n < 1 else round(n)
            n1, n2 = int(n), int(n) + 1
            F_n1 = (A_r + A_w / n1) * (n1 * C_w + C_r)
            F_n2 = (A_r + A_w / n2) * (n2 * C_w + C_r)
            n_star = n1 if F_n1 <= F_n2 else n2

            D = conso_series.get(code_produit, 0)
            tau = 1
            denom = (n_star * C_w + C_r * tau)
            if denom <= 0:
                warn_msgs.append(f"[{sheet}] Dénominateur non positif pour Q* — ignoré.")
                continue

            if D is None or D <= 0:
                warn_msgs.append(f"[{sheet}] Demande non positive D={D} → Q*=0.")
                Q_r_star = 0.0
            else:
                Q_r_star = ((2 * (A_r + A_w / n_star) * D) / denom) ** 0.5

            Q_w_star = n_star * Q_r_star
            rows.append({
                "Code Produit": str(code_produit),
                "n*": int(n_star),
                "Qr*": round(float(Q_r_star), 2),
                "Qw*": round(float(Q_w_star), 2),
            })
        except Exception as e:
            warn_msgs.append(f"[{sheet}] Échec : {e}")

    result_df = pd.DataFrame(rows).sort_values("Code Produit") if rows else pd.DataFrame(
        columns=["Code Produit", "n*", "Qr*", "Qw*"]
    )
    return result_df, info_msgs, warn_msgs
# ============================================================
# TAB 2 — Forecasting & Best Params
# ============================================================

# ---------- SES ----------
def ses_forecast_array(x, alpha=0.2):
    x = pd.Series(x).fillna(0.0).astype(float).values
    if (x == 0).all():
        return {"forecast_per_period": 0.0, "level": 0.0}
    l = x[0]
    for t in range(1, len(x)):
        l = alpha * x[t] + (1 - alpha) * l
    return {"forecast_per_period": float(l), "level": float(l)}

# ---------- Croston ----------
def croston_forecast_array(x, alpha=0.2):
    x = pd.Series(x).fillna(0.0).astype(float).values
    x = np.where(x < 0, 0.0, x)
    if (x == 0).all():
        return {"forecast_per_period": 0.0, "z_t": 0.0, "p_t": float("inf")}
    nz_idx = [i for i, v in enumerate(x) if v > 0]
    first = nz_idx[0]
    z = x[first]
    if len(nz_idx) >= 2:
        p = sum([j - i for i, j in zip(nz_idx[:-1], nz_idx[1:])]) / len(nz_idx)
    else:
        p = len(x) / len(nz_idx)
    periods_since_demand = 0
    for t in range(first + 1, len(x)):
        periods_since_demand += 1
        if x[t] > 0:
            I_t = periods_since_demand
            z = alpha * x[t] + (1 - alpha) * z
            p = alpha * I_t + (1 - alpha) * p
            periods_since_demand = 0
    f = z / p
    return {"forecast_per_period": float(f), "z_t": float(z), "p_t": float(p)}

# ---------- SBA ----------
def sba_forecast_array(x, alpha=0.2):
    crost = croston_forecast_array(x, alpha)
    f = crost["forecast_per_period"] * (1 - alpha / 2.0)
    return {"forecast_per_period": float(f), "z_t": crost["z_t"], "p_t": crost["p_t"]}

# ---------- Load matrix ----------
def load_matrix_timeseries(excel_path: str, sheet_name: str):
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    prod_col = df.columns[0]
    new_cols = [prod_col]
    for c in df.columns[1:]:
        try:
            new_cols.append(pd.to_datetime(c))
        except:
            new_cols.append(c)
    df.columns = new_cols
    return df, prod_col

# ---------- Rolling function (generic) ----------
def rolling_with_new_logic(
    excel_path: str,
    product_code: str,
    sheet_name: str,
    method: str,
    alpha: float,
    window_ratio: float,
    interval: int,
):
    df, prod_col = load_matrix_timeseries(excel_path, sheet_name)
    row = df.loc[df[prod_col] == product_code]
    if row.empty:
        raise ValueError(f"Produit '{product_code}' introuvable dans '{sheet_name}'.")
    series = row.drop(columns=[prod_col]).T.squeeze()
    series.index = pd.to_datetime(series.index)
    series = series.sort_index()
    full_idx = pd.date_range(series.index.min(), series.index.max(), freq="D")
    daily = series.reindex(full_idx, fill_value=0.0).astype(float)
    values = daily.values
    split_index = int(len(values) * window_ratio)
    if split_index < 2:
        return pd.DataFrame()
    out_rows = []
    for i in range(split_index, len(values)):
        if (i - split_index) % interval == 0:
            train = values[:i]
            test_date = daily.index[i]
            real_demand = float(values[i])
            if method == "SES":
                fdict = ses_forecast_array(train, alpha=alpha)
            elif method == "CROSTON":
                fdict = croston_forecast_array(train, alpha=alpha)
            elif method == "SBA":
                fdict = sba_forecast_array(train, alpha=alpha)
            else:
                raise ValueError("Méthode non supportée.")
            f = float(fdict["forecast_per_period"])
            out_rows.append({
                "date": test_date.date(),
                "real_demand": real_demand,
                "forecast_per_period": f,
                "forecast_error": float(real_demand - f),
            })
    return pd.DataFrame(out_rows)

# ---------- Metrics ----------
def compute_metrics(df_run: pd.DataFrame):
    if df_run.empty or "forecast_error" not in df_run:
        return np.nan, np.nan, np.nan, np.nan
    e = df_run["forecast_error"].astype(float)
    ME = e.mean()
    absME = e.abs().mean()
    MSE = (e**2).mean()
    RMSE = np.sqrt(MSE)
    return ME, absME, MSE, RMSE

# ---------- Grid search ----------
def grid_search_methods(excel_path, sheet_name, product_codes, method, alphas, windows, intervals):
    all_results = []
    best_rows_per_code = []
    for code in product_codes:
        metrics_rows = []
        for a in alphas:
            for w in windows:
                for itv in intervals:
                    df_run = rolling_with_new_logic(
                        excel_path=excel_path,
                        product_code=code,
                        sheet_name=sheet_name,
                        method=method,
                        alpha=a,
                        window_ratio=w,
                        interval=itv,
                    )
                    ME, absME, MSE, RMSE = compute_metrics(df_run)
                    row = {
                        "code": code,
                        "alpha": a,
                        "window_ratio": w,
                        "interval": itv,
                        "ME": ME,
                        "absME": absME,
                        "MSE": MSE,
                        "RMSE": RMSE,
                        "n_points": len(df_run),
                    }
                    metrics_rows.append(row)
                    all_results.append(row)
        df_metrics = pd.DataFrame(metrics_rows)
        if df_metrics.empty:
            continue
        best_idx = (df_metrics["RMSE"]).idxmin()
        best = df_metrics.loc[best_idx]
        best_rows_per_code.append({
            "code": code,
            "best_alpha": best["alpha"],
            "best_window": best["window_ratio"],
            "best_interval": best["interval"],
            "best_RMSE": best["RMSE"],
            "n_points": best["n_points"],
        })
    df_all = pd.DataFrame(all_results)
    df_best = pd.DataFrame(best_rows_per_code)
    return df_all, df_best
# ============================================================
# MAIN STREAMLIT APP — Unified Tabs
# ============================================================

import streamlit as st

st.set_page_config(page_title="Demand Classification & Forecasting", layout="wide")

st.title("📊 Unified App — Classification & Forecasting")

# Tabs
tab1, tab2 = st.tabs(["🔎 Classification", "📈 Forecasting & Best Params"])

# --------------------- TAB 1 ---------------------
with tab1:
    st.header("Classification (p vs CV²)")
    uploaded = st.file_uploader("Téléverser le classeur de classification", type=["xlsx", "xls"], key="clf")
    uploaded_opt = st.file_uploader("Classeur optimisation (optionnel)", type=["xlsx", "xls"], key="opt")
    if uploaded is not None:
        try:
            import pandas as pd
            xls = pd.ExcelFile(uploaded)
            sheet_name = st.selectbox("Choisir la feuille", xls.sheet_names)
            if sheet_name:
                compute_and_show(uploaded, sheet_name, uploaded_opt)
        except Exception as e:
            st.error(f"Erreur lecture fichier : {e}")
    else:
        st.info("➡️ Téléversez un fichier Excel de classification.")

# --------------------- TAB 2 ---------------------
with tab2:
    st.header("Forecasting & Best Params")

    st.markdown("### Paramètres")
    excel_file = st.file_uploader("Fichier Excel (time series)", type=["xlsx"], key="forecast")
    sheet_name = st.text_input("Nom de la feuille", "Feuil1")
    product_codes = st.text_area("Codes produits (séparés par virgule)", "EM0392, EM0400, EM1091").split(",")
    product_codes = [c.strip() for c in product_codes if c.strip()]

    alphas = st.multiselect("Alphas", [0.1, 0.2, 0.3, 0.4, 0.5], default=[0.1, 0.4])
    windows = st.multiselect("Window ratios", [0.6, 0.7, 0.8], default=[0.6, 0.8])
    intervals = st.multiselect("Intervals", [5, 10, 20], default=[5, 20])

    method = st.selectbox("Méthode", ["SES", "CROSTON", "SBA"])

    if st.button("🚀 Lancer Grid Search"):
        if excel_file is None:
            st.error("Veuillez téléverser un fichier Excel.")
        else:
            df_all, df_best = grid_search_methods(
                excel_path=excel_file,
                sheet_name=sheet_name,
                product_codes=product_codes,
                method=method,
                alphas=alphas,
                windows=windows,
                intervals=intervals,
            )
            st.subheader("✅ Résultats complets")
            st.dataframe(df_all, use_container_width=True)
            st.subheader("⭐ Best Params")
            st.dataframe(df_best, use_container_width=True)

            # Téléchargement
            import io
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
                df_all.to_excel(writer, sheet_name="All_Results", index=False)
                df_best.to_excel(writer, sheet_name="Best_Params", index=False)
            buf.seek(0)
            st.download_button(
                "Télécharger résultats Excel",
                data=buf,
                file_name="forecasting_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
# ============================================================
# FINAL SIMULATION & SENSITIVITY
# ============================================================

import numpy as np
import pandas as pd
import streamlit as st

# ------------------- Simulation Logic -------------------
def run_final_simulation(df_best_params: pd.DataFrame, service_levels=[0.90, 0.92, 0.95, 0.98]):
    """
    Run final simulation for each product with different service levels.
    Returns:
        per_code_results: dict of DataFrames (detailed per product)
        summary: aggregated results by service level
    """
    per_code_results = {}
    summary_rows = []

    for _, row in df_best_params.iterrows():
        code = row["code"]
        alpha = row["alpha"]
        window = row["window_ratio"]
        interval = row["recalc_interval"]
        method = row["method"]
        Qr_star = row.get("Qr*", 50.0)  # dummy placeholder
        Qw_star = row.get("Qw*", 100.0) # dummy placeholder
        n_star = row.get("n*", 2)

        # Generate fake demand & inventory trajectory
        dates = pd.date_range("2024-08-01", periods=10, freq="20D")
        demand = np.random.randint(100, 5000, size=len(dates))
        stock = np.zeros(len(dates))
        stock[0] = 5000
        actions, status, sl = [], [], []

        for t in range(len(dates)):
            real = demand[t]
            if t > 0:
                stock[t] = stock[t-1]
            stock[t] -= real
            if stock[t] < 0:
                status.append("rupture")
            else:
                status.append("holding")

            if stock[t] < 2000:
                actions.append(f"commander_Qr*_{Qr_star}")
                stock[t] += Qr_star
            else:
                actions.append("pas_de_commande")

            sl.append(0.95)  # just fixed for now

        df_detail = pd.DataFrame({
            "date": dates,
            "code": code,
            "methode": method,
            "intervalle": interval,
            "demande_reelle": demand,
            "stock_apres_intervalle": stock,
            "politique_commande": actions,
            "Qr_etoile": Qr_star,
            "Qw_etoile": Qw_star,
            "n_etoile": n_star,
            "statut_stock": status,
            "service_level": sl,
        })
        per_code_results[code] = df_detail

        # Aggregate for each service level
        for slv in service_levels:
            summary_rows.append([
                code, slv,
                stock.mean(), stock.std(),
                stock.mean() * 1.1, stock.std() * 1.1,
                (status.count("holding")/len(status))*100,
                (status.count("rupture")/len(status))*100,
                Qr_star, Qw_star, n_star
            ])

    summary = pd.DataFrame(summary_rows, columns=[
        "code","service_level","ROP_u_moy","SS_u_moy","ROP_f_moy","SS_f_moy",
        "holding_pct","rupture_pct","Qr_star","Qw_star","n_star"
    ])
    return per_code_results, summary

# ------------------- Streamlit UI -------------------
def show_final_simulation_ui(df_best_params):
    st.subheader("▶️ Recalcul final au niveau de service")

    if st.button("Run Final Simulation"):
        per_code, summary = run_final_simulation(df_best_params)

        # Detailed results per product
        for code, df_detail in per_code.items():
            st.markdown(f"### === {code} — {df_detail['methode'].iloc[0]} (SL=0.95) ===")
            st.dataframe(df_detail, use_container_width=True)

        # Summary across service levels
        st.markdown("📊 Résumé global — moyennes par code et niveau de service")
        st.dataframe(summary, use_container_width=True)

        # Download option
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
            for code, df_detail in per_code.items():
                df_detail.to_excel(writer, sheet_name=code, index=False)
            summary.to_excel(writer, sheet_name="Summary", index=False)
        buf.seek(0)
        st.download_button(
            "Télécharger résultats simulation (Excel)",
            data=buf,
            file_name="simulation_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
