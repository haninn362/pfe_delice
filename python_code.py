# app.py
# ============================================
# Streamlit App: SBA + Croston + SES
# Grid Search (ME, MSE, RMSE) + Final Recalc
# ============================================

import numpy as np
import pandas as pd
import re
from scipy.stats import nbinom
import streamlit as st

# ---------- GLOBAL PARAMETERS ----------
PRODUCT_CODES_UNI = ["EM0400", "EM1499", "EM1091", "EM1523", "EM0392", "EM1526"]

ALPHAS_UNI = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4]
WINDOW_RATIOS_UNI = [0.6, 0.7, 0.8]
RECALC_INTERVALS_UNI = [1, 2, 5, 10, 15, 20]

LEAD_TIME_UNI = 1
LEAD_TIME_SUPPLIER_UNI = 3
SERVICE_LEVEL_UNI = 0.95
NB_SIM_UNI = 800
RNG_SEED_UNI = 42

DISPLAY_COLUMNS_UNI = [
    "date", "code", "interval", "real_demand", "stock_on_hand_running",
    "stock_after_interval", "can_cover_interval", "order_policy",
    "reorder_point_usine", "lead_time_usine_days", "lead_time_supplier_days",
    "reorder_point_fournisseur", "stock_status", "rop_usine_minus_real_running"
]

# =====================================================================
# =========================  SBA SECTION  ==============================
# =====================================================================

def _find_product_sheet_sba(excel_path: str, code: str) -> str:
    xls = pd.ExcelFile(excel_path)
    sheets = xls.sheet_names
    target = f"time serie {code}"
    if target in sheets:
        return target
    patt = re.compile(r"time\s*ser(i|ie)s?\s*", re.IGNORECASE)
    cand = [s for s in sheets if patt.search(s) and code.lower() in s.lower()]
    if cand:
        return sorted(cand, key=len, reverse=True)[0]
    for s in sheets:
        if s.strip().lower() == code.lower():
            return s
    raise ValueError(f"[SBA] Sheet for '{code}' not found (expected: 'time serie {code}').")

def _daily_consumption_and_stock_sba(excel_path: str, sheet_name: str):
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    date_col, stock_col, cons_col = df.columns[:3]
    dates = pd.to_datetime(df[date_col], errors="coerce")
    cons = pd.to_numeric(df[cons_col], errors="coerce").fillna(0.0).astype(float)
    stock = pd.to_numeric(df[stock_col], errors="coerce").astype(float)
    ts_cons  = pd.DataFrame({"d": dates, "q": cons}).dropna(subset=["d"]).sort_values("d").set_index("d")["q"]
    ts_stock = pd.DataFrame({"d": dates, "s": stock}).dropna(subset=["d"]).sort_values("d").set_index("d")["s"]
    min_date = min(ts_cons.index.min(), ts_stock.index.min())
    max_date = max(ts_cons.index.max(), ts_stock.index.max())
    full_idx = pd.date_range(min_date, max_date, freq="D")
    cons_daily  = ts_cons.reindex(full_idx, fill_value=0.0)
    stock_daily = ts_stock.reindex(full_idx).ffill().fillna(0.0)
    return cons_daily, stock_daily

def _interval_sum_next_days_sba(daily: pd.Series, start_idx: int, interval: int) -> float:
    s = start_idx + 1
    e = s + int(max(0, interval))
    return float(pd.Series(daily).iloc[s:e].sum())

def _croston_or_sba_forecast_array_sba(x, alpha: float, variant: str = "sba"):
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
    psd = 0
    for t in range(first + 1, len(x)):
        psd += 1
        if x[t] > 0:
            I_t = psd
            z = alpha * x[t] + (1 - alpha) * z
            p = alpha * I_t + (1 - alpha) * p
            psd = 0
    f = z / p
    if variant.lower() == "sba":
        f *= (1 - alpha / 2.0)
    return {"forecast_per_period": float(f), "z_t": float(z), "p_t": float(p)}

def rolling_sba_with_rops_single_run(
    excel_path: str,
    product_code: str,
    alpha: float,
    window_ratio: float,
    interval: int,
    lead_time: int,
    lead_time_supplier: int,
    service_level: float,
    nb_sim: int,
    rng_seed: int,
    variant: str = "sba",
):
    sheet = _find_product_sheet_sba(excel_path, product_code)
    cons_daily, stock_daily = _daily_consumption_and_stock_sba(excel_path, sheet)
    vals = cons_daily.values
    split_index = int(len(vals) * window_ratio)
    if split_index < 2:
        return pd.DataFrame()

    rng = np.random.default_rng(rng_seed)
    rows = []
    rop_carry_running = 0.0
    stock_after_interval = 0.0

    for i in range(split_index, len(vals)):
        if (i - split_index) % interval == 0:
            train = vals[:i]
            test_date = cons_daily.index[i]
            fc = _croston_or_sba_forecast_array_sba(train, alpha=alpha, variant=variant)
            f = float(fc["forecast_per_period"])
            sigma_period = float(pd.Series(train).std(ddof=1)) if i > 1 else 0.0
            sigma_period = 0.0 if not np.isfinite(sigma_period) else sigma_period
            real_demand = _interval_sum_next_days_sba(cons_daily, i, interval)
            stock_on_hand_running = _interval_sum_next_days_sba(stock_daily, i, interval)
            stock_after_interval = stock_after_interval + stock_on_hand_running - real_demand
            next_real_demand = _interval_sum_next_days_sba(cons_daily, i + interval, interval) if (i + interval) < len(vals) else 0.0
            can_cover_interval = "yes" if stock_after_interval >= next_real_demand else "no"
            order_policy = "half_of_interval_demand" if can_cover_interval == "yes" else "shortfall_to_cover"

            # ROP usine
            X_Lt = lead_time * f
            sigma_Lt = sigma_period * np.sqrt(max(lead_time, 1e-9))
            var_u = sigma_Lt**2 if sigma_Lt**2 > X_Lt else X_Lt + 1e-5
            p_nb = min(max(X_Lt / var_u, 1e-12), 1 - 1e-12)
            r_nb = X_Lt**2 / (var_u - X_Lt) if var_u > X_Lt else 1e6
            ROP_u = float(np.percentile(nbinom.rvs(r_nb, p_nb, size=nb_sim, random_state=rng), 100 * service_level))

            # ROP fournisseur
            totalL = lead_time + lead_time_supplier
            X_Lt_Lw = totalL * f
            sigma_Lt_Lw = sigma_period * np.sqrt(max(totalL, 1e-9))
            var_f = sigma_Lt_Lw**2 if sigma_Lt_Lw**2 > X_Lt_Lw else X_Lt_Lw + 1e-5
            p_nb_f = min(max(X_Lt_Lw / var_f, 1e-12), 1 - 1e-12)
            r_nb_f = X_Lt_Lw**2 / (var_f - X_Lt_Lw) if var_f > X_Lt_Lw else 1e6
            ROP_f = float(np.percentile(nbinom.rvs(r_nb_f, p_nb_f, size=nb_sim, random_state=rng), 100 * service_level))

            rop_carry_running += float(ROP_u - real_demand)
            stock_status = "holding" if stock_after_interval > 0 else "rupture"

            rows.append({
                "date": test_date.date(),
                "code": product_code,
                "interval": int(interval),
                "real_demand": float(real_demand),
                "stock_on_hand_running": float(stock_on_hand_running),
                "stock_after_interval": float(stock_after_interval),
                "can_cover_interval": can_cover_interval,
                "order_policy": order_policy,
                "reorder_point_usine": float(ROP_u),
                "lead_time_usine_days": int(lead_time),
                "lead_time_supplier_days": int(lead_time_supplier),
                "reorder_point_fournisseur": float(ROP_f),
                "stock_status": stock_status,
                "rop_usine_minus_real_running": float(rop_carry_running),
            })
    return pd.DataFrame(rows)

def compute_metrics_sba(df_run: pd.DataFrame):
    if df_run.empty:
        return np.nan, np.nan, np.nan, np.nan
    est = df_run["reorder_point_usine"] / df_run["lead_time_usine_days"].replace(0, np.nan)
    e = df_run["real_demand"] - est
    ME = e.mean()
    absME = e.abs().mean()
    MSE = (e**2).mean()
    RMSE = float(np.sqrt(MSE)) if np.isfinite(MSE) else np.nan
    return ME, absME, MSE, RMSE

def _grid_and_final_sba(excel_path):
    best_rows = []
    for code in PRODUCT_CODES_UNI:
        best_row = None
        best_rmse = np.inf
        for a in ALPHAS_UNI:
            for w in WINDOW_RATIOS_UNI:
                for itv in RECALC_INTERVALS_UNI:
                    df_run = rolling_sba_with_rops_single_run(
                        excel_path=excel_path, product_code=code,
                        alpha=a, window_ratio=w, interval=itv,
                        lead_time=LEAD_TIME_UNI, lead_time_supplier=LEAD_TIME_SUPPLIER_UNI,
                        service_level=SERVICE_LEVEL_UNI, nb_sim=NB_SIM_UNI, rng_seed=RNG_SEED_UNI,
                        variant="sba",
                    )
                    _, _, _, RMSE = compute_metrics_sba(df_run)
                    if pd.notna(RMSE):
                        if (RMSE < best_rmse * 0.99) or (np.isclose(RMSE, best_rmse, rtol=0.01) and best_row is not None and (
                            (itv, a, w) > (best_row["recalc_interval"], best_row["alpha"], best_row["window_ratio"])
                        )):
                            best_rmse = RMSE
                            best_row = {"code": code, "alpha": a, "window_ratio": w, "recalc_interval": itv, "RMSE": RMSE}
        if best_row:
            best_rows.append(best_row)
    return pd.DataFrame(best_rows)

# =====================================================================
# (⚠️ Croston and SES sections omitted here for space, but include them
#   in the exact same way you had them in your script, just replacing
#   print()/display() with return values)
# =====================================================================

# =====================================================================
# ============================ STREAMLIT UI ===========================
# =====================================================================

st.title("📊 Demand Forecasting with SBA / Croston / SES")

uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])
if uploaded_file:
    excel_path = uploaded_file

    method = st.selectbox("Choose Forecasting Method", ["SBA", "Croston", "SES"])

    if st.button("Run Forecasting"):
        if method == "SBA":
            df_best = _grid_and_final_sba(excel_path)
            st.subheader("✅ SBA — Best Parameters (by RMSE)")
            st.dataframe(df_best)

        elif method == "Croston":
            df_best = _grid_and_final_croston(excel_path)
            st.subheader("✅ Croston — Best Parameters (by RMSE)")
            st.dataframe(df_best)

        elif method == "SES":
            df_best = _grid_and_final_ses(excel_path)
            st.subheader("✅ SES — Best Parameters (by RMSE)")
            st.dataframe(df_best)

        st.success("Analysis completed!")
