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
# ================== INSERT YOUR FUNCTIONS HERE =======================
# (all the SBA / Croston / SES functions you shared remain the same)
# =====================================================================
# Just remove all 'print()' and 'display()', replace them with return values
# =====================================================================

# Example modification:
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

    df_best_sba = pd.DataFrame(best_rows)
    return df_best_sba


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

