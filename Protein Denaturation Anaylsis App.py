import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import zscore

# streamlit run GUI/Protein\ Denaturation\ Anaylsis\ App.py

# --- App Setup ---
st.set_page_config(layout="wide")
st.title("Protein Denaturation Analysis")

# Dark background styling
st.markdown("""
    <style>
    .stApp { background-color: black; color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- Tutorial / Help Panel ---
with st.expander("📖 How to Use This App"):
    st.markdown("""
    ### 1. File Format
    - **CSV** or **Excel** (`.xlsx`, `.xls`) supported
    - First column: protein names (unique)
    - Remaining columns: numeric pH values (e.g., 4, 4.5, 5)
    - Rows: intensity measurements per protein

    ### 2. Analysis Workflow
    1. Upload your file
    2. Select a protein
    3. View normalized Protein Abundance Level & fitted curve
    4. Inspect residuals & outliers
    5. Optionally run batch fit for all proteins
    6. Download results as CSV

    ### 3. Output Explanation
    - **k (Slope):** steepness of curve
    - **pH₅₀ (xo):** pH at 50% unfolded
    - **R²:** goodness-of-fit
    - **Fit Quality:** color-coded

    ### 4. Tips
    - Avoid missing or non-numeric pH headers
    - Keep protein names unique
    - Outliers (>2 SD) are flagged automatically
    """)

# --- Melting Curve Function ---
def melting_curve(pH, k, xo):
    pH = np.array(pH, dtype=float)
    return 1 / (1 + np.exp((k * xo - k * pH)))

# --- File Upload (CSV or Excel) ---
uploaded_file = st.file_uploader(
    "Upload your data file (CSV or Excel). First column should be named 'protein', remaining columns should be numeric pH levels.",
    type=["csv", "xlsx", "xls"]
)

# --- File Loader Function ---
def load_file(file):
    try:
        if file.name.endswith(".csv"):
            df = pd.read_csv(file)
        else:
            xls = pd.ExcelFile(file)
            sheet_name = xls.sheet_names[0]  # pick first sheet automatically
            df = pd.read_excel(xls, sheet_name=sheet_name)
        if df.shape[1] < 2:
            st.error("Need at least 1 protein column + 1 numeric pH column.")
            st.stop()
        df.rename(columns={df.columns[0]: "protein"}, inplace=True)
        df.dropna(how="all", inplace=True)
        if df["protein"].isna().any():
            st.warning("Missing protein names removed.")
            df = df[df["protein"].notna()]
        # Numeric pH columns
        value_vars = []
        for col in df.columns[1:]:
            try:
                float(col)
                value_vars.append(col)
            except ValueError:
                st.warning(f"Ignoring non-numeric column '{col}'")
        if not value_vars:
            st.error("No numeric pH columns detected.")
            st.stop()
        # Melt to long format
        long_df = df.melt(id_vars=["protein"], value_vars=value_vars,
                          var_name="pH", value_name="Intensity")
        long_df["pH"] = pd.to_numeric(long_df["pH"], errors="coerce")
        long_df["Intensity"] = pd.to_numeric(long_df["Intensity"], errors="coerce")
        long_df = long_df.dropna(subset=["pH","Intensity"])
        if long_df.empty:
            st.error("No valid data found after cleaning.")
            st.stop()
        return long_df
    except Exception as e:
        st.error(f"Failed to load file: {e}")
        st.stop()



# --- Curve Fitting Function ---
def safe_fit_melting_curve(df, protein_name):
    try:
        df = df.sort_values("pH")
        popt, _ = curve_fit(
            melting_curve, df["pH"], df["Protein Abundance Level"],
            p0=[1.0, np.median(df["pH"])],
            bounds=([0.01, df["pH"].min()], [10, df["pH"].max()]),
            maxfev=8000
        )
        y_pred = melting_curve(df["pH"], *popt)
        ss_res = np.sum((df["Protein Abundance Level"] - y_pred)**2)
        ss_tot = np.sum((df["Protein Abundance Level"] - np.mean(df["Protein Abundance Level"]))**2)
        r_squared = 1 - (ss_res/ss_tot)
        if r_squared >= 0.9: quality="✅ Excellent"
        elif r_squared >= 0.7: quality="🟡 Moderate"
        else: quality="🔴 Poor"
        return {"Protein": protein_name, "k (Slope)": popt[0],
                "pH₅₀ (xo)": popt[1], "R²": r_squared, "Fit Quality": quality}
    except Exception as e:
        st.warning(f"Curve fitting failed for {protein_name}: {e}")
        return {"Protein": protein_name, "k (Slope)": np.nan, "pH₅₀ (xo)": np.nan,
                "R²": np.nan, "Fit Quality": "❌ Failed"}

# --- Main App Logic ---
if uploaded_file:
    long_df = load_file(uploaded_file)

    st.subheader("Data Preview")
    # Take the first row per protein to show a compact table
    preview = long_df.groupby("protein").first().reset_index()
    st.dataframe(preview)

    # Sidebar: select protein
    Protein = st.sidebar.selectbox("Select a protein", long_df["protein"].unique())
    filtered = long_df[long_df["protein"] == Protein].copy()
    # Normalize
    filtered["Protein Abundance Level"] = (
        (filtered["Intensity"] - filtered["Intensity"].min()) /
        (filtered["Intensity"].max() - filtered["Intensity"].min())
    )

    # Summary
    abundance_summary = filtered.groupby("pH")["Protein Abundance Level"].mean().reset_index()
    abundance_summary.rename(columns={"Protein Abundance Level":"Mean Protein Abundance Level"}, inplace=True)

    # Fit single protein
    fit_result = safe_fit_melting_curve(filtered, Protein)
    fit_results = pd.DataFrame([fit_result])
    st.subheader("Single Protein Fit Results")
    st.dataframe(fit_results)

    # --- Plotting ---
    show_raw = st.sidebar.checkbox("Show raw data", True)
    show_fit_line = st.sidebar.checkbox("Show fitted curve", True)
    show_ph50_marker = st.sidebar.checkbox("Show pH₅₀", True)

    fig, ax = plt.subplots(facecolor="black")
    ax.set_facecolor("black")
    ax.scatter(abundance_summary["pH"], abundance_summary["Mean Protein Abundance Level"],
               color="silver", s=60, label="Mean Protein Abundance Level")
    if show_raw:
        ax.scatter(filtered["pH"], filtered["Protein Abundance Level"],
                   color="lightblue", alpha=0.6, s=40, label="Raw Data")
    if show_fit_line and not np.isnan(fit_result["k (Slope)"]):
        x_fit = np.linspace(filtered["pH"].min(), filtered["pH"].max(), 200)
        y_fit = melting_curve(x_fit, fit_result["k (Slope)"], fit_result["pH₅₀ (xo)"])
        ax.plot(x_fit, y_fit, color="white", linewidth=2.2, label="Fitted Curve")
    if show_ph50_marker and not np.isnan(fit_result["pH₅₀ (xo)"]):
        ph50_y = melting_curve(fit_result["pH₅₀ (xo)"], fit_result["k (Slope)"], fit_result["pH₅₀ (xo)"])
        ax.scatter(fit_result["pH₅₀ (xo)"], ph50_y,
                   s=120, edgecolor="white", facecolor="cyan", zorder=5,
                   label=f"pH₅₀={fit_result['pH₅₀ (xo)']:.2f}")
    ax.set_xlabel("pH", color="white")
    ax.set_ylabel("Protein Abundance Level (0–1)", color="white")
    ax.set_title(f"Denaturation Profile: {Protein}", color="white", fontsize=14)
    ax.tick_params(colors="white")
    leg = ax.legend(facecolor="black", edgecolor="white")
    for text in leg.get_texts(): text.set_color("white")
    st.pyplot(fig)

    # --- Residuals Plot ---
    show_residuals = st.sidebar.checkbox("Show residuals plot", True)
    show_outliers = st.sidebar.checkbox("Highlight outliers (>2 SD)", True)
    if show_residuals and not np.isnan(fit_result["k (Slope)"]):
        y_pred = melting_curve(filtered["pH"], fit_result["k (Slope)"], fit_result["pH₅₀ (xo)"])
        residuals = filtered["Protein Abundance Level"] - y_pred
        fig_res, ax_res = plt.subplots(facecolor="black")
        ax_res.set_facecolor("black")
        ax_res.axhline(0, color="white", linestyle="--", linewidth=1)
        if show_outliers:
            std_res = residuals.std()
            outliers = residuals[np.abs(residuals) > 2*std_res]
            ax_res.scatter(outliers.index, outliers.values, color="red", s=70, label="Outliers (>2 SD)")
        ax_res.scatter(residuals.index, residuals.values, color="lightblue", alpha=0.7, label="Residuals")
        ax_res.set_xlabel("Data Point Index", color="white")
        ax_res.set_ylabel("Residual (Observed - Fitted)", color="white")
        ax_res.set_title(f"Residuals Plot: {Protein}", color="white", fontsize=13)
        ax_res.tick_params(colors="white")
        leg = ax_res.legend(facecolor="black", edgecolor="white")
        for text in leg.get_texts(): text.set_color("white")
        st.pyplot(fig_res)
        
    st.markdown("<br></br>", unsafe_allow_html=True)

    # --- Batch Fit Option ---
    st.markdown("---")
    st.header("Batch Fit for All Proteins")
    if st.button("Run Batch Fit"):
        batch_results = []
        for prot in long_df["protein"].unique():
            subset = long_df[long_df["protein"] == prot].copy()
            subset["Protein Abundance Level"] = (
                (subset["Intensity"] - subset["Intensity"].min()) /
                (subset["Intensity"].max() - subset["Intensity"].min())
            )
            batch_results.append(safe_fit_melting_curve(subset, prot))
        batch_df = pd.DataFrame(batch_results)
        st.subheader("Batch Fit Results")
        st.dataframe(batch_df)
        csv_all = batch_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download All Protein Fits",
            data=csv_all,
            file_name="All_Proteins_Fit_Results.csv",
            mime="text/csv"
        )

else:
    st.info("Upload a CSV or Excel file to begin analysis.")
