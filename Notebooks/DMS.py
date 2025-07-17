# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "marimo",
#     "matplotlib==3.9.4",
#     "numpy==2.0.2",
#     "pandas==2.3.1",
#     "pyarrow==20.0.0",
#     "scikit-learn==1.6.1",
#     "scipy==1.13.1",
#     "seaborn==0.13.2",
# ]
# ///

import marimo

__generated_with = "0.14.11"
app = marimo.App(width="medium", css_file="", html_head_file="")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import pandas as pd
    import textwrap
    from pathlib import Path
    from scipy.stats import spearmanr
    from sklearn.linear_model import LinearRegression
    import matplotlib as mpl
    import seaborn as sns
    import numpy as np
    import matplotlib.pyplot as plt
    return (
        LinearRegression,
        Path,
        mo,
        mpl,
        np,
        pd,
        plt,
        sns,
        spearmanr,
        textwrap,
    )


@app.cell(hide_code=True)
def _(Path, pd, textwrap):
    csv_path = Path("Dataframes/summary_avgprefs.csv")
    df = pd.read_csv(csv_path)

    aa_cols = [
        "A", "C", "D", "E", "F", "G", "H", "I", "K", "L",
        "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"
    ]

    sequence = "".join(df[aa_cols].idxmax(axis=1).tolist())

    fasta_lines = [">Estimated_sequence_from_DMS"] + textwrap.wrap(sequence, 60)
    fasta_path = Path("Dataframes/estimated_sequence.fasta")
    fasta_path.write_text("\n".join(fasta_lines))
    return


@app.cell
def _(mo):
    mo.md(r"""# Test output""")
    return


@app.cell(hide_code=True)
def _(pd):
    df_650_FT_DF = pd.read_csv(
        "Dataframes/df_650_FT_DF_with_loess.csv", keep_default_na=False
    )

    df_650_FT_DF_HA = df_650_FT_DF[df_650_FT_DF["Segment"] == "HA"]

    #df_650_FT_DF_HA['log_likelihood_LOESS'] = df_650_FT_DF_HA['log_likelihood_LOESS'].astype(int)

    df_3B_FT_DF = pd.read_csv(
        "Dataframes/df_3B_FT_DF_with_loess.csv", keep_default_na=False
    )

    df_3B_FT_DF_HA = df_3B_FT_DF[df_3B_FT_DF["Segment"] == "HA"]
    return df_3B_FT_DF_HA, df_650_FT_DF_HA


@app.cell(hide_code=True)
def _(Path, df_3B_FT_DF_HA, df_650_FT_DF_HA, pd):
    dms_path = Path("Dataframes/summary_avgprefs.csv")          # adjust if needed
    dms = pd.read_csv(dms_path)

    AA_COLS = ["A","C","D","E","F","G","H","I","K","L",
               "M","N","P","Q","R","S","T","V","W","Y"]
    missing = set(AA_COLS) - set(dms.columns)
    if missing:
        raise ValueError(f"DMS table is missing columns: {sorted(missing)}")

    def compute_dms_score(seq: str, prefs: pd.DataFrame) -> float:
        total = 0.0
        max_site = min(len(seq), len(prefs))
        for i in range(max_site):
            aa = seq[i]
            # Skip non-standard or gap characters
            if aa not in prefs.columns:
                continue
            total += prefs.at[i, aa]   # row i (0-based), column letter
        return total

    df_650_FT_DF_HA["dms_score"] = df_650_FT_DF_HA["sequence"].apply(lambda s: compute_dms_score(s, dms))

    df_3B_FT_DF_HA["dms_score"] = df_3B_FT_DF_HA["sequence"].apply(lambda s: compute_dms_score(s, dms))

    df_650_FT_DF_HA
    return


@app.cell(hide_code=True)
def _(mpl, plt, sns):
    mpl.rcParams.update({
        "figure.facecolor":   "white",   # top-level figure bg
        "axes.facecolor":     "white",   # axes / subplot bg
        "savefig.facecolor":  "white",   # files you write with plt.savefig
        "axes.edgecolor":     "black",
        "axes.labelcolor":    "black",
        "xtick.color":        "black",
        "ytick.color":        "black",
        "text.color":         "black",
    })

    # If you use Seaborn, switch to a white style too
    sns.set(style="white", palette="Set2")

    # (Optional) keep the light grid you already had
    plt.style.use("seaborn-v0_8-whitegrid")
    return


@app.cell(hide_code=True)
def _(LinearRegression, np, pd, plt, sns, spearmanr):
    def time_maxfreq_spear(model_df, LOESS=False):

        if(LOESS == False):
                ll_col = "log_likelihood" 
        else: 
                ll_col = "corrected_log_likelihood"

        for segment, sub in model_df.groupby("Segment"):
            df = sub.copy()

            # ensure numeric
            df[ll_col] = pd.to_numeric(df[ll_col], errors="coerce")
            df = df.dropna(subset=[ll_col])

            rho, pval = spearmanr(df["dms_score"], df[ll_col])
            print(f"{segment}: Spearman ρ = {rho:.3f}, p = {pval:.3g}")

            df["time_rank"] = df["dms_score"].rank()
            df["freq_rank"] = df[ll_col].rank()

            X = df[["time_rank"]].values
            y = df["freq_rank"].values
            lm = LinearRegression().fit(X, y)

            sns.scatterplot(x="dms_score", y=ll_col, data=df)
            x_line = np.linspace(df["dms_score"].min(), df["dms_score"].max(), 100)
            x_line_rank = pd.Series(x_line).rank(method="first").values
            y_line_rank = lm.predict(x_line_rank.reshape(-1, 1))
            # convert predicted ranks to percentiles safely
            percentiles = 100 * (y_line_rank - 1) / (len(df) - 1)
            y_line = np.percentile(df[ll_col], percentiles.clip(0, 100))
            plt.plot(x_line, y_line, "--", color="red", label=f"Spearman fit (ρ={rho:.2f})")
            plt.legend()
            plt.title(f"{segment}: dms_score vs maximum frequency")
            plt.show()
    return (time_maxfreq_spear,)


@app.cell(hide_code=True)
def _(df_650_FT_DF_HA):
    df_650_FT_DF_HA_FT = df_650_FT_DF_HA[df_650_FT_DF_HA["Model"] == "Fine_Tune_650M"]
    df_650_FT_DF_HA_Base = df_650_FT_DF_HA[df_650_FT_DF_HA["Model"] == "650M"]
    return df_650_FT_DF_HA_Base, df_650_FT_DF_HA_FT


@app.cell(hide_code=True)
def _(df_650_FT_DF_HA_Base, time_maxfreq_spear):
    time_maxfreq_spear(df_650_FT_DF_HA_Base)
    return


@app.cell(hide_code=True)
def _(df_650_FT_DF_HA_FT):
    df_650_FT_DF_HA_FT_More_1990 = df_650_FT_DF_HA_FT[df_650_FT_DF_HA_FT["time"] > 1990]
    df_650_FT_DF_HA_FT_LESS_1990 = df_650_FT_DF_HA_FT[df_650_FT_DF_HA_FT["time"] < 1990]
    return df_650_FT_DF_HA_FT_LESS_1990, df_650_FT_DF_HA_FT_More_1990


@app.cell(hide_code=True)
def _(df_650_FT_DF_HA_FT_More_1990, time_maxfreq_spear):
    time_maxfreq_spear(df_650_FT_DF_HA_FT_More_1990)
    return


@app.cell(hide_code=True)
def _(df_650_FT_DF_HA_FT_LESS_1990, time_maxfreq_spear):
    time_maxfreq_spear(df_650_FT_DF_HA_FT_LESS_1990)
    return


@app.cell(hide_code=True)
def _(df_650_FT_DF_HA_FT_More_1990, time_maxfreq_spear):
    time_maxfreq_spear(df_650_FT_DF_HA_FT_More_1990, LOESS=True)
    return


@app.cell(hide_code=True)
def _(df_650_FT_DF_HA_FT_LESS_1990, time_maxfreq_spear):
    time_maxfreq_spear(df_650_FT_DF_HA_FT_LESS_1990, LOESS=True)
    return


@app.cell(hide_code=True)
def _():
    return


if __name__ == "__main__":
    app.run()
