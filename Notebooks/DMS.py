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
def _(mo):
    mo.md(r"""## Calculate time series for DMS data""")
    return


@app.cell
def _(df_650_FT_DF_HA_FT):
    df_650_FT_DF_HA_FT
    return


@app.cell
def _(pd, spearmanr):
    # calculate spearman cc for each time frame
    def spearman_correlation_calculation(df, x_col, y_col, model_name, segment, time_label):
        spearman_corr, p_value = spearmanr(df[x_col], df[y_col])
        return {
            "Model": model_name,
            "Segment": segment,
            "Time_Range": time_label,
            "Spearman_Correlation": spearman_corr,
            "P_Value": p_value
        }

    def spearman_correlation(df, model):
        results = []

        for segment, group in df.groupby('Segment'):
            df_segment = df[df['Segment'] == segment]

            if segment == "PA":
                df_segment = df_segment[df_segment['node'] != 'A/Viamao/LACENRS-974/2015']

            df_segment_FT = df_segment[df_segment["Model"] == f"Fine_Tune_{model}"]
            #df_segment_FT_LS = df_segment[df_segment["Model"] == f"LOESS_Fine_Tune_{model}"]
            #df_segment_BS = df_segment[df_segment["Model"] == f"{model}"]

            time_ranges = [
                (1965, 1975, "1970"),
                (1970, 1980, "1975"),
                (1975, 1985, "1980"),
                (1980, 1990, "1985"),
                (1985, 1995, "1990"),
                (1990, 2000, "1995"),
                (1995, 2005, "2000"),
                (2000, 2010, "2005"),
                (2010, 2015, "2010"),
                (2010, 2020, "2015"),
                (2015, 2020, "2020"),
                (2020, None, "2025")  
            ]

            for start, end, label in time_ranges:
                if end is None:
                    df_segment_FT_label = df_segment_FT[df_segment_FT['time'] >= start]

                else:
                    df_segment_FT_label = df_segment_FT[(df_segment_FT['time'] >= start) & (df_segment_FT['time'] <= end)]

                results.append(spearman_correlation_calculation(df_segment_FT_label, "max_frequency", "dms_score", f"LOESS_Fine_Tune_{model}", segment, label))

        return pd.DataFrame(results)
    return (spearman_correlation,)


@app.cell
def _(df_3B_FT_DF_HA, df_650_FT_DF_HA_FT, spearman_correlation):
    df_650_FT_DF_HA_FT_time_series = spearman_correlation(df_650_FT_DF_HA_FT, "650M")
    df_3B_FT_DF_HA_FT_time_series = spearman_correlation(df_3B_FT_DF_HA, "3B")
    return df_3B_FT_DF_HA_FT_time_series, df_650_FT_DF_HA_FT_time_series


@app.cell
def _(np, plt, sns):
    def create_spearman_summary_plot(df, model):
        sns.set_style("whitegrid")
        custom_params = {"axes.spines.right": False, "axes.spines.top": False}
        sns.set_theme(style="ticks", rc=custom_params)

        fig, ax = plt.subplots(figsize=(10, 5))

        ax = sns.lineplot(
            data=df,
            x='Time_Range',
            y='Spearman_Correlation',
            hue='Model',
            marker="o",
            legend=False,
            zorder=1,
            ax=ax,
            errorbar=None,   
        )

        ax.set_title(f"Spearman CC DMS vs Max Frequency - Fine Tune {model} Model")
        ax.set_xlabel("Time Range")
        ax.set_ylabel("Spearman CC")

        label_positions = []
        for line, model in zip(ax.lines, df['Model'].unique()):
            y = line.get_ydata()[-1]
            x = line.get_xdata()[-1]

            if not np.isfinite(y) or not np.isfinite(x):
                continue

            if model in ("Base - 3B Model", "Base - 650M Model"):
                y = line.get_ydata()[-4]
                x = line.get_xdata()[-4]
            elif model in (
                "Fine Tune - 650M Model - LR 2.5e-05",
                "Fine Tune - 650M Model - LR 1e-05",
                "Fine Tune - 650M Model - LR 1e-06"
            ):
                y = line.get_ydata()[-3]
                x = line.get_xdata()[-3]
            else:
                while any(abs(y - pos) < 0.025 for pos in label_positions):
                    y += 0.007

            """
            label_positions.append(y)

            ax.annotate(
                model,
                xy=(x, y),
                xytext=(5, 0),
                textcoords="offset points",
                color=line.get_color(),
                fontsize=12,
                weight='bold',
                ha='left',
                va='center',
                zorder=2,
            )
            """

        plt.tight_layout()
        plt.show()
    return (create_spearman_summary_plot,)


@app.cell
def _(create_spearman_summary_plot, df_650_FT_DF_HA_FT_time_series):
    create_spearman_summary_plot(df_650_FT_DF_HA_FT_time_series, "650M")
    return


@app.cell
def _(create_spearman_summary_plot, df_3B_FT_DF_HA_FT_time_series):
    create_spearman_summary_plot(df_3B_FT_DF_HA_FT_time_series, "3B")
    return


if __name__ == "__main__":
    app.run()
