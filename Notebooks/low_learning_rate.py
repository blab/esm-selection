# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "altair==5.5.0",
#     "duckdb==1.3.2",
#     "marimo",
#     "matplotlib==3.9.4",
#     "nbformat==5.10.4",
#     "numpy==2.0.2",
#     "openai==1.98.0",
#     "pandas==2.3.1",
#     "polars[pyarrow]==1.32.0",
#     "pytest==8.4.1",
#     "scipy==1.13.1",
#     "seaborn==0.13.2",
#     "sqlglot==27.6.0",
#     "ty==0.0.1a16",
#     "vegafusion==2.0.2",
#     "vl-convert-python==1.8.0",
# ]
# ///

import marimo

__generated_with = "0.15.1"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""# Setup""")
    return


@app.cell
def _(mo):
    mo.md(r"""## Install libraries""")
    return


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import os
    from pathlib import Path
    import glob
    import matplotlib.pyplot as plt
    import json
    import matplotlib.cm as cm
    import matplotlib as mpl
    import seaborn as sns
    import colorsys
    return Path, cm, colorsys, glob, json, mo, mpl, os, pd, plt, sns


@app.cell
def _(mo):
    mo.md(r"""## Import dataframes""")
    return


@app.cell
def _(os):
    current_directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_directory)
    os.chdir("..")
    return


@app.cell
def _(os):
    cwd = os.getcwd()
    print(cwd)
    return


@app.cell
def _(Path, glob, pd):
    def load_fine_tune_results(
        *subfolders,
        base_dir: str = "Flu_Snakemake_Pipeline/results/max_freqs_log_likelyhood_Fine_Tune",
    ) -> pd.DataFrame:

        full_path = Path(base_dir, *subfolders)
        glob_pattern = str(full_path / "*.csv")
        csv_files = glob.glob(glob_pattern)

        frames = []
        for fname in csv_files:
            df = pd.read_csv(fname)

            # Segment from filename
            seg = Path(fname).stem.rsplit("_", 1)[-1].upper()
            df["Segment"] = seg

            # Extract metadata from folder names
            metadata = {key: value for f in subfolders if "~" in f for key, value in [f.split("~")]}

            model = metadata.get("model")
            lr = metadata.get("learning_rate", "")
            epochs = metadata.get("epochs", "")
            df["Model"] = f"Fine_Tune_{model}" if model else None
            df["Learning_rate"] = lr
            df["Epochs"] = epochs
            df["Model_training_time"] = metadata.get("time") if "time" in metadata else None
            df["tree"] = metadata.get("next_tree")

            frames.append(df)

        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return (load_fine_tune_results,)


@app.cell
def _(load_fine_tune_results):
    df_3B_FT_DF_1990_LR_5e_10 = load_fine_tune_results(
        "next_tree~h3n2",
        "epochs~1",
        "learning_rate~5e-10",
        "model~esm2_t36_3B_UR50D",
        "time~1990",
    )

    df_3B_FT_DF_1990_LR_5e_07 = load_fine_tune_results(
        "next_tree~h3n2",
        "epochs~1",
        "learning_rate~5e-07",
        "model~esm2_t36_3B_UR50D",
        "time~1990",
    )

    df_3B_FT_DF_1990_LR_5e_05 = load_fine_tune_results(
        "next_tree~h3n2",
        "epochs~1",
        "learning_rate~5e-05",
        "model~esm2_t36_3B_UR50D",
        "time~1990",
    )
    return df_3B_FT_DF_1990_LR_5e_07, df_3B_FT_DF_1990_LR_5e_10


@app.cell
def _(json, os, pd):
    #Add time to Model dfs

    def extract_node_times(tree_data, segment):
        node_list = []

        def recurse_nodes(node):
            name = node.get('name')
            num_date = node.get('node_attrs', {}).get('num_date', {}).get('value')
            if name and num_date is not None:
                node_list.append({'Segment': segment, 'node': name, 'time': num_date})
            for child in node.get('children', []):
                recurse_nodes(child)

        root = tree_data.get('tree', tree_data)
        recurse_nodes(root)
        return node_list


    def process_directory(directory):

        all_data = []

        for filename in os.listdir(directory):

            segment = filename[:-5]  # remove the '.json' suffix
            if (filename == f"{filename[:-5]}.json"):
                file_path = os.path.join(directory, filename)
                with open(file_path, 'r') as f:
                    tree_data = json.load(f)

            segment_data = extract_node_times(tree_data, segment)
            all_data.extend(segment_data)
        return pd.DataFrame(all_data)

    def merge_time(models_df, tree):
        directory = f"Flu_Snakemake_Pipeline/input/trees/{tree}/"
        df = process_directory(directory)
        df['Segment'] = df['Segment'].str.upper()
        models_df = models_df.merge(df, on=['Segment', 'node'], how='left')
        #models_df = models_df[models_df['time'] >= 1991]
        return models_df
    return (merge_time,)


@app.cell
def _(df_3B_FT_DF_1990_LR_5e_07, df_3B_FT_DF_1990_LR_5e_10, merge_time):
    df_3B_FT_DF_1990_LR_5e_10_Time = merge_time(df_3B_FT_DF_1990_LR_5e_10, "h3n2")
    df_3B_FT_DF_1990_LR_5e_07_Time = merge_time(df_3B_FT_DF_1990_LR_5e_07, "h3n2")
    #df_3B_FT_DF_1990_LR_5e_05_Time = merge_time(df_3B_FT_DF_1990_LR_5e_05, "h3n2")
    return df_3B_FT_DF_1990_LR_5e_07_Time, df_3B_FT_DF_1990_LR_5e_10_Time


@app.cell
def _(mpl, plt, sns):
    mpl.rcParams.update({
        "figure.facecolor":   "white",  
        "axes.facecolor":     "white",   
        "savefig.facecolor":  "white",   
        "axes.edgecolor":     "black",
        "axes.labelcolor":    "black",
        "xtick.color":        "black",
        "ytick.color":        "black",
        "text.color":         "black",
    })

    sns.set(style="white", palette="Set2")
    sns.set(style='ticks', palette='Set2')
    plt.style.use("seaborn-v0_8-whitegrid")
    return


@app.cell
def _(cm, colorsys, os, plt):
    def esm_vs_time_scatterplot_seperate(model_df, model_name):
        for segment, group in model_df.groupby('Segment'):

                def darken_color(rgb, factor=0.7):
                    h, l, s = colorsys.rgb_to_hls(*rgb)
                    r, g, b = colorsys.hls_to_rgb(h, max(0, l * factor), s)
                    return (r, g, b, 1.0)

                def plot_esm_score(ax, df, title, Fine_Tune=False):
                    norm = plt.Normalize(df["log_likelihood"].min(), df["log_likelihood"].max())
                    cmap = plt.get_cmap("viridis")
                    colors = cmap(norm(df["log_likelihood"]))
                    edgecolors = [darken_color(c[:3], factor=0.7) for c in colors]
                    sc = ax.scatter(
                        df["time"],
                        df["log_likelihood"],
                        c=colors,
                        edgecolors=edgecolors,
                        linewidths=0.5,
                        alpha=0.7,
                        zorder=1
                    )

                    high_freq_df = df[df["max_frequency"] >= 1].sort_values("time")
                    ax.plot(
                        high_freq_df["time"],
                        high_freq_df["log_likelihood"],
                        linestyle='-',
                        color='black',
                        linewidth=3,
                        alpha=0.6,
                        label='Max Freq ≥ 0.99',
                        zorder=2
                    )

                    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
                    sm.set_array([])
                    cbar = plt.colorbar(sm, ax=ax, orientation='vertical')
                    ax.set_title(title)
                    if Fine_Tune:
                        ax.axvline(1990, color='gray', linestyle='--', linewidth=1.5)
                    ax.set_ylabel("ESM Score")
                    ax.grid(True, color='lightgray', linestyle='-', linewidth=0.75)
                    ax.spines[['right', 'top']].set_visible(False)
                    return ax

                #df1 = model_df[model_df['Model'] == f"Fine_Tune_{model_name}"]
                df1 = model_df
                df1 = df1[df1['Segment'] == segment]
                if segment == "PA":
                    df1 = df1[df1['node'] != 'A/Viamao/LACENRS-974/2015']

                df2 = model_df[model_df['Model'] == model_name]
                df2 = df2[df2['Segment'] == segment]

                if segment == "PA":
                    df2 = df2[df2['node'] != 'A/Viamao/LACENRS-974/2015']

                fig, (ax1) = plt.subplots(1, 1, figsize=(10, 7), sharex=True)
                plot_esm_score(ax1, df1, f"{segment.upper()} - 3B Fine Tune", Fine_Tune=True)
                #plot_esm_score(ax2, df2, f"{segment.upper()} - 3B Base Model")
                ax1.set_xlabel("Date")

                os.makedirs(f"Flu_Figures/Combined_{model_name}_v_{model_name}_TN_Scatterplots/", exist_ok=True)
                plt.savefig(f"Flu_Figures/Combined_{model_name}_v_{model_name}_TN_Scatterplots/{segment}_{model_name}_v_{model_name}_TN_Scatterplots.png", dpi=300)

                plt.tight_layout()
                plt.show()
    return (esm_vs_time_scatterplot_seperate,)


@app.cell
def _(df_3B_FT_DF_1990_LR_5e_05_Time, esm_vs_time_scatterplot_seperate):
    esm_vs_time_scatterplot_seperate(df_3B_FT_DF_1990_LR_5e_05_Time, "_esm2_t36_3B_UR50D")
    return


@app.cell
def _(df_3B_FT_DF_1990_LR_5e_07_Time, esm_vs_time_scatterplot_seperate):
    esm_vs_time_scatterplot_seperate(df_3B_FT_DF_1990_LR_5e_07_Time, "_esm2_t36_3B_UR50D")
    return


@app.cell
def _(df_3B_FT_DF_1990_LR_5e_10_Time, esm_vs_time_scatterplot_seperate):
    esm_vs_time_scatterplot_seperate(df_3B_FT_DF_1990_LR_5e_10_Time, "_esm2_t36_3B_UR50D")
    return


@app.cell
def _(load_fine_tune_results):
    df_15B_FT_DF_1990_LR_5e_05 = load_fine_tune_results(
        "next_tree~h3n2",
        "epochs~1",
        "learning_rate~5e-05",
        "model~esm2_t48_15B_UR50D",
        "time~1990",
    )

    df_15B_FT_DF_1990_LR_5e_07 = load_fine_tune_results(
        "next_tree~h3n2",
        "epochs~1",
        "learning_rate~5e-07",
        "model~esm2_t48_15B_UR50D",
        "time~1990",
    )
    return df_15B_FT_DF_1990_LR_5e_05, df_15B_FT_DF_1990_LR_5e_07


@app.cell
def _(df_15B_FT_DF_1990_LR_5e_05, df_15B_FT_DF_1990_LR_5e_07, merge_time):
    df_15B_FT_DF_1990_LR_5e_05_Time = merge_time(df_15B_FT_DF_1990_LR_5e_05, "h3n2")
    df_15B_FT_DF_1990_LR_5e_07_Time = merge_time(df_15B_FT_DF_1990_LR_5e_07, "h3n2")
    return df_15B_FT_DF_1990_LR_5e_05_Time, df_15B_FT_DF_1990_LR_5e_07_Time


@app.cell
def _(df_15B_FT_DF_1990_LR_5e_05_Time, esm_vs_time_scatterplot_seperate):
    esm_vs_time_scatterplot_seperate(df_15B_FT_DF_1990_LR_5e_05_Time, "_esm2_t36_15B_UR50D")
    return


@app.cell
def _(df_15B_FT_DF_1990_LR_5e_07_Time, esm_vs_time_scatterplot_seperate):
    esm_vs_time_scatterplot_seperate(df_15B_FT_DF_1990_LR_5e_07_Time, "_esm2_t36_15B_UR50D")
    return


@app.cell
def _():
    return


@app.cell
def _(Path, pd):
    def load_fine_tune_results_2(
        *subfolders,
        base_dir: str = "Flu_Snakemake_Pipeline/results/max_freqs_log_likelyhood_Fine_Tune",
    ) -> pd.DataFrame:
        """
        Load fine-tuning CSVs under `base_dir`, with optional folder tokens like:
          "next_tree~h3n2", "epochs~1", "learning_rate~5e-05", "model~model~esm2_t33_650M_UR50D",
          "time~1990", "lora~lora" or "lora~nolora".

        Behavior:
          - If a 'lora~...' token is provided, search only that branch.
          - If no 'lora~...' token is provided, search both 'lora~lora' and 'lora~nolora'.
          - Searches recursively for *.csv files.
          - Adds 'LoRA' column from folder metadata ('lora' or 'nolora').
          - Robust metadata parsing: splits on the FIRST '~' so values may contain '~'.

        Returns:
          Concatenated DataFrame of all found CSVs (or empty DataFrame if none).
        """
        base_path = Path(base_dir)

        # Determine whether the caller explicitly specified a lora branch
        has_lora = any(str(f).startswith("lora~") for f in subfolders)

        # Build the set of root paths to search
        if has_lora:
            search_roots = [base_path.joinpath(*subfolders)]
        else:
            # Search both branches if lora not specified
            search_roots = [
                base_path.joinpath(*subfolders, "lora~lora"),
                base_path.joinpath(*subfolders, "lora~nolora"),
            ]

        frames = []

        for root in search_roots:
            if not root.exists():
                continue

            # Recursively find CSVs
            for csv_path in root.rglob("*.csv"):
                try:
                    df = pd.read_csv(csv_path)
                except Exception:
                    # Skip unreadable/bad files rather than failing the whole load
                    continue

                # Segment from filename (last underscore-separated token)
                seg = csv_path.stem.rsplit("_", 1)[-1].upper()
                df["Segment"] = seg

                # Extract metadata from ALL ancestor directories relative to base_dir
                metadata = {}
                try:
                    rel_parts = csv_path.parent.relative_to(base_path).parts
                except ValueError:
                    # Fallback if file isn't under base_path for any reason
                    rel_parts = csv_path.parent.parts

                for part in rel_parts:
                    if "~" in part:
                        key, value = part.split("~", 1)  # split only once
                        metadata[key] = value

                # Clean up model value if it accidentally contains a nested 'model~' prefix
                model_val = metadata.get("model")
                if model_val and model_val.startswith("model~"):
                    model_val = model_val.split("model~", 1)[1]

                df["Model"] = f"Fine_Tune_{model_val}" if model_val else None
                df["Learning_rate"] = metadata.get("learning_rate", "")
                df["Epochs"] = metadata.get("epochs", "")
                df["Model_training_time"] = metadata.get("time")
                df["tree"] = metadata.get("next_tree") or metadata.get("tree")
                df["LoRA"] = metadata.get("lora")  # "lora" or "nolora"

                frames.append(df)

        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    return (load_fine_tune_results_2,)


@app.cell
def _(load_fine_tune_results_2):
    df_no_lora_5_05 = load_fine_tune_results_2(
        "next_tree~h3n2",
        "epochs~1",
        "learning_rate~5e-05",
        "model~esm2_t33_650M_UR50D",
        "time~1990",
        "lora~nolora",
    )

    df_no_lora_5_07 = load_fine_tune_results_2(
        "next_tree~h3n2",
        "epochs~1",
        "learning_rate~5e-07",
        "model~esm2_t33_650M_UR50D",
        "time~1990",
        "lora~nolora",
    )

    df_lora_5_05 = load_fine_tune_results_2(
        "next_tree~h3n2",
        "epochs~1",
        "learning_rate~5e-05",
        "model~esm2_t33_650M_UR50D",
        "time~1990",
        "lora~lora",
    )

    df_lora_5_07 = load_fine_tune_results_2(
        "next_tree~h3n2",
        "epochs~1",
        "learning_rate~5e-07",
        "model~esm2_t33_650M_UR50D",
        "time~1990",
        "lora~lora",
    )

    df_lora_5_0005 = load_fine_tune_results_2(
        "next_tree~h3n2",
        "epochs~1",
        "learning_rate~0.0005",
        "model~esm2_t33_650M_UR50D",
        "time~1990",
        "lora~lora",
    )
    return (
        df_lora_5_0005,
        df_lora_5_05,
        df_lora_5_07,
        df_no_lora_5_05,
        df_no_lora_5_07,
    )


@app.cell
def _(load_fine_tune_results_2):
    df_no_lora_5_05_3B = load_fine_tune_results_2(
        "next_tree~h3n2",
        "epochs~1",
        "learning_rate~5e-05",
        "model~esm2_t36_3B_UR50D",
        "time~1990",
        "lora~nolora",
    )

    df_lora_5_05_3B = load_fine_tune_results_2(
        "next_tree~h3n2",
        "epochs~1",
        "learning_rate~5e-05",
        "model~esm2_t36_3B_UR50D",
        "time~1990",
        "lora~lora",
    )

    df_nolora_5_07_3B = load_fine_tune_results_2(
        "next_tree~h3n2",
        "epochs~1",
        "learning_rate~5e-07",
        "model~esm2_t36_3B_UR50D",
        "time~1990",
        "lora~nolora",
    )
    return df_lora_5_05_3B, df_no_lora_5_05_3B, df_nolora_5_07_3B


@app.cell
def _(
    df_lora_5_0005,
    df_lora_5_05,
    df_lora_5_05_3B,
    df_lora_5_07,
    df_no_lora_5_05,
    df_no_lora_5_05_3B,
    df_no_lora_5_07,
    df_nolora_5_07_3B,
    merge_time,
):

    df_lora_5_05_Time = merge_time(df_lora_5_05, "h3n2")
    df_lora_5_07_Time = merge_time(df_lora_5_07, "h3n2")
    df_lora_5_0005_Time = merge_time(df_lora_5_0005, "h3n2")

    df_nolora_5_07_3B_Time = merge_time(df_nolora_5_07_3B, "h3n2")

    df_no_lora_5_05_Time = merge_time(df_no_lora_5_05, "h3n2")
    df_no_lora_5_07_Time = merge_time(df_no_lora_5_07, "h3n2")

    df_no_lora_5_05_3B_Time = merge_time(df_no_lora_5_05_3B, "h3n2")
    df_lora_5_05_3B_Time = merge_time(df_lora_5_05_3B, "h3n2")
    return df_lora_5_0005_Time, df_no_lora_5_05_Time, df_nolora_5_07_3B_Time


@app.cell
def _(df_no_lora_5_05_Time, esm_vs_time_scatterplot_seperate):
    esm_vs_time_scatterplot_seperate(df_no_lora_5_05_Time, "_esm2_t33_650M_UR50D")
    return


@app.cell
def _(df_nolora_5_07_Time, esm_vs_time_scatterplot_seperate):
    esm_vs_time_scatterplot_seperate(df_nolora_5_07_Time, "_esm2_t33_650M_UR50D")
    return


@app.cell
def _(df_nolora_5_07_3B_Time, esm_vs_time_scatterplot_seperate):
    esm_vs_time_scatterplot_seperate(df_nolora_5_07_3B_Time, "_esm2_t33_650M_UR50D")
    return


@app.cell
def _(df_no_lora_5_05_Time, esm_vs_time_scatterplot_seperate):
    esm_vs_time_scatterplot_seperate(df_no_lora_5_05_Time, "_esm2_t33_650M_UR50D")
    return


@app.cell
def _(df_lora_5_0005_Time, esm_vs_time_scatterplot_seperate):
    esm_vs_time_scatterplot_seperate(df_lora_5_0005_Time, "_esm2_t33_650M_UR50D")
    return


@app.cell
def _(df_lora_5_0005_Time):
    df_lora_5_0005_Time
    return


@app.cell
def _(df_lora_Time, esm_vs_time_scatterplot_seperate):
    esm_vs_time_scatterplot_seperate(df_lora_Time, "_esm2_t33_650M_UR50D")
    return


@app.cell
def _(df_lora_0005_Time, esm_vs_time_scatterplot_seperate):
    esm_vs_time_scatterplot_seperate(df_lora_0005_Time, "_esm2_t33_650M_UR50D")
    return


@app.cell
def _(df_lora_005_Time, esm_vs_time_scatterplot_seperate):
    esm_vs_time_scatterplot_seperate(df_lora_005_Time, "_esm2_t33_650M_UR50D")
    return


@app.cell
def _(df_lora_0005_Time):
    df_lora_0005_Time
    return


@app.cell
def _(pd):
    import re
    import numpy as np
    from scipy.stats import spearmanr

    def spearman_ll_by_segment(
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        segment_cols,
        ll_cols=None,
        check_segments_equal: bool = True,
    ) -> pd.DataFrame:
        """
        For each segment (group defined by `segment_cols`), compute Spearman correlations
        between the matching log_likelihood columns in df1 and df2.
    
        Assumes row positions already align between df1 and df2.

        Returns a tidy DataFrame with columns:
          [*segment_cols, "column", "rho", "pvalue", "n"]
        """
        # normalize segment_cols to a list
        if isinstance(segment_cols, (str, int)):
            segment_cols = [segment_cols]

        # optional safety: make sure segment columns match across df1 and df2
        if check_segments_equal:
            if not df1[segment_cols].reset_index(drop=True).equals(
                df2[segment_cols].reset_index(drop=True)
            ):
                raise ValueError("segment columns differ between df1 and df2, but positions were expected to match.")

        # infer log_likelihood columns if not provided (case-insensitive)
        if ll_cols is None:
            pat = re.compile(r"log[\s_-]*likelihood", re.IGNORECASE)
            cand1 = [c for c in df1.columns if isinstance(c, str) and pat.search(c)]
            cand2 = [c for c in df2.columns if isinstance(c, str) and pat.search(c)]
            # keep df1 order, intersect with df2
            ll_cols = [c for c in cand1 if c in set(cand2)]
            if not ll_cols:
                raise ValueError("Could not infer log_likelihood columns; pass ll_cols explicitly.")

        out_rows = []
        # group by segment(s); keep NaN groups if present
        try:
            gb = df1.groupby(segment_cols, dropna=False, sort=False)
        except TypeError:  # older pandas without dropna=
            gb = df1.groupby(segment_cols, sort=False)

        for seg_key, g in gb:
            idx = g.index
            for col in ll_cols:
                x = df1.loc[idx, col].to_numpy()
                y = df2.loc[idx, col].to_numpy()
                m = np.isfinite(x) & np.isfinite(y)
                n = int(m.sum())
                if n >= 2:
                    rho, p = spearmanr(x[m], y[m])
                else:
                    rho, p = np.nan, np.nan

                # normalize seg_key into dict
                if len(segment_cols) == 1:
                    seg_vals = {segment_cols[0]: seg_key}
                else:
                    seg_vals = {c: v for c, v in zip(segment_cols, seg_key)}

                out_rows.append({**seg_vals, "column": col, "rho": rho, "pvalue": p, "n": n})

        res = pd.DataFrame(out_rows)
        # nice ordering
        return res.sort_values(segment_cols + ["column"]).reset_index(drop=True)

    return np, re, spearman_ll_by_segment


@app.cell
def _(df_lora_0005_Time, df_no_lora_Time, spearman_ll_by_segment):
    res = spearman_ll_by_segment(df_no_lora_Time, df_lora_0005_Time, segment_cols=["segment"])
    print(res)
    return


@app.cell
def _(df_lora_005_Time, df_no_lora_Time, spearman_ll_by_segment):
    print(spearman_ll_by_segment(df_no_lora_Time, df_lora_005_Time, segment_cols=["segment"]))
    return


@app.cell
def _(df_lora_Time, df_no_lora_Time, spearman_ll_by_segment):
    print(spearman_ll_by_segment(df_no_lora_Time, df_lora_Time, segment_cols=["segment"]))
    return


@app.cell
def _(Path, json, np, pd, plt, re):
    from __future__ import annotations

    from datetime import datetime
    from typing import Optional




    def extract_segment_from_filename(filename: str) -> Optional[str]:
        """
        Extract the flu segment token (e.g., 'mp', 'ha') from common filename patterns.
        """
        name = Path(filename).name
        patterns = [
            r"fine_tune_fasta_([A-Za-z0-9]+)_ft",  # fine_tune_fasta_<SEG>_ft_...
            r"fine_tune_fasta_([A-Za-z0-9]+)_",    # fine_tune_fasta_<SEG>_...
            r"fasta_([A-Za-z0-9]+)_",              # ...fasta_<SEG>_...
            r"^([A-Za-z0-9]+)_metrics",            # <SEG>_metrics.json (fallback)
        ]
        for pat in patterns:
            m = re.search(pat, name)
            if m:
                return m.group(1)
        return None


    def _utc_from_timestamp(ts):
        if ts is None:
            return "unknown time"
        try:
            return datetime.utcfromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M UTC")
        except Exception:
            return "unknown time"


    def _plot_loss(steps, train_loss, val_steps, val_loss, title, dpi=150, save_path: Optional[Path]=None, show=False):
        fig = plt.figure()
        if train_loss:
            plt.plot(steps, train_loss, label="Train loss")
        if val_loss:
            use_val_steps = val_steps if (val_steps and len(val_steps)==len(val_loss)) else list(range(1, len(val_loss)+1))
            plt.plot(use_val_steps, val_loss, label="Validation loss")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title(title)
        plt.legend()
        plt.grid(True)
        if save_path is not None:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight", dpi=dpi)
        if show:
            try:
                from IPython.display import display
                display(fig)
            except Exception:
                pass
        plt.close(fig)


    def _plot_lr(steps, lr, title, dpi=150, save_path: Optional[Path]=None, show=False):
        if not lr:
            return
        fig = plt.figure()
        plt.plot(steps, lr, label="Learning rate")
        plt.xlabel("Step")
        plt.ylabel("Learning rate")
        plt.title(title)
        plt.legend()
        plt.grid(True)
        if save_path is not None:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, bbox_inches="tight", dpi=dpi)
        if show:
            try:
                from IPython.display import display
                display(fig)
            except Exception:
                pass
        plt.close(fig)


    def _summarize(train_loss, val_loss, val_steps):
        final_train = float(train_loss[-1]) if train_loss else None
        final_val   = float(val_loss[-1])   if val_loss   else None
        best_val = best_step = None
        if val_loss:
            best_idx = int(np.argmin(val_loss))
            best_val = float(val_loss[best_idx])
            best_step = int(val_steps[best_idx]) if (val_steps and len(val_steps)==len(val_loss)) else best_idx + 1
        return final_train, final_val, best_val, best_step


    def process_json(json_path: str | Path, out_dir: str | Path, dpi=150, show=False, quiet=False):
        """
        Process a single JSON, write plots, and return a dict summary.
        """
        path = Path(json_path)
        data = json.loads(path.read_text())
        hist = data.get("history", {})

        train_loss = hist.get("train_loss_per_step", []) or []
        val_loss   = hist.get("val_loss_per_step", []) or []
        val_steps  = hist.get("val_loss_steps_at", []) or []
        lr         = hist.get("lr_per_step", []) or []

        steps = list(range(1, len(train_loss) + 1)) if train_loss else list(range(1, len(lr) + 1))

        model    = data.get("model", "unknown-model")
        schedule = data.get("schedule", "unknown-schedule")
        json_segment  = data.get("segment")
        file_segment  = extract_segment_from_filename(path.name)
        segment = file_segment or json_segment or "unknown-seg"
        ts_str  = _utc_from_timestamp(data.get("timestamp"))

        title_base = f"{segment} • {model} • {schedule} schedule • {ts_str}"

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = path.stem
        loss_png = out_dir / f"{stem}_loss.png"
        lr_png   = out_dir / f"{stem}_lr.png"

        _plot_loss(steps, train_loss, val_steps, val_loss, f"Loss vs Step\n{title_base}", dpi=dpi, save_path=loss_png, show=show)
        _plot_lr(steps, lr, f"Learning Rate Schedule (per step)\n{title_base}", dpi=dpi, save_path=lr_png, show=show)

        final_train, final_val, best_val, best_step = _summarize(train_loss, val_loss, val_steps)

        if not quiet:
            print(f"[{path.name}]")
            if final_train is not None: print(f"  Final train loss: {final_train:.4f}")
            if final_val   is not None: print(f"  Final val loss:   {final_val:.4f}")
            if best_val    is not None: print(f"  Best val loss:    {best_val:.4f} at step {best_step}")
            print(f"  Saved: {loss_png.name}")
            if lr: print(f"  Saved: {lr_png.name}")
            print()

        return {
            "file": path.name,
            "loss_png": str(loss_png),
            "lr_png": str(lr_png) if lr else None,
            "final_train": final_train,
            "final_val": final_val,
            "best_val": best_val,
            "best_step": best_step,
            "segment": segment,
            "model": model,
            "schedule": schedule,
            "timestamp_utc": ts_str,
        }


    def batch_generate(indir: str, pattern: str="*.json", out_dir: str | None=None, dpi: int=150, show: bool=False, quiet: bool=False):
        """
        Jupyter-friendly batch runner.

        Returns:
          (df, summaries)
            - df: pandas DataFrame of summaries (or None if pandas not installed)
            - summaries: list of per-file dict summaries
        """
        base = Path(indir)
        if not base.exists() or not base.is_dir():
            raise FileNotFoundError(f"'{indir}' is not a directory or does not exist.")

        out = Path(out_dir) if out_dir else (base / "plots")
        out.mkdir(parents=True, exist_ok=True)

        files = sorted(base.glob(pattern))
        if not files:
            print(f"No files matched pattern '{pattern}' in {base}")
            return (None, [])

        print(f"Found {len(files)} file(s). Writing plots to: {out}")
        summaries = []
        for fp in files:
            try:
                s = process_json(fp, out, dpi=dpi, show=show, quiet=quiet)
                summaries.append(s)
            except Exception as e:
                print(f"Failed on {fp.name}: {e}")

        df = None
        if pd is not None and summaries:
            df = pd.DataFrame(summaries)

        return df, summaries

    return (batch_generate,)


@app.cell
def _(batch_generate):
    df, summaries = batch_generate(
        indir="/Users/cavendan/Downloads/metrics_650_nolora",
        pattern="*_metrics.json",
        out_dir=None,
        dpi=150,
        show=True,
        quiet=False
    )
    df

    return


@app.cell
def _(Path, pd):
    def load_base_results(
        *subfolders,
        base_dir: str = "/Users/cavendan/Desktop/esm-selection/Flu_Snakemake_Pipeline/results/max_freqs_log_likelyhood/h3n2/650M",
    ) -> pd.DataFrame:

        base_path = Path(base_dir)

        # Did the caller explicitly specify a lora branch?
        has_lora = any(str(f).startswith("lora~") for f in subfolders)

        # Build search roots
        if has_lora:
            search_roots = [base_path.joinpath(*subfolders)]
        else:
            # Prefer lora branches if they exist; otherwise fall back to the path itself.
            lora_candidates = [
                base_path.joinpath(*subfolders, "lora~lora"),
                base_path.joinpath(*subfolders, "lora~nolora"),
            ]
            existing_lora = [p for p in lora_candidates if p.exists()]
            search_roots = existing_lora or [base_path.joinpath(*subfolders)]

        frames = []

        for root in search_roots:
            if not root.exists():
                continue

            # Recursively find CSVs (works even if there are no subfolders)
            for csv_path in root.rglob("*.csv"):
                try:
                    df = pd.read_csv(csv_path)
                except Exception:
                    # Skip unreadable/bad files rather than failing the whole load
                    continue

                # Segment from filename (last underscore-separated token)
                seg = csv_path.stem.rsplit("_", 1)[-1].upper()
                df["Segment"] = seg

                # Extract metadata from ALL ancestor directories relative to base_dir
                metadata = {}
                try:
                    rel_parts = csv_path.parent.relative_to(base_path).parts
                except ValueError:
                    rel_parts = csv_path.parent.parts

                for part in rel_parts:
                    if "~" in part:
                        key, value = part.split("~", 1)  # split only once
                        metadata[key] = value

                # Clean up model value if it accidentally contains a nested 'model~' prefix
                model_val = metadata.get("model")
                if model_val and model_val.startswith("model~"):
                    model_val = model_val.split("model~", 1)[1]

                df["Model"] = f"Fine_Tune_{model_val}" if model_val else None
                df["Learning_rate"] = metadata.get("learning_rate", "")
                df["Epochs"] = metadata.get("epochs", "")
                df["Model_training_time"] = metadata.get("time")
                df["tree"] = metadata.get("next_tree") or metadata.get("tree")
                df["LoRA"] = metadata.get("lora")  # "lora" or "nolora"

                frames.append(df)

        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    return (load_base_results,)


@app.cell
def _(load_base_results):
    base_650 = load_base_results()
    return (base_650,)


@app.cell
def _(base_650):
    base_650
    return


@app.cell
def _(base_650, merge_time):
    base_650_Time = merge_time(base_650, "h3n2")
    return (base_650_Time,)


@app.cell
def _(cm, colorsys, os, plt):
    def esm_vs_time_scatterplot_seperate_3(model_df, model_name):
        for segment, group in model_df.groupby('Segment'):

                def darken_color(rgb, factor=0.7):
                    h, l, s = colorsys.rgb_to_hls(*rgb)
                    r, g, b = colorsys.hls_to_rgb(h, max(0, l * factor), s)
                    return (r, g, b, 1.0)

                def plot_esm_score(ax, df, title, Fine_Tune=False):
                    norm = plt.Normalize(df["log_likelihood"].min(), df["log_likelihood"].max())
                    cmap = plt.get_cmap("viridis")
                    colors = cmap(norm(df["log_likelihood"]))
                    edgecolors = [darken_color(c[:3], factor=0.7) for c in colors]
                    sc = ax.scatter(
                        df["time"],
                        df["log_likelihood"],
                        c=colors,
                        edgecolors=edgecolors,
                        linewidths=0.5,
                        alpha=0.7,
                        zorder=1
                    )

                    high_freq_df = df[df["max_frequency"] >= 1].sort_values("time")
                    ax.plot(
                        high_freq_df["time"],
                        high_freq_df["log_likelihood"],
                        linestyle='-',
                        color='black',
                        linewidth=3,
                        alpha=0.6,
                        label='Max Freq ≥ 0.99',
                        zorder=2
                    )

                    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
                    sm.set_array([])
                    cbar = plt.colorbar(sm, ax=ax, orientation='vertical')
                    ax.set_title(title)
                    #if Fine_Tune:
                    #    ax.axvline(1990, color='gray', linestyle='--', linewidth=1.5)
                    ax.set_ylabel("ESM Score")
                    ax.grid(True, color='lightgray', linestyle='-', linewidth=0.75)
                    ax.spines[['right', 'top']].set_visible(False)
                    return ax

                #df1 = model_df[model_df['Model'] == f"Fine_Tune_{model_name}"]
                df1 = model_df
                df1 = df1[df1['Segment'] == segment]
                if segment == "PA":
                    df1 = df1[df1['node'] != 'A/Viamao/LACENRS-974/2015']

                df2 = model_df[model_df['Model'] == model_name]
                df2 = df2[df2['Segment'] == segment]

                if segment == "PA":
                    df2 = df2[df2['node'] != 'A/Viamao/LACENRS-974/2015']

                fig, (ax1) = plt.subplots(1, 1, figsize=(10, 7), sharex=True)
                plot_esm_score(ax1, df1, f"{segment.upper()} - 650M Base Model", Fine_Tune=True)
                #plot_esm_score(ax2, df2, f"{segment.upper()} - 3B Base Model")
                ax1.set_xlabel("Date")

                os.makedirs(f"Flu_Figures/Combined_{model_name}_v_{model_name}_TN_Scatterplots/", exist_ok=True)
                plt.savefig(f"Flu_Figures/Combined_{model_name}_v_{model_name}_TN_Scatterplots/{segment}_{model_name}_v_{model_name}_TN_Scatterplots.png", dpi=300)

                plt.tight_layout()
                plt.show()
    return (esm_vs_time_scatterplot_seperate_3,)


@app.cell
def _(base_650_Time, esm_vs_time_scatterplot_seperate_3):
    esm_vs_time_scatterplot_seperate_3(base_650_Time, "esm2_t33_650M_UR50D")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
