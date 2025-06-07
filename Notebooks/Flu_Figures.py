import marimo

__generated_with = "0.13.6"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""### Import Libraries""")
    return


@app.cell
def importlibraries():
    import marimo as mo
    import pandas as pd
    from pathlib import Path
    import os
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    from scipy import stats
    from matplotlib import gridspec
    from scipy.stats import spearmanr
    import json
    import sys
    import argparse
    from augur.utils import annotate_parents_for_tree
    import Bio.Phylo
    from Bio import Phylo
    import matplotlib.colors as mcolors
    from matplotlib import colormaps
    import colorsys
    import matplotlib.cm as cm
    import numpy as np
    import glob
    import re 
    return Path, glob, gridspec, json, mo, os, pd, plt, sns, spearmanr


@app.cell
def _(mo):
    mo.md(r"""### Set Working directory outside of notebook folder""")
    return


@app.cell
def _(os):
    os.chdir("..")
    return


@app.cell
def _(mo):
    mo.md(r"""### Load base dataframes""")
    return


@app.cell
def _(Path, glob, pd):
    def load_base_results(
        *subfolders,
        base_dir: str = "Flu_Snakemake_Pipeline/results/max_freqs_log_likelyhood",
        verbose: bool = True,
    ) -> pd.DataFrame:

        full_path = Path(base_dir, *subfolders)
        glob_pattern = str(full_path / "*.csv")
        csv_files = glob.glob(glob_pattern)

        if verbose:
            print(f"Looking in: {full_path}")
            print(f"Found {len(csv_files)} CSV files.")

        frames = []
        for fname in csv_files:
            try:
                df = pd.read_csv(fname)

                seg = Path(fname).stem.rsplit("_", 1)[-1].upper()
                df["Segment"] = seg

                metadata = {key: value for f in subfolders if "~" in f for key, value in [f.split("~")]}
                df["Model"] = f"Base_{metadata.get('model')}" if "model" in metadata else None
                df["tree"] = metadata.get("next_tree")
                df["Model_training_time"] = metadata.get("time")

                frames.append(df)

            except Exception as e:
                if verbose:
                    print(f"Error reading {fname}: {e}")

        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    return (load_base_results,)


@app.cell
def _(load_base_results):
    df_650_Base = load_base_results(
        "next_tree~h3n2",
        "epochs~1",
        "learning_rate~5e-05",
        "model~esm2_t33_650M_UR50D",
        "time~All",
    )

    df_3B_Base = load_base_results(
        "next_tree~h3n2",
        "epochs~1",
        "learning_rate~5e-05",
        "model~esm2_t36_3B_UR50D",
        "time~All",
    )

    df_15B_Base = load_base_results(
        "next_tree~h3n2",
        "epochs~1",
        "learning_rate~5e-05",
        "model~esm2_t48_15B_UR50D",
        "time~All",
    )
    return df_15B_Base, df_3B_Base, df_650_Base


@app.cell
def _():
    ### Combine Base Dataframes
    return


@app.cell
def _(df_15B_Base, df_3B_Base, df_650_Base, pd):
    df_all_base = pd.concat([df_15B_Base,df_3B_Base, df_650_Base], ignore_index=True)

    df_all_base
    return (df_all_base,)


@app.cell
def _():
    ### Create Runtime Figure
    return


@app.cell
def _(df_all_base, os, pd, plt, sns):
    short_names = {
        'Base_esm2_t33_650M_UR50D': '650M',
        'Base_esm2_t36_3B_UR50D':  '3B',
        'Base_esm2_t48_15B_UR50D': '15B'
    }

    df_all_base['Model_short'] = df_all_base['Model'].map(short_names)

    model_order_short = ['650M', '3B', '15B']
    df_all_base['Model_short'] = pd.Categorical(
        df_all_base['Model_short'],
        categories=model_order_short,
        ordered=True
    )


    df_all_sorted = df_all_base.sort_values('Model_short')

    sns.set_theme(style="whitegrid")

    palette_short = {
        '650M': '#0a2463',
        '3B':   '#f4d35e',
        '15B':  '#890304'
    }

    sns.barplot(
        data      = df_all_sorted,
        x         = 'Segment',
        y         = 'runtime',
        hue       = 'Model_short',
        hue_order = model_order_short,
        palette   = palette_short,
        errorbar  = None,
    )

    plt.title("ESM Runtime")
    plt.xlabel("Segment")
    plt.ylabel("Runtime (Seconds)")
    plt.legend(title="Model", frameon=False, loc='upper left')
    plt.tight_layout()

    newpath = "Flu_Figures/" 
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    output_path = os.path.join(newpath, "runtime_esm2_base_mdls.png")
    plt.savefig(output_path, dpi=300)

    #plt.close()
    plt.show()
    return model_order_short, newpath, palette_short, short_names


@app.cell
def _(mo):
    mo.md(r"""### Create Three Base Model Comparisons per segment figures""")
    return


@app.cell
def _(df_all_base, gridspec, newpath, os, plt, sns, spearmanr):
    #plot all three models side by side, individual figures per segment

    sns.set_theme(style="whitegrid")
    sns.set_style("ticks")

    def plot_regression(ax, data, x_col, y_col, title, ylabel="", color="#0a2463"):
        sns.regplot(data=data, y=y_col, x=x_col, ax=ax, scatter_kws={'s': 50, 'alpha': 0.35, 'color': color}, line_kws={'color': 'black'})
        ax.set_title(title)
        ax.set_xlabel("")
        spearman_corr, p_value = spearmanr(data[y_col], data[x_col])
        textstr = (
            f'ρ = {spearman_corr:.2f}\n'
            f'P < {p_value:.2f}\n'
        )
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.0))
        ax.set_ylabel(ylabel, weight='bold')
        #ax.set_xlim(data[x_col].min(), data[x_col].max())
        ax.set_ylim(0, 1.1)

    def plot_histogram(ax, data, mean_value, title, xlabel="", color="#0a2463"):
        sns.histplot(data=data, x="log_likelyhood", ax=ax, color= color)
        ax.set_title(title)
        ax.axvline(mean_value, color='black', linestyle='-', linewidth=1.5, ymax=0.9)
        ax.set_ylabel("")
        ax.set_xlabel(xlabel, weight='bold' if xlabel else 'normal')

    segment = "MP"

    for segment, group in df_all_base.groupby('Segment'):

        fig = plt.figure(figsize=(5 * 3, 8))
        gs_main = gridspec.GridSpec(3, 3, height_ratios=[1, 0.4, 0.4])

        df = df_all_base[df_all_base['Segment'] == segment]

        if segment == "PA":
            df = df[df['node'] != 'A/Viamao/LACENRS-974/2015']

        #df_650 = df[df['Model'] == "650M"]
        #df_3B = df[df['Model'] == "3B"]
        #df_15B = df[df['Model'] == "15B"]

        df_650 = df[df['Model'] == "Base_esm2_t33_650M_UR50D"]
        df_3B = df[df['Model'] == "Base_esm2_t36_3B_UR50D"]
        df_15B = df[df['Model'] == "Base_esm2_t48_15B_UR50D"]
    
        df_below_1_650 = df_650[df_650['max_frequency'] < 0.1]
        df_above_1_650 = df_650[df_650['max_frequency'] >= 0.99]
        df_below_1_3B = df_3B[df_3B['max_frequency'] < 0.1]
        df_above_1_3B = df_3B[df_3B['max_frequency'] >= 0.99]
        df_below_1_15B = df_15B[df_15B['max_frequency'] < 0.1]
        df_above_1_15B = df_15B[df_15B['max_frequency'] >= 0.99]

        ax = fig.add_subplot(gs_main[0, 0])
        ax_1 = fig.add_subplot(gs_main[0, 1], sharey=ax)
        ax_2 = fig.add_subplot(gs_main[0, 2], sharey=ax)

        ax1 = fig.add_subplot(gs_main[1, 0], sharex=ax)
        ax2 = fig.add_subplot(gs_main[2, 0], sharex=ax)

        ax1_1 = fig.add_subplot(gs_main[1, 1], sharex=ax_1)
        ax2_1 = fig.add_subplot(gs_main[2, 1], sharex=ax_1)

        ax1_2 = fig.add_subplot(gs_main[1, 2], sharex=ax_2)
        ax2_2 = fig.add_subplot(gs_main[2, 2], sharex=ax_2)

        plot_regression(ax, df_650, "log_likelyhood", "max_frequency", f"{segment.upper()} - 650M Model - Max Freq. vs LL", ylabel="Max Frequency")
        plot_regression(ax_1, df_3B, "log_likelyhood", "max_frequency", f"{segment.upper()} - 3B Model - Max Freq. vs LL", color='#f4d35e')
        plot_regression(ax_2, df_15B, "log_likelyhood", "max_frequency", f"{segment.upper()} - 15B Model - Max Freq. vs LL", color='#890304')

        mean_below_1_650 = df_below_1_650['log_likelyhood'].mean()
        mean_below_1_3B = df_below_1_3B['log_likelyhood'].mean()
        mean_below_1_15B = df_below_1_15B['log_likelyhood'].mean()

        plot_histogram(ax1, df_below_1_650, mean_below_1_650, "max. freq. (0.0, 0.1)")
        plot_histogram(ax1_1, df_below_1_3B, mean_below_1_3B, "max. freq. (0.0, 0.1)", color='#f4d35e')
        plot_histogram(ax1_2, df_below_1_15B, mean_below_1_15B, "max. freq. (0.0, 0.1)", color='#890304')

        mean_above_1_650 = df_above_1_650['log_likelyhood'].mean()
        mean_above_1_3B = df_above_1_3B['log_likelyhood'].mean()
        mean_above_1_15B = df_above_1_15B['log_likelyhood'].mean()

        plot_histogram(ax2, df_above_1_650, mean_above_1_650, "max. freq. (0.99, 1.0)", xlabel="Log Likelyhood")
        plot_histogram(ax2_1, df_above_1_3B, mean_above_1_3B, "max. freq. (0.99, 1.0)", xlabel="Log Likelyhood", color='#f4d35e')
        plot_histogram(ax2_2, df_above_1_15B, mean_above_1_15B, "max. freq. (0.99, 1.0)", xlabel="Log Likelyhood", color='#890304')

        for axis in [ax, ax_1, ax_2, ax1, ax2, ax1_1, ax2_1, ax1_2, ax2_2]:
            axis.spines[['right', 'top']].set_visible(False)

        fig.text(0.01, 0.3, 'Count', va='center', rotation='vertical', fontsize=12, weight='bold')
        plt.tight_layout()

        newpath_ESM_base = "Flu_Figures/ESM_vs_Max_Freq_Plots_Base_Models_Comparison/" 
        if not os.path.exists(newpath):
                os.makedirs(newpath)

        plt.savefig(f"Flu_Figures/ESM_vs_Max_Freq_Plots_Base_Models_Comparison/{segment}_LL_vs_Max_Frequency.png", dpi=300)
        plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""### Save Summary Stats for base models""")
    return


@app.cell
def _(df_all_base, newpath, os, pd, spearmanr):
    #Save summary stats for base model comparison

    results = []

    for model, group_all_base in df_all_base.groupby('Model'):
      for segment_all_base, group_all_base in df_all_base.groupby('Segment'):
    
        df_all_base_filtered = df_all_base[df_all_base['Segment'] == segment_all_base]
        df_all_base_filtered = df_all_base_filtered[df_all_base_filtered['Model'] == model]

        if segment_all_base == "pa":
          df_all_base_filtered = df_all_base_filtered[df_all_base_filtered['node'] != 'A/Viamao/LACENRS-974/2015']

        df_below_01 = df_all_base_filtered[df_all_base_filtered['max_frequency'] < 0.1]
        df_above_1 = df_all_base_filtered[df_all_base_filtered['max_frequency'] >= 0.99]
    
        spearman_corr, p_value = spearmanr(df_all_base_filtered['max_frequency'], df_all_base_filtered['log_likelyhood'])

        results.append({
            "Model": model,
            "Segment": segment_all_base,
            "Spearman Correlation Coefficient between Max Frequency and LL": spearman_corr,
            "P-value": p_value,
            "Mean ESM LL below 0.1": df_below_01['log_likelyhood'].mean(),
            "Mean ESM LL above 0.99": df_above_1['log_likelyhood'].mean(),
            "Difference in LL ESM Means": df_above_1['log_likelyhood'].mean() - df_below_01['log_likelyhood'].mean()
        })

        results_df = pd.DataFrame(results)

    print(results_df)

    newpath_all_base = "Flu_Summary_Statistics/" 
    if not os.path.exists(newpath):
            os.makedirs(newpath)

    results_df.to_csv("Flu_Summary_Statistics/ESM_vs_Max_Freq_Summary_Statistics_Base_Models_Comparison.csv", index=False)
    return (results_df,)


@app.cell
def _(results_df):
    # Get average of the Spearman correlation coefficients for each model

    results_df.groupby('Model')['Spearman Correlation Coefficient between Max Frequency and LL'].mean()
    return


@app.cell
def _(mo):
    mo.md(r"""### Spearman Correlation Coefficient Figure For Base Models""")
    return


@app.cell
def _(model_order_short, palette_short, pd, plt, results_df, short_names, sns):
    def make_base_speareman_summary_figure(results_df):

        results_df['Model_short'] = results_df['Model'].map(short_names)
        results_df['Model_short'] = pd.Categorical(
            results_df['Model_short'],
            categories=model_order_short,
            ordered=True
        )

        results_df = results_df.dropna(subset=['Model_short'])

        results_df_sorted = results_df.sort_values('Model_short')

        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(8,5))

        sns.barplot(
            data=results_df_sorted,
            x='Segment',
            y='Spearman Correlation Coefficient between Max Frequency and LL',
            hue='Model_short',           
            hue_order=model_order_short, 
            palette=palette_short,
            errorbar=None
        )

        plt.title("Spearman Correlation between Max Freq. and LL across Models")
        plt.xlabel("Segment")
        plt.ylabel("Spearman Correlation Coefficient")
        plt.legend(title="Model Size", frameon=False, loc='lower left')
        plt.tight_layout()
        plt.savefig("Flu_Figures/Spearman_Correlation_Comparision_Base_Model.png", dpi=300)
    
        return plt.show()

    make_base_speareman_summary_figure(results_df)
    return


@app.cell
def _(mo):
    mo.md(r"""### Function To Add ESM to Nextstrain Tree""")
    return


@app.cell
def _(json, os, pd):
    #Add ESM LL to Nextstrain Tree

    def add_ESM_LL_to_Nextstrain_Tree(directory, output_dir):
      for filename in os.listdir(directory):
          if filename.endswith(".csv"):
              file_path = os.path.join(directory, filename)

              segment = filename.rsplit('_', 1)[-1].replace('.csv', '')

              LL_freq_file_df = pd.read_csv(file_path)

              if segment == "pa":
                LL_freq_file_df = LL_freq_file_df[LL_freq_file_df['node'] != 'A/Viamao/LACENRS-974/2015']

              formatted_dict = {
                  "nodes": {
                      key: {"ESM_score": value}
                      for key, value in zip(LL_freq_file_df["node"], LL_freq_file_df["log_likelyhood"])
                  }
              }

              with open(f"/Users/Carlos/Desktop/Bedford/esm-selection/Flu_Snakemake_Pipeline/input/trees/h3n2/{segment}.json", 'r') as fh:
              #with open(f"Flu_Snakemake_Pipeline/h3n2_Sequences/h3n2_60y_{segment}.json", 'r') as fh:
                dataset = json.load(fh)


              node_data = formatted_dict
              esm_scores = {name: info["ESM_score"] for name, info in node_data['nodes'].items()}

              def recurse(n):
                if n["name"] in esm_scores:
                  n["node_attrs"]["ESM_score"] = {"value": esm_scores[n["name"]]}
                for c in n.get("children", []):
                  recurse(c)

              recurse(dataset["tree"])

              dataset['meta']["colorings"].insert(0, {"key": "ESM_score", "title": "esm scores", "type": "continuous"})

              os.makedirs(output_dir, exist_ok=True)
          
              with open(f"{output_dir}/{segment}_ESM_Tree.json", 'w') as fh:
                json.dump(dataset, fh, indent=2)
    return (add_ESM_LL_to_Nextstrain_Tree,)


@app.cell
def _(mo):
    mo.md(r"""### Add ESM Log Likelihood To Nextstrain Tree:""")
    return


@app.cell
def _(mo):
    mo.md(r"""##### 650M ESM Nextstrain Tree""")
    return


@app.cell
def _(add_ESM_LL_to_Nextstrain_Tree, os):
    folder_path_650M = "Flu_Snakemake_Pipeline/results/max_freqs_log_likelyhood/next_tree~h3n2/epochs~1/learning_rate~5e-05/model~esm2_t33_650M_UR50D/time~1990"

    if os.path.isdir(folder_path_650M):

        add_ESM_LL_to_Nextstrain_Tree(
            folder_path_650M,
            f"Flu_Trees/ESM_Trees_650M_base"
        )
    return


@app.cell
def _(mo):
    mo.md(r"""##### 3B ESM Nextstrain Tree""")
    return


@app.cell
def _(add_ESM_LL_to_Nextstrain_Tree, os):
    folder_path_3B = "/Users/Carlos/Desktop/Bedford/esm-selection/Flu_Snakemake_Pipeline/results/max_freqs_log_likelyhood/next_tree~h3n2/epochs~1/learning_rate~5e-05/model~esm2_t36_3B_UR50D/time~1990"

    if os.path.isdir(folder_path_3B):

        add_ESM_LL_to_Nextstrain_Tree(
            folder_path_3B,
            f"Flu_Trees/ESM_Trees_3B_base"
        )
    return


@app.cell
def _(mo):
    mo.md(r"""##### 15B ESM Nextstrain Tree""")
    return


@app.cell
def _(add_ESM_LL_to_Nextstrain_Tree, os):
    folder_path_15B = "/Users/Carlos/Desktop/Bedford/esm-selection/Flu_Snakemake_Pipeline/results/max_freqs_log_likelyhood/next_tree~h3n2/epochs~1/learning_rate~5e-05/model~esm2_t48_15B_UR50D/time~All"

    if os.path.isdir(folder_path_15B):

        add_ESM_LL_to_Nextstrain_Tree(
            folder_path_15B,
            f"Flu_Trees/ESM_Trees_15B_base"
        )
    return


@app.cell
def _(mo):
    mo.md(r"""### Function to load in fine tune dataframes""")
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
def _(mo):
    mo.md(r"""### Load in fine tuned 1990 datasets""")
    return


@app.cell
def _(load_fine_tune_results):
    #Default fine tune

    df_650_FT_DF_Time_1990 = load_fine_tune_results(
        "next_tree~h3n2",
        "epochs~1",
        "learning_rate~5e-05",
        "model~esm2_t33_650M_UR50D",
        "time~1990",
    )

    df_3B_FT_DF_Time_1990 = load_fine_tune_results(
        "next_tree~h3n2",
        "epochs~1",
        "learning_rate~5e-05",
        "model~esm2_t36_3B_UR50D",
        "time~1990",
    )

    # epochs 5

    df_650_FT_DF_Time_1990_EP_5 = load_fine_tune_results(
        "next_tree~h3n2",
        "epochs~5",
        "learning_rate~5e-05",
        "model~esm2_t33_650M_UR50D",
        "time~1990",
    )

    df_3B_FT_DF_Time_1990_EP_5 = load_fine_tune_results(
        "next_tree~h3n2",
        "epochs~5",
        "learning_rate~5e-05",
        "model~esm2_t36_3B_UR50D",
        "time~1990",
    )

    # learning rate adjustments

    df_650_FT_DF_Time_1990_LR_2_5e_05 = load_fine_tune_results(
        "next_tree~h3n2",
        "epochs~1",
        "learning_rate~2.5e-05",
        "model~esm2_t33_650M_UR50D",
        "time~1990",
    )

    df_3B_FT_DF_Time_1990_LR_2_5e_05 = load_fine_tune_results(
        "next_tree~h3n2",
        "epochs~1",
        "learning_rate~2.5e-05",
        "model~esm2_t36_3B_UR50D",
        "time~1990",
    )

    df_650_FT_DF_Time_1990_LR_1e_05 = load_fine_tune_results(
        "next_tree~h3n2",
        "epochs~1",
        "learning_rate~1e-05",
        "model~esm2_t33_650M_UR50D",
        "time~1990",
    )

    df_3B_FT_DF_Time_1990_LR_1e_05 = load_fine_tune_results(
        "next_tree~h3n2",
        "epochs~1",
        "learning_rate~1e-05",
        "model~esm2_t36_3B_UR50D",
        "time~1990",
    )

    df_650_FT_DF_Time_1990_LR_5e_06 = load_fine_tune_results(
        "next_tree~h3n2",
        "epochs~1",
        "learning_rate~5e-06",
        "model~esm2_t33_650M_UR50D",
        "time~1990",
    )

    df_3B_FT_DF_Time_1990_LR_5e_06 = load_fine_tune_results(
        "next_tree~h3n2",
        "epochs~1",
        "learning_rate~5e-06",
        "model~esm2_t36_3B_UR50D",
        "time~1990",
    )
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
