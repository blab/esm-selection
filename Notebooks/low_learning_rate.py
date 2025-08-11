# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "altair==5.5.0",
#     "duckdb==1.3.2",
#     "marimo",
#     "matplotlib==3.9.4",
#     "nbformat==5.10.4",
#     "openai==1.98.0",
#     "pandas==2.3.1",
#     "polars[pyarrow]==1.32.0",
#     "pytest==8.4.1",
#     "seaborn==0.13.2",
#     "sqlglot==27.6.0",
#     "ty==0.0.1a16",
#     "vegafusion==2.0.2",
#     "vl-convert-python==1.8.0",
# ]
# ///

import marimo

__generated_with = "0.14.16"
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
    df_3B_FT_DF_2005_LR_5e_10 = load_fine_tune_results(
        "next_tree~h3n2",
        "epochs~1",
        "learning_rate~5e-10",
        "model~esm2_t36_3B_UR50D",
        "time~2005",
    )

    df_3B_FT_DF_2005_LR_5e_07 = load_fine_tune_results(
        "next_tree~h3n2",
        "epochs~1",
        "learning_rate~5e-07",
        "model~esm2_t36_3B_UR50D",
        "time~2005",
    )

    df_3B_FT_DF_2005_LR_5e_05 = load_fine_tune_results(
        "next_tree~h3n2",
        "epochs~1",
        "learning_rate~5e-05",
        "model~esm2_t36_3B_UR50D",
        "time~2005",
    )
    return (
        df_3B_FT_DF_2005_LR_5e_05,
        df_3B_FT_DF_2005_LR_5e_07,
        df_3B_FT_DF_2005_LR_5e_10,
    )


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
def _(df_3B_FT_DF_2005_LR_5e_10):
    df_3B_FT_DF_2005_LR_5e_10
    return


@app.cell
def _(
    df_3B_FT_DF_2005_LR_5e_05,
    df_3B_FT_DF_2005_LR_5e_07,
    df_3B_FT_DF_2005_LR_5e_10,
    merge_time,
):
    df_3B_FT_DF_2005_LR_5e_10_Time = merge_time(df_3B_FT_DF_2005_LR_5e_10, "h3n2")
    df_3B_FT_DF_2005_LR_5e_07_Time = merge_time(df_3B_FT_DF_2005_LR_5e_07, "h3n2")
    df_3B_FT_DF_2005_LR_5e_05_Time = merge_time(df_3B_FT_DF_2005_LR_5e_05, "h3n2")
    return (df_3B_FT_DF_2005_LR_5e_10_Time,)


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
def _(df_3B_FT_DF_2005_LR_5e_10_Time):
    df_3B_FT_DF_2005_LR_5e_10_Time
    return


@app.cell
def _(cm, colorsys, df_3B_FT_DF_2005_LR_5e_10_Time, os, plt):
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
                        ax.axvline(2005, color='gray', linestyle='--', linewidth=1.5)
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
                plot_esm_score(ax1, df1, f"{segment.upper()} - 3B Fine Tune Model", Fine_Tune=True)
                #plot_esm_score(ax2, df2, f"{segment.upper()} - 3B Base Model")
                ax1.set_xlabel("Date")

                os.makedirs(f"Flu_Figures/Combined_{model_name}_v_{model_name}_TN_Scatterplots/", exist_ok=True)
                plt.savefig(f"Flu_Figures/Combined_{model_name}_v_{model_name}_TN_Scatterplots/{segment}_{model_name}_v_{model_name}_TN_Scatterplots.png", dpi=300)

                plt.tight_layout()
                plt.show()

    esm_vs_time_scatterplot_seperate(df_3B_FT_DF_2005_LR_5e_10_Time, "_esm2_t36_3B_UR50D")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
