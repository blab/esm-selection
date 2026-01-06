# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "altair==5.5.0",
#     "duckdb==1.3.2",
#     "ipython==8.18.1",
#     "marimo",
#     "matplotlib==3.9.4",
#     "nbformat==5.10.4",
#     "numpy==2.0.2",
#     "openai==1.95.1",
#     "pandas==2.3.1",
#     "polars[pyarrow]==1.31.0",
#     "pytest==8.4.1",
#     "scikit-learn==1.6.1",
#     "scipy==1.13.1",
#     "seaborn==0.13.2",
#     "sqlglot==27.0.0",
#     "ty==0.0.1a15",
#     "vegafusion==2.0.2",
#     "vl-convert-python==1.8.0",
# ]
# ///

import marimo

__generated_with = "0.17.6"
app = marimo.App(width="medium", app_title="LOESS Updates")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Applying LOESS to previous ESM work
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In this notebook I will be reproducing some of my previous work I presented at lab meeting with LOESS correction applied, to remove the negative trend of ESM score vs Time.

    All code is included in this notebook, but I collapsed it for cleanliness.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setup
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Install Libraries
    """)
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import numpy as np
    import os
    import seaborn as sns
    from scipy.stats import spearmanr
    import colorsys
    import matplotlib.cm as cm
    from matplotlib.ticker import ScalarFormatter
    from matplotlib.ticker import FormatStrFormatter
    #from mpl_toolkits.axes_grid1 import make_axes_locatable
    from sklearn.linear_model import LinearRegression
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    np.seterr(divide='ignore', over='ignore', invalid='ignore')
    import ty
    return (
        LinearRegression,
        ScalarFormatter,
        cm,
        colorsys,
        mo,
        mpl,
        np,
        pd,
        plt,
        sns,
        spearmanr,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Patch to fix LOESS package

    There was an issue previously where some flu segments with a generally flat slope, broke the LOESS library. This was due to areas high data similarity causing a division by zero error. I fixed this by manually installing the library and changing instances where there would be a division by a zero to division of really small number instead.

    This allows areas of a flu segment that have a relativley flat slope to remain unchanged, but we can apply LOESS to the rest of the points.
    """)
    return


@app.cell(hide_code=True)
def _(np):
    class polyfit1d:

        def __init__(self, x, y, degree, weights):

            sqw = np.sqrt(weights)
            a = x[:, None]**np.arange(degree + 1)
            self.degree = degree
            self.coeff = np.linalg.lstsq(a*sqw[:, None], y*sqw, rcond=None)[0]
            self.yfit = a @ self.coeff

        def eval(self, x):

            a = x**np.arange(self.degree + 1)
            yout = a @ self.coeff

            return yout

    def biweight_sigma(y, zero=False):

        y = np.ravel(y)
        if zero:
            d = y
        else:
            d = y - np.median(y)

        mad = np.median(np.abs(d))
        u2 = (d / (9.*mad))**2  # c = 9
        good = u2 < 1.
        u1 = 1. - u2[good]
        num = y.size * ((d[good]*u1**2)**2).sum()
        den = (u1*(1. - 5.*u2[good])).sum()
        sigma = np.sqrt(num/(den*(den - 1.)))  # see note in above reference

        return sigma


    def rotate_points(x, y, ang):
        theta = np.radians(ang)
        xNew = x*np.cos(theta) - y*np.sin(theta)
        yNew = x*np.sin(theta) + y*np.cos(theta)

        return xNew, yNew

    def loess_1d(x, y, xnew=None, degree=1, frac=0.5, npoints=None, rotate=False, sigy=None):

        if frac == 0:
            return y, np.ones_like(y)

        assert x.size == y.size, 'Input vectors (X, Y) must have the same size'

        if npoints is None:
            npoints = int(np.ceil(frac*x.size))

        if rotate:

            assert xnew is None, "`rotate` not supported with `xnew`"

            # Robust calculation of the axis of maximum variance
            #
            nsteps = 180
            angles = np.arange(nsteps)
            sig = np.zeros(nsteps)
            for j, ang in enumerate(angles):
                x2, y2 = rotate_points(x, y, ang)
                sig[j] = biweight_sigma(x2)
            k = np.argmax(sig)  # Find index of max value
            x, y = rotate_points(x, y, angles[k])

        if xnew is None:

            xnew = x

        ynew = np.empty_like(xnew, dtype=float)
        wout = np.empty_like(ynew)

        for j, xj in enumerate(xnew):

            dist = np.abs(x - xj)
            w = np.argsort(dist)[:npoints]
            dist_weights = (1 - (dist[w]/dist[w[-1]])**3)**3  # tricube function distance weights
            yfit = polyfit1d(x[w], y[w], degree, dist_weights).yfit

            # Robust fit from Sec.2 of Cleveland (1979)
            # Use errors if those are known.
            #
            bad = None
            for p in range(10):  # do at most 10 iterations

                if sigy is None:                # Errors are unknown
                    aerr = np.abs(yfit - y[w])  # Note ABS()
                    mad = np.median(aerr)       # Characteristic scale

                    if mad == 0:
                        #mad = np.finfo(float).tiny
                        mad = np.maximum(mad, 1e-10)
                    uu = (aerr/(6*mad))**2      # For a Gaussian: sigma=1.4826*MAD
                else:                           # Errors are assumed known
                    uu = ((yfit - y[w])/(4*sigy[w]))**2  # 4*sig ~ 6*mad

                uu = uu.clip(0, 1)
                biweights = (1 - uu)**2
                tot_weights = dist_weights*biweights
                poly = polyfit1d(x[w], y[w], degree, tot_weights)
                yfit = poly.yfit
                badOld = bad
                bad = biweights < 0.34    # 99% confidence outliers
                if np.array_equal(badOld, bad):
                    break

            if np.array_equal(x, xnew):
                ynew[j] = yfit[0]
                wout[j] = biweights[0]
            else:
                ynew[j] = poly.eval(xj)
                wout[j] = 1

        if rotate:
            xnew, ynew = rotate_points(xnew, ynew, -angles[k])
            j = np.argsort(xnew)
            xnew, ynew = xnew[j], ynew[j]

        return xnew, ynew, wout
    return (loess_1d,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Regenerating ESM vs Time Plots
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Load dataframes trained up to 1990
    """)
    return


@app.cell(hide_code=True)
def _(pd):
    df_650_FT_DF = pd.read_csv(
        "Dataframes/650M_Fine_Tune_Up_To_1990.csv", keep_default_na=False
    )

    df_3B_FT_DF = pd.read_csv(
        "Dataframes/3B_Fine_Tune_Up_To_1990.csv", keep_default_na=False
    )
    return df_3B_FT_DF, df_650_FT_DF


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Rewrite LOESS application to work with any dataframe and store as a function
    """)
    return


@app.cell(hide_code=True)
def _(loess_1d, pd):
    def apply_loess_to_segment(
        df: pd.DataFrame,
        x_col: str = "time",
        y_col: str = "log_likelihood",
        degree: int = 2,
        frac: float = 0.15,
    ) -> pd.DataFrame:
        x = df[x_col].values
        y = df[y_col].values

        _, y_smoothed, w_smoothed = loess_1d(
            x=x,
            y=y,
            xnew=x,
            degree=degree,
            frac=frac,
        )

        df.loc[:, f"{y_col}_LOESS"] = y_smoothed
        df.loc[:, "loess_weight"] = w_smoothed

        return df


    def apply_loess_to_finetune_models(
        df: pd.DataFrame,
        x_col: str = "time",
        y_col: str = "log_likelihood",
        degree: int = 2,
        frac: float = 0.15,
    ) -> pd.DataFrame:
        df = df.copy()

        fine_tune_mask = df["Model"].str.startswith("Fine_Tune")
        fine_tune_df = df.loc[fine_tune_mask].copy()

        smoothed_parts: list[pd.DataFrame] = []

        for (_, _), group in fine_tune_df.groupby(
            ["Segment", "Model"], sort=False
        ):
            group = apply_loess_to_segment(
                group, x_col=x_col, y_col=y_col, degree=degree, frac=frac
            )
            smoothed_parts.append(
                group[
                    ["Segment", "Model", x_col, f"{y_col}_LOESS", "loess_weight"]
                ]
            )

        if smoothed_parts:
            smoothed_df = pd.concat(smoothed_parts, ignore_index=True)
            smoothed_df = smoothed_df.groupby(
                ["Segment", "Model", x_col], as_index=False
            ).first()
        else:
            smoothed_df = pd.DataFrame(
                columns=[
                    "Segment",
                    "Model",
                    x_col,
                    f"{y_col}_LOESS",
                    "loess_weight",
                ]
            )

        df = df.merge(
            smoothed_df,
            on=["Segment", "Model", x_col],
            how="left",
            validate="many_to_one",
        )

        df[f"{y_col}_LOESS"] = pd.to_numeric(df[f"{y_col}_LOESS"], errors="coerce")
        df["corrected_log_likelihood"] = df[y_col] - df[f"{y_col}_LOESS"]

        return df
    return (apply_loess_to_finetune_models,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Apply LOESS to Fine Tune ESM Models (650M and 3B parameters) trained up to 1990
    """)
    return


@app.cell(hide_code=True)
def _(apply_loess_to_finetune_models, df_3B_FT_DF, df_650_FT_DF):
    df_650_FT_DF_with_loess = apply_loess_to_finetune_models(df_650_FT_DF)
    df_3B_FT_DF_with_loess = apply_loess_to_finetune_models(df_3B_FT_DF)

    df_650_FT_DF_with_loess.to_csv('Dataframes/df_650_FT_DF_with_loess.csv', index=False)
    df_3B_FT_DF_with_loess.to_csv('Dataframes/df_3B_FT_DF_with_loess.csv', index=False)
    return df_3B_FT_DF_with_loess, df_650_FT_DF_with_loess


@app.cell(hide_code=True)
def _(df_3B_FT_DF_with_loess, df_650_FT_DF_with_loess):
    df_650_FT_DF_with_loess_FT = df_650_FT_DF_with_loess[df_650_FT_DF_with_loess['Model'] == 'Fine_Tune_650M']
    df_3B_FT_DF_with_loess_FT = df_3B_FT_DF_with_loess[df_3B_FT_DF_with_loess['Model'] == 'Fine_Tune_3B']
    return df_3B_FT_DF_with_loess_FT, df_650_FT_DF_with_loess_FT


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Generate time vs ESM score (log likelihood) plots
    """)
    return


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
def _(ScalarFormatter, cm, colorsys, np, plt):
    def darken_color(rgb, factor=0.7):
        h, l, s = colorsys.rgb_to_hls(*rgb)
        r, g, b = colorsys.hls_to_rgb(h, max(0, l * factor), s)
        return (r, g, b, 1.0)

    def plot_esm_score(ax, df, title, Fine_Tune=False, LOESS=False):

        if(LOESS == False):
            ll_col = "log_likelihood" 
        else: 
            ll_col = "corrected_log_likelihood"

        norm = plt.Normalize(df[ll_col].min(), df[ll_col].max())
        cmap = plt.get_cmap("viridis")
        colors = cmap(norm(df[ll_col]))
        edgecolors = [darken_color(c[:3], factor=0.7) for c in colors]

        ax.scatter(
            df["time"],
            df[ll_col],
            c=colors,
            edgecolors=edgecolors,
            linewidths=0.5,
            alpha=0.7,
            zorder=1
        )

        high_freq_df = (
            df[
                (df["max_frequency"] > 1) &
                (df["node"].str.contains("NODE_"))
            ]
            .sort_values("time")
        )
        ax.plot(
            high_freq_df["time"],
            high_freq_df[ll_col],
            linestyle='-',
            color='black',
            linewidth=3,
            alpha=0.6,
            label='Max Freq = 1 & NODE_',
            zorder=2
        )

        ax.yaxis.offsetText.set_visible(False)

        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = plt.colorbar(sm, ax=ax, orientation='vertical',
                            pad=0.02,        
                            extend='both'
                           )   

        cbar.ax.yaxis.offsetText.set_visible(False)

        ax.set_title(title, fontsize=10)

        if Fine_Tune:
            ax.axvline(1990, color='gray', linestyle='--', linewidth=1.5)

        ax.set_ylabel("ESM Score", fontsize=8)
        ax.grid(True, color='lightgray', linestyle='-', linewidth=0.75)
        ax.spines[['right', 'top']].set_visible(False)
        ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        ax.ticklabel_format(style='plain', axis='x')
        ax.set_xlim(1965, 2025)

        y_min, y_max = df[ll_col].min(), df[ll_col].max()
        pad = (y_max - y_min) * 0.05 if y_max != y_min else 1.0
        ax.set_ylim(y_min - pad, y_max + pad)

        return ax

    def esm_vs_time_3x8_grid(model_df, model_name):
        segments = sorted(model_df['Segment'].unique())
        fig, axs = plt.subplots(len(segments), 3, figsize=(15, 30), sharex=True, sharey=False)

        for i, segment in enumerate(segments):
            df_ft   = model_df[(model_df['Model'] == f"Fine_Tune_{model_name}") & (model_df['Segment'] == segment)]
            df_base = model_df[(model_df['Model'] ==       model_name    ) & (model_df['Segment'] == segment)]

            if segment == "PA":
                df_ft   = df_ft[df_ft['node'] != 'A/Viamao/LACENRS-974/2015']
                df_base = df_base[df_base['node'] != 'A/Viamao/LACENRS-974/2015']

            ax1, ax2, ax3 = axs[i, 0], axs[i, 1], axs[i, 2]

            plot_esm_score(ax1, df_base, f"{segment.upper()} • {model_name} Base")
            plot_esm_score(ax2, df_ft,   f"{segment.upper()} • {model_name} FT", Fine_Tune=True)
            plot_esm_score(ax3, df_ft, f"{segment.upper()} • {model_name} LOESS", Fine_Tune=True, LOESS=True)

            if i == len(segments) - 1:
                for ax in (ax1, ax2, ax3):
                    ax.set_xlabel("Year", fontsize=8)


            years = np.arange(1960, 2021, 20)    
            for ax in axs.flat:                    
                ax.set_xticks(years)               
                ax.set_xticklabels(years,         
                                   rotation=0,    
                                   ha='right',
                                   fontsize=10
                                  )
                ax.tick_params(axis='x',
                               which='major',
                               labelbottom=True)   

        plt.tight_layout(h_pad=2, w_pad=1)

        #plt.show()

        return plt.gcf() 
    return darken_color, esm_vs_time_3x8_grid


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Generate time vs ESM score plots for 650M parameter model trained up to 1990
    """)
    return


@app.cell(hide_code=True)
def _(df_650_FT_DF_with_loess, esm_vs_time_3x8_grid):
    esm_vs_time_3x8_grid(df_650_FT_DF_with_loess, "650M")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    _In this figure column 1 are the ESM scores generated from the base 650M parameter model. Column 2 is the 650M model fine tuned to each individual segement. Column 3 is the same fine tuned results with LOESS correction applied._

    This figure shows us that the LOESS correction is working correctly. For each segment, there is no longer a droppout of ESM score over time.

    For some segments, such as MP, which were relatively flat before LOESS correction, there is little difference between the fine tune and LOESS corrected fine tune plots in terms of general trend, but average ESM score is now much closer to zero. Points that fall before 1990 (the training period), are less impacted by LOESS corrected than test period points - post 1990.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Generate time vs ESM score plots for 3B parameter model trained up to 1990
    """)
    return


@app.cell(hide_code=True)
def _(df_3B_FT_DF_with_loess, esm_vs_time_3x8_grid):
    esm_vs_time_3x8_grid(df_3B_FT_DF_with_loess, "3B")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    There is not a major difference with the 650M and 3B parameter model figures, and we see here that the trend removal also is working for the 3B model segements too.
    """)
    return


@app.cell(hide_code=True)
def _(ScalarFormatter, cm, darken_color, np, plt):
    def plot_loess_finetune(ax, df, title):
        ll_col = "corrected_log_likelihood"

        norm = plt.Normalize(df[ll_col].min(), df[ll_col].max())
        cmap = plt.get_cmap("viridis")
        colors = cmap(norm(df[ll_col]))
        edgecolors = [darken_color(c[:3], factor=0.7) for c in colors]

        ax.scatter(
            df["time"], df[ll_col], c=colors,
            edgecolors=edgecolors, linewidths=0.5, alpha=0.7, zorder=1
        )

        high_freq = df[df["max_frequency"] >= 1].sort_values("time")
        ax.plot(
            high_freq["time"], high_freq[ll_col],
            linestyle='-', color='black', linewidth=3, alpha=0.6,
            label='Max Freq ≥ 1', zorder=2
        )

        ax.yaxis.offsetText.set_visible(False)
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.02, extend='both')
        cbar.ax.yaxis.offsetText.set_visible(False)

        ax.set_title(title, fontsize=10)
        ax.axvline(1990, color='gray', linestyle='--', linewidth=1.5)
        ax.set_ylabel("ESM Score", fontsize=8)
        ax.grid(True, color='lightgray', linestyle='-', linewidth=0.75)
        ax.spines[['right', 'top']].set_visible(False)
        ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        ax.ticklabel_format(style='plain', axis='x')
        ax.set_xlim(1965, 2025)
        y_min, y_max = df[ll_col].min(), df[ll_col].max()
        pad = (y_max - y_min) * 0.05 if y_max != y_min else 1.0
        ax.set_ylim(y_min - pad, y_max + pad)
        return ax

    def two_col_loess_finetune(df_left, df_right, model_name_left, model_name_right):

        segments = sorted(df_left['Segment'].unique())
        fig, axs = plt.subplots(len(segments), 2, figsize=(12, 4 * len(segments)), sharex=True)

        for i, seg in enumerate(segments):
            # Filter only fine-tune LOESS data
            df_left_ft  = df_left[(df_left['Model'] == f"Fine_Tune_{model_name_left}") & (df_left['Segment'] == seg)]
            df_right_ft = df_right[(df_right['Model'] == f"Fine_Tune_{model_name_right}") & (df_right['Segment'] == seg)]

            if seg == "PA":
                df_left_ft = df_left_ft[df_left_ft['node'] != 'A/Viamao/LACENRS-974/2015']
                df_right_ft = df_right_ft[df_right_ft['node'] != 'A/Viamao/LACENRS-974/2015']

            ax_left, ax_right = axs[i, 0], axs[i, 1]
            plot_loess_finetune(ax_left,  df_left_ft,  f"{seg} • Fine_Tune_{model_name_left}")
            plot_loess_finetune(ax_right, df_right_ft, f"{seg} • Fine_Tune_{model_name_right}")

            # X-axis and ticks
            if i == len(segments) - 1:
                ax_left.set_xlabel("Year", fontsize=8)
                ax_right.set_xlabel("Year", fontsize=8)

            years = np.arange(1960, 2021, 20)
            for ax in (ax_left, ax_right):
                ax.set_xticks(years)
                ax.set_xticklabels(years, rotation=0, ha='right', fontsize=10)

        plt.tight_layout(h_pad=2, w_pad=1)
        return plt.gcf()
    return (two_col_loess_finetune,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### ESM with LOESS correction vs Maximum Frequency
    """)
    return


@app.cell(hide_code=True)
def _(df_3B_FT_DF_with_loess, df_650_FT_DF_with_loess, two_col_loess_finetune):
    two_col_loess_finetune(df_650_FT_DF_with_loess, df_3B_FT_DF_with_loess, "650M", "3B")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Per TB's suggestion I removed the first two columns of my figures with the base dataset and finetune without LOESS.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Find position of trunk along time
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(rf"""
    ### Calculate nearest neighbors

    For every internal node with a maximum frequency of  1, I found the 10 nearest terminal nodes. From there I got the minimum a maximum of those 10 neighbor nodes. This is a method I thought of to quantify where trunk of the tree sits relative to the points around it.
    """)
    return


@app.cell(hide_code=True)
def _(List, np, pd):
    def neighbor_likelihood_extremes_by_segment(
        df: pd.DataFrame,
        n_neighbors: int = 10,
        freq_threshold: float = 1.0,
    ) -> pd.DataFrame:

        required = {
            "Segment",
            "node",
            "max_frequency",
            "corrected_log_likelihood",
            "time",
        }
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"DataFrame missing columns: {', '.join(missing)}")

        summaries: List[dict] = []

        for seg, seg_df in df.groupby("Segment"):
            node_max = seg_df.groupby("node")["max_frequency"].max()
            eligible_nodes = node_max[node_max >= freq_threshold].index

            seg_idx      = seg_df.index.to_numpy()
            times        = seg_df["time"].to_numpy()
            log_ll       = seg_df["corrected_log_likelihood"].to_numpy()
            nodes_arr    = seg_df["node"].to_numpy()
            max_freq_arr = seg_df["max_frequency"].to_numpy()

            for node in eligible_nodes:
                node_max_freq = node_max[node]

                rep_mask = (nodes_arr == node) & (max_freq_arr == node_max_freq)
                if not rep_mask.any():
                    continue

                rep_i = np.where(rep_mask)[0][0]
                t0 = times[rep_i]
                node_ll = log_ll[rep_i]

                if node_max_freq == freq_threshold == 1.0:
                    node_label = f"NODE_{node}"
                else:
                    node_label = node

                deltas = np.abs(times - t0)
                other_idx = np.where(nodes_arr != node)[0]

                if other_idx.size:
                    nearest  = other_idx[np.argsort(deltas[other_idx])[:n_neighbors]]
                    neigh_ll = log_ll[nearest]
                    best, worst = neigh_ll.max(), neigh_ll.min()
                else:
                    best = worst = np.nan

                if np.isnan(best) or np.isnan(worst) or best == worst:
                    trunk_position = np.nan
                else:
                    trunk_position = (node_ll - worst) / (best - worst)
                    trunk_position = np.clip(trunk_position, 0.0, 1.0)

                summaries.append({
                    "Segment": seg,
                    "node": node_label,
                    "node_time": t0,
                    "max_frequency": node_max_freq,
                    "node_log_likelihood": node_ll,
                    "best_neighbor_ll": best,
                    "worst_neighbor_ll": worst,
                    "trunk_position": trunk_position,
                })

        return pd.DataFrame(summaries)
    return (neighbor_likelihood_extremes_by_segment,)


@app.cell(hide_code=True)
def _(
    df_3B_FT_DF_with_loess_FT,
    df_650_FT_DF_with_loess_FT,
    neighbor_likelihood_extremes_by_segment,
):
    df_650_NN = neighbor_likelihood_extremes_by_segment(df_650_FT_DF_with_loess_FT, n_neighbors=10)
    df_3B_NN = neighbor_likelihood_extremes_by_segment(df_3B_FT_DF_with_loess_FT, n_neighbors=10)
    return df_3B_NN, df_650_NN


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Plot position of trunk over time
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To calculate trunk position for each node with a maximum frequency of 1, I found the 10 points nearest in time to that given node. I found the maximum and minimum log-likelihoods for these 10 neighbor nodes. Then I found where the log-likelihood for our trunk node (node with a maximum frequency of 1) sits between the maximum and minimum log-likelihoods for the neighbor nodes.

    Ex:

    *b* = Max LL of neighbor nodes: 5

    *a* = Min LL of neighbor nodes: -5

    *x* = Internal node with max frequency of 1 of -1

    < -5 - - - [-1] - - - - - 5 >

    -1 sits at the lower 40% of between the maximum and minimum of the nearest neighbors.

    Or can written as the equation:

    $`P = \frac{x - a}{b - a}`$

    These plots below use the LOESS normalized log-likelihoods. I clipped the data so any point with a calculated proportional position above 1 was set to 1, and any point below 0 to 0.
    """)
    return


@app.cell(hide_code=True)
def _(plt, sns, spearmanr):
    def trunk_time_split(model_df, split_year=1990):
        for segment, group in model_df.groupby("Segment"):
            fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
            subsets = {
                f"≤ {split_year}": group[group["node_time"] <= split_year],
                f"> {split_year}":  group[group["node_time"] >  split_year]
            }

            for ax, (label, df) in zip(axes, subsets.items()):
                if df.empty:
                    ax.set_visible(False)
                    continue

                rho, pval = spearmanr(df["node_time"], df["trunk_position"])
                df = df.copy()
                df["time_rank"] = df["node_time"].rank()
                df["freq_rank"] = df["trunk_position"].rank()

                #lm = LinearRegression().fit(
                #    df[["time_rank"]], df["freq_rank"]
                #)

                sns.scatterplot(x="node_time", y="trunk_position", data=df, ax=ax)

                #x_line = np.linspace(df["node_time"].min(),
                #                     df["node_time"].max(), 100)
                #x_line_df = pd.DataFrame({"time_rank": pd.Series(x_line).rank()})

                #y_line_rank = lm.predict(x_line_df)

                #y_line = np.interp(
                #    y_line_rank,
                #    np.arange(1, len(df) + 1),
                #    np.sort(df["trunk_position"].values)
                #)

                #ax.plot(x_line, y_line, "--", label=f"ρ={rho:.2f}, p={pval:.2g}")
                ax.set_title(f"{segment} Trunk position over time {label}")
                ax.set_xlabel("Time")
                #ax.legend()

            axes[0].set_ylabel("trunk position")
            plt.tight_layout()
            plt.show()
    return (trunk_time_split,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Trunk position over time for 650M parameter model
    """)
    return


@app.cell(hide_code=True)
def _(df_650_NN, trunk_time_split):
    trunk_time_split(df_650_NN) 
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ####Trunk position over time for 3B parameter model
    """)
    return


@app.cell(hide_code=True)
def _(df_3B_NN, trunk_time_split):
    trunk_time_split(df_3B_NN) 
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    _For these plots I separated them between the testing and training datasets_

    One thing I notice is that for both HA and NA, in the training dataset, the relative position of the trunk drops off right at 1990. ESM scores these nodes higher than the maximum frequency of the tree. We don't see this same pattern with the test data, which is relatively scatttered.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.image(src="Images/HA_ESM_vs_Time.png")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Plotting ESM score over time we can see this pattern more clearly - slightly before and after 1990, terminal nodes are scored higher than the trunk.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Compare pre and post LOESS spearman correlation coefficient
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Calculate spearman correlation coefficient for each segment for 650M and 3B models
    """)
    return


@app.cell(hide_code=True)
def _(df_3B_FT_DF_with_loess, df_650_FT_DF_with_loess, pd, spearmanr):
    #calculate summary statistics for fine-tune models

    def summary_stats(model_df, base_name, time_frame):
      results = []

      for model, group in model_df.groupby('Model'):
        for segment, group in model_df.groupby('Segment'):

          df = model_df[model_df['Segment'] == segment]
          df = df[df['Model'] == model]

          if base_name == "PA":
            df = df[df['node'] != 'A/Viamao/LACENRS-974/2015']

          df_below_01 = df[df['max_frequency'] < 0.1]
          df_above_1 = df[df['max_frequency'] >= 0.99]

          spearman_corr, p_value = spearmanr(df['max_frequency'], df['log_likelihood'])

          results.append({
              "Model": model,
              "Segment": segment,
              "Spearman Correlation Coefficient between Max Frequency and LL": spearman_corr,
              "P-value": p_value,
              "Mean ESM LL below 0.1": df_below_01['log_likelihood'].mean(),
              "Mean ESM LL above 0.99": df_above_1['log_likelihood'].mean(),
              "Difference in LL ESM Means": df_above_1['log_likelihood'].mean() - df_below_01['log_likelihood'].mean(),
              "Time Frame": time_frame
          })

          results_df = pd.DataFrame(results)

          if(model == "Fine_Tune_3B" or model == "Fine_Tune_650M"):

              spearman_corr, p_value = spearmanr(df['max_frequency'], df['corrected_log_likelihood'])

              results.append({
                  "Model": f"LOESS_{model}",
                  "Segment": segment,
                  "Spearman Correlation Coefficient between Max Frequency and LL": spearman_corr,
                  "P-value": p_value,
                  "Mean ESM LL below 0.1": df_below_01['corrected_log_likelihood'].mean(),
                  "Mean ESM LL above 0.99": df_above_1['corrected_log_likelihood'].mean(),
                  "Difference in LL ESM Means": df_above_1['corrected_log_likelihood'].mean() - df_below_01['corrected_log_likelihood'].mean(),
                  "Time Frame": time_frame
              })

              results_df = pd.DataFrame(results)

      #print("____________________________")
      #print(f"Summary Statistics for {base_name} Model - {time_frame}")
      #print(results_df.groupby('Model')['Spearman Correlation Coefficient between Max Frequency and LL'].mean())

      #results_df.to_csv(f"Flu_Summary_Statistics/ESM_vs_Max_Freq_Summary_Fine_Tune_{base_name}_Statistics.csv", index=False)
      return results_df

    df_3B_FT_DF_Time_Above_1990 = df_3B_FT_DF_with_loess[df_3B_FT_DF_with_loess['time'] >= 1991]
    df_650_FT_DF_Time_Above_1990 = df_650_FT_DF_with_loess[df_650_FT_DF_with_loess['time'] >= 1991]
    df_3B_FT_DF_Time_Below_1990 = df_3B_FT_DF_with_loess[df_3B_FT_DF_with_loess['time'] <= 1990]
    df_650_FT_DF_Time_Below_1990 = df_650_FT_DF_with_loess[df_650_FT_DF_with_loess['time'] <= 1990]

    df_3B_FT_DF_Time_Above_1990_Results_DF = summary_stats(df_3B_FT_DF_Time_Above_1990, "3B", "Post 1990")
    df_650_FT_DF_Time_Above_1990_Results_DF = summary_stats(df_650_FT_DF_Time_Above_1990, "650M", "Post 1990")
    df_3B_FT_DF_Time_Below_1990_Results_DF = summary_stats(df_3B_FT_DF_Time_Below_1990, "3B", "Pre 1990")
    df_650_FT_DF_Time_Below_1990_Results_DF = summary_stats(df_650_FT_DF_Time_Below_1990, "650M", "Pre 1990")

    # Combine all results into a single DataFrame
    combined_results = pd.concat([df_3B_FT_DF_Time_Above_1990_Results_DF, df_650_FT_DF_Time_Above_1990_Results_DF, df_3B_FT_DF_Time_Below_1990_Results_DF, df_650_FT_DF_Time_Below_1990_Results_DF], ignore_index=True)
    return (
        df_3B_FT_DF_Time_Above_1990_Results_DF,
        df_3B_FT_DF_Time_Below_1990_Results_DF,
        df_650_FT_DF_Time_Above_1990_Results_DF,
        df_650_FT_DF_Time_Below_1990_Results_DF,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Plot spearman correlation coefficient for each model with and without LOESS
    """)
    return


@app.cell(hide_code=True)
def _(
    df_3B_FT_DF_Time_Above_1990_Results_DF,
    df_3B_FT_DF_Time_Below_1990_Results_DF,
    df_650_FT_DF_Time_Above_1990_Results_DF,
    df_650_FT_DF_Time_Below_1990_Results_DF,
    pd,
    plt,
    sns,
):
    def plot_spearman_barplot(ax, df, model_order, palette, title, xaxis=""):
        df['Model'] = pd.Categorical(df['Model'], categories=model_order, ordered=True)
        df = df.sort_values('Model')

        sns.barplot(
            data=df,
            x='Segment',
            y='Spearman Correlation Coefficient between Max Frequency and LL',
            hue='Model',
            hue_order=model_order,
            errorbar=None,
            palette=palette,
            ax=ax
        )
        ax.set_title(title)
        ax.set_xlabel(xaxis, weight='bold')
        ax.set_ylabel("Spearman CC (Max Freq. vs LL)", weight='bold')
        ax.legend(title="Model", frameon=False, loc='lower left')

    def combined_average_spearman_fine_tune_compare(df_3B, df_650M, df_3B_FT, df_650M_FT):
        model_order_3B = ['3B', 'Fine_Tune_3B', 'LOESS_Fine_Tune_3B']
        model_order_650M = ['650M', 'Fine_Tune_650M', 'LOESS_Fine_Tune_650M']

        palette_3B = {
            '3B': '#0a2463',
            'Fine_Tune_3B': '#f4d35e',
            'LOESS_Fine_Tune_3B': '#890304'
        }

        palette_650M = {
            '650M': '#0a2463',
            'Fine_Tune_650M': '#f4d35e',
            'LOESS_Fine_Tune_650M': '#890304',
        }

        fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharey=True)

        plot_spearman_barplot(axes[0, 0], df_3B, model_order_3B, palette_3B, "3B - Fine Tune vs LOESS (Post-1990)", xaxis="")
        plot_spearman_barplot(axes[0, 1], df_650M, model_order_650M, palette_650M, "650M - Fine Tune vs LOESS (Post-1990)", xaxis="")
        plot_spearman_barplot(axes[1, 0], df_3B_FT, model_order_3B, palette_3B, "3B - Fine Tune vs LOESS (Pre-1990)", xaxis="Segment")
        plot_spearman_barplot(axes[1, 1], df_650M_FT, model_order_650M, palette_650M, "650M - Fine Tune vs LOESS (Pre-1990)", xaxis="Segment")

        plt.tight_layout()
        plt.show()

    combined_average_spearman_fine_tune_compare(df_3B_FT_DF_Time_Above_1990_Results_DF, df_650_FT_DF_Time_Above_1990_Results_DF, df_3B_FT_DF_Time_Below_1990_Results_DF, df_650_FT_DF_Time_Below_1990_Results_DF)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    _These figure show Spearman CC between Maximum frequency and Log Likelyhood for both the 3B and 650M models before and after the training period of 1990. Shown here are the base model in blue, the fine tune model in yellow, and the fine tune model with LOESS in red._

    We see that both the base and fine tune models have a much higher spearman CC in the testing dataset (post 1990) than in the training dataset (pre 1990). However this trend is reversed after LOESS correction is applied where the LOESS corrected fine tune models have a much higher spearman CC for training data than for test data - where spearman CC is nearly zero.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Comparing maximum frequency to time
    """)
    return


@app.cell(hide_code=True)
def _(LinearRegression, df_650_FT_DF_with_loess, np, pd, plt, sns, spearmanr):
    def time_maxfreq_spear(model_df):

        for segment, group in model_df.groupby('Segment'):

            df = model_df[model_df['Segment'] == segment]

            #df = df[df["time"] > 1990]

            rho, pval = spearmanr(df["time"], 
                                 df["max_frequency"])
            print(f"Spearman ρ = {rho:.3f}, p = {pval:.3g}")

            df = df.copy()
            df["time_rank"] = df["time"].rank()
            df["freq_rank"] = df["max_frequency"].rank()

            X = df[["time_rank"]].values
            y = df["freq_rank"].values
            lm = LinearRegression().fit(X, y)

            sns.scatterplot(x="time", y="max_frequency", data=df)

            x_line = np.linspace(df["time"].min(), df["time"].max(), 100)
            x_line_rank = pd.Series(x_line).rank(method="first", pct=False).values
            y_line_rank = lm.predict(x_line_rank.reshape(-1, 1))
            y_line = np.percentile(df["max_frequency"], 100 * (y_line_rank - 1) / (len(df) - 1))

            plt.plot(x_line, y_line, color="red", linestyle="--",
                     label=f"Spearman fit (ρ={rho:.2f})")
            plt.legend()

            plt.title(f"{segment} time vs maximum frequency")

            plt.show()

    time_maxfreq_spear(df_650_FT_DF_with_loess)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    _These figures plot maximum frequency against time for each flu segment, each point is a node on the tree._

    These figures show us that over time there is a moderate decrease in maximum frequency scores over time, with the majority of points and the points with lowest maximum frequencies concentrated the furthest in time, at the bottom right of all plots.

    This helps explain why before LOESS correction there is an increase in spearman CC (between maximum frequency and log likelihood) when running ESM on the test dataset.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Recreating sliding window spearman CC plots
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Calculate spearman CC for each 20 year time window
    """)
    return


@app.cell(hide_code=True)
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
                (1970, 1990, "1980"),
                (1980, 2000, "1990"),
                (1990, 2010, "2000"),
                (2000, 2020, "2010"),
                (2010, None, "2020"),
            ]

            for start, end, label in time_ranges:
                if end is None:
                    df_segment_FT_label = df_segment_FT[df_segment_FT['time'] >= start]
                    #df_segment_BS_label = df_segment_BS[df_segment_BS['time'] >= start]
                    #df_segment_FT_LS_label = df_segment_FT_LS[df_segment_FT_LS['time'] >= start]
                else:
                    df_segment_FT_label = df_segment_FT[(df_segment_FT['time'] >= start) & (df_segment_FT['time'] <= end)]
                    #df_segment_BS_label = df_segment_BS[(df_segment_BS['time'] >= start) & (df_segment_BS['time'] <= end)]
                    #df_segment_FT_LS_label = df_segment_FT_LS[(df_segment_FT_LS['time'] >= start) & (df_segment_FT_LS['time'] <= end)]

                results.append(spearman_correlation_calculation(df_segment_FT_label, "max_frequency", "log_likelihood", f"Fine_Tune_{model}", segment, label))
                #results.append(spearman_correlation_calculation(df_segment_BS_label, "max_frequency", "log_likelihood", model, segment, label))
                results.append(spearman_correlation_calculation(df_segment_FT_label, "max_frequency", "corrected_log_likelihood", f"LOESS_Fine_Tune_{model}", segment, label))

        return pd.DataFrame(results)
    return (spearman_correlation,)


@app.cell(hide_code=True)
def _(df_3B_FT_DF_with_loess, df_650_FT_DF_with_loess, spearman_correlation):
    df_3B_FT_DF_Time_spearman = spearman_correlation(df_3B_FT_DF_with_loess, "3B")
    df_650_FT_DF_Time_spearman = spearman_correlation(df_650_FT_DF_with_loess, "650M")
    return df_3B_FT_DF_Time_spearman, df_650_FT_DF_Time_spearman


@app.cell(hide_code=True)
def _(df_3B_FT_DF_Time_spearman, df_650_FT_DF_Time_spearman):
    df_650_FT_DF_Time_spearman_Base = df_650_FT_DF_Time_spearman[df_650_FT_DF_Time_spearman['Model'] == '650M']
    df_650_FT_DF_Time_spearman_Fine_Tune = df_650_FT_DF_Time_spearman[df_650_FT_DF_Time_spearman['Model'] == 'Fine_Tune_650M']
    df_650_FT_LS_DF_Time_spearman_Fine_Tune = df_650_FT_DF_Time_spearman[df_650_FT_DF_Time_spearman['Model'] == 'LOESS_Fine_Tune_650M']

    df_3B_FT_DF_Time_spearman_Base = df_3B_FT_DF_Time_spearman[df_3B_FT_DF_Time_spearman['Model'] == '3B']
    df_3B_FT_DF_Time_spearman_Fine_Tune = df_3B_FT_DF_Time_spearman[df_3B_FT_DF_Time_spearman['Model'] == 'Fine_Tune_3B']
    df_3B_FT_LS_DF_Time_spearman_Fine_Tune = df_3B_FT_DF_Time_spearman[df_3B_FT_DF_Time_spearman['Model'] == 'LOESS_Fine_Tune_3B']
    return (
        df_3B_FT_DF_Time_spearman_Base,
        df_3B_FT_DF_Time_spearman_Fine_Tune,
        df_3B_FT_LS_DF_Time_spearman_Fine_Tune,
        df_650_FT_DF_Time_spearman_Base,
        df_650_FT_DF_Time_spearman_Fine_Tune,
        df_650_FT_LS_DF_Time_spearman_Fine_Tune,
    )


@app.function(hide_code=True)
def mean_spearman_segments(df, model_name):
    df_Fine_Tune_Summary = df.groupby('Time_Range', as_index=False)['Spearman_Correlation'].mean()
    df_Fine_Tune_Summary["Model"] = model_name 

    return df_Fine_Tune_Summary


@app.cell(hide_code=True)
def _(
    df_3B_FT_DF_Time_spearman_Base,
    df_3B_FT_DF_Time_spearman_Fine_Tune,
    df_3B_FT_LS_DF_Time_spearman_Fine_Tune,
    df_650_FT_DF_Time_spearman_Base,
    df_650_FT_DF_Time_spearman_Fine_Tune,
    df_650_FT_LS_DF_Time_spearman_Fine_Tune,
    pd,
):
    df_3B_FT_DF_Time_spearman_Fine_Tune_Summary = mean_spearman_segments(df_3B_FT_DF_Time_spearman_Fine_Tune, "Fine Tune - 3B Model")
    df_650_FT_DF_Time_spearman_Fine_Tune_Summary = mean_spearman_segments(df_650_FT_DF_Time_spearman_Fine_Tune, "Fine Tune - 650M Model")

    df_3B_DF_Time_Spearman_Summary = mean_spearman_segments(df_3B_FT_DF_Time_spearman_Base, "Base - 3B Model")
    df_650_DF_Time_Spearman_Summary = mean_spearman_segments(df_650_FT_DF_Time_spearman_Base, "Base - 650M Model")

    combined_spearman_summary = pd.concat([
        df_3B_FT_DF_Time_spearman_Fine_Tune_Summary,
        df_650_FT_DF_Time_spearman_Fine_Tune_Summary,
        df_3B_DF_Time_Spearman_Summary,
        df_650_DF_Time_Spearman_Summary,
        df_650_FT_LS_DF_Time_spearman_Fine_Tune,
        df_3B_FT_LS_DF_Time_spearman_Fine_Tune
    ], ignore_index=True)
    return (combined_spearman_summary,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Create time window figures
    """)
    return


@app.cell(hide_code=True)
def _(np, plt, sns):
    def create_spearman_summary_plot(df):
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

        ax.set_title("Spearman CC Summary All Models")
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

        plt.tight_layout()
        plt.show()
    return (create_spearman_summary_plot,)


@app.cell(hide_code=True)
def _(combined_spearman_summary, create_spearman_summary_plot):
    create_spearman_summary_plot(combined_spearman_summary)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    _This figure shows spearman CC between the 650M and 3B parameters models before and after LOESS correction, spearman CC is calculated 10 years before and after each timepoint. IE 2000 would be spearman CC between 1990-2010_

    With LOESS correction spearman CC drops off after training dataset (1990) and remains relatively flat.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Import larger flu tree

    Here we will compare the larger flu tree with ~1000 more nodes to the base flu tree I was working with, I'll compare a training window of 1990 to a training window of 2005 as well as I did in my presentation.
    """)
    return


@app.cell(hide_code=True)
def _(pd):
    combined_LG_SM_1990_2005 = pd.read_csv('Dataframes/LG_SM_1990_2005.csv', keep_default_na=False)
    return (combined_LG_SM_1990_2005,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Apply LOESS correction to larger tree
    """)
    return


@app.cell(hide_code=True)
def _(combined_LG_SM_1990_2005):
    LG_1990 = combined_LG_SM_1990_2005[(combined_LG_SM_1990_2005["Model_training_time"] == 1990) & (combined_LG_SM_1990_2005["tree"] == "h3n2-Large")]
    LG_2005 = combined_LG_SM_1990_2005[(combined_LG_SM_1990_2005["Model_training_time"] == 2005) & (combined_LG_SM_1990_2005["tree"] == "h3n2-Large")]
    SM_1990 = combined_LG_SM_1990_2005[(combined_LG_SM_1990_2005["Model_training_time"] == 1990) & (combined_LG_SM_1990_2005["tree"] == "h3n2")]
    SM_2005 = combined_LG_SM_1990_2005[(combined_LG_SM_1990_2005["Model_training_time"] == 2005) & (combined_LG_SM_1990_2005["tree"] == "h3n2")]
    return LG_1990, LG_2005, SM_1990, SM_2005


@app.cell(hide_code=True)
def _(LG_1990, apply_loess_to_finetune_models):
    LG_1990_with_loess = apply_loess_to_finetune_models(LG_1990)
    return (LG_1990_with_loess,)


@app.cell(hide_code=True)
def _(LG_2005, apply_loess_to_finetune_models):
    LG_2005_with_loess = apply_loess_to_finetune_models(LG_2005)
    return (LG_2005_with_loess,)


@app.cell(hide_code=True)
def _(SM_1990, apply_loess_to_finetune_models):
    SM_1990_with_loess = apply_loess_to_finetune_models(SM_1990)
    return (SM_1990_with_loess,)


@app.cell(hide_code=True)
def _(SM_2005, apply_loess_to_finetune_models):
    SM_2005_with_loess = apply_loess_to_finetune_models(SM_2005)
    return (SM_2005_with_loess,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Calculate spearman CC for large and small trees for each testing period
    """)
    return


@app.cell(hide_code=True)
def _(
    LG_1990_with_loess,
    LG_2005_with_loess,
    SM_1990_with_loess,
    SM_2005_with_loess,
    spearman_correlation,
):
    LG_1990_with_loess_spearman = spearman_correlation(LG_1990_with_loess, "650M")
    LG_2005_with_loess_spearman = spearman_correlation(LG_2005_with_loess, "650M")
    SM_1990_with_loess_spearman = spearman_correlation(SM_1990_with_loess, "650M")
    SM_2005_with_loess_spearman = spearman_correlation(SM_2005_with_loess, "650M")
    return (
        LG_1990_with_loess_spearman,
        LG_2005_with_loess_spearman,
        SM_1990_with_loess_spearman,
        SM_2005_with_loess_spearman,
    )


@app.cell(hide_code=True)
def _(
    LG_1990_with_loess_spearman,
    LG_2005_with_loess_spearman,
    SM_1990_with_loess_spearman,
    SM_2005_with_loess_spearman,
):
    LG_1990_spearman_FT = LG_1990_with_loess_spearman[LG_1990_with_loess_spearman['Model'] == 'Fine_Tune_650M']
    LG_1990_spearman_FT_LS = LG_1990_with_loess_spearman[LG_1990_with_loess_spearman['Model'] == 'LOESS_Fine_Tune_650M']

    LG_2005_spearman_FT = LG_2005_with_loess_spearman[LG_2005_with_loess_spearman['Model'] == 'Fine_Tune_650M']
    LG_2005_spearman_FT_LS = LG_2005_with_loess_spearman[LG_2005_with_loess_spearman['Model'] == 'LOESS_Fine_Tune_650M']

    SM_1990_spearman_FT = SM_1990_with_loess_spearman[SM_1990_with_loess_spearman['Model'] == 'Fine_Tune_650M']
    SM_1990_spearman_FT_LS = SM_1990_with_loess_spearman[SM_1990_with_loess_spearman['Model'] == 'LOESS_Fine_Tune_650M']

    SM_2005_spearman_FT = SM_2005_with_loess_spearman[SM_2005_with_loess_spearman['Model'] == 'Fine_Tune_650M']
    SM_2005_spearman_FT_LS = SM_2005_with_loess_spearman[SM_2005_with_loess_spearman['Model'] == 'LOESS_Fine_Tune_650M']
    return (
        LG_1990_spearman_FT,
        LG_1990_spearman_FT_LS,
        LG_2005_spearman_FT,
        LG_2005_spearman_FT_LS,
        SM_1990_spearman_FT,
        SM_1990_spearman_FT_LS,
        SM_2005_spearman_FT,
        SM_2005_spearman_FT_LS,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Generate Large tree LOESS Comparison for 1990 and 2005
    """)
    return


@app.cell(hide_code=True)
def _(
    LG_1990_spearman_FT,
    LG_1990_spearman_FT_LS,
    LG_2005_spearman_FT,
    LG_2005_spearman_FT_LS,
    SM_1990_spearman_FT,
    SM_1990_spearman_FT_LS,
    SM_2005_spearman_FT,
    SM_2005_spearman_FT_LS,
    pd,
):
    LG_1990_spearman_FT_Summary = mean_spearman_segments(LG_1990_spearman_FT, "1990 Large Tree")
    LG_1990_spearman_FT_LS_Summary = mean_spearman_segments(LG_1990_spearman_FT_LS, "1990 Large Tree with LOESS")

    LG_2005_spearman_FT_Summary = mean_spearman_segments(LG_2005_spearman_FT, "2005 Large Tree")
    LG_2005_spearman_FT_LS_Summary = mean_spearman_segments(LG_2005_spearman_FT_LS, "2005 Large Tree with LOESS")

    SM_1990_spearman_FT_Summary = mean_spearman_segments(SM_1990_spearman_FT, "1990 Small Tree")
    SM_1990_spearman_FT_LS_Summary = mean_spearman_segments(SM_1990_spearman_FT_LS, "1990 Small Tree with LOESS")

    SM_2005_spearman_FT_Summary = mean_spearman_segments(SM_2005_spearman_FT, "2005 Small Tree")
    SM_2005_spearman_FT_LS_Summary = mean_spearman_segments(SM_2005_spearman_FT_LS, "2005 Small Tree with LOESS")

    LG_combined_spearman_summary = pd.concat([
        LG_1990_spearman_FT_Summary,
        LG_1990_spearman_FT_LS_Summary,
        LG_2005_spearman_FT_Summary,
        LG_2005_spearman_FT_LS_Summary,
    ], ignore_index=True)

    SM_combined_spearman_summary = pd.concat([
        SM_1990_spearman_FT_Summary,
        SM_1990_spearman_FT_LS_Summary,
        SM_2005_spearman_FT_Summary,
        SM_2005_spearman_FT_LS_Summary
    ], ignore_index=True)
    return LG_combined_spearman_summary, SM_combined_spearman_summary


@app.cell(hide_code=True)
def _(LG_combined_spearman_summary, create_spearman_summary_plot):
    create_spearman_summary_plot(LG_combined_spearman_summary)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    _This figure is a sliding window comparision of the larger flu tree, which added an additional ~1000 samples, with models trained up to 1990 and 2005 before and after LOESS correction._

    For the larger tree before LOESS correction we see a rise of spearman CC after 1990, with spearman CC peaking at 2010. After LOESS we no longer see this spike at 2010.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Generate Small tree LOESS Comparison for 1990 and 2005
    """)
    return


@app.cell(hide_code=True)
def _(SM_combined_spearman_summary, create_spearman_summary_plot):
    create_spearman_summary_plot(SM_combined_spearman_summary)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    _This figure shows a sliding window comparision of spearman CC's between training the model up 1990 and up to 2005. This figure features the datasets with and without LOESS correction. Small tree is the base tree that was used for earlier parts of this project and notebook._

    With LOESS correction we no longer see the large increase in spearman CC at 2010 and instead a constant decrease in spearman CC. The dataset trained up to 2005 has a higher spearman CC than the dataset trained up to 1990 across all points.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Calculate time series cross Validation
    """)
    return


@app.cell(hide_code=True)
def _(pd):
    df_650_FT_DF_Time_series_Validation = pd.read_csv('Dataframes/df_650_FT_DF_Time_Series_Validation.csv', keep_default_na=False)
    return (df_650_FT_DF_Time_series_Validation,)


@app.cell(hide_code=True)
def _(apply_loess_to_finetune_models, df_650_FT_DF_Time_series_Validation, pd):
    time_vals = df_650_FT_DF_Time_series_Validation["Model_training_time"].unique()
    smoothed_parts = []
    for t in time_vals:
        subset = df_650_FT_DF_Time_series_Validation[
            df_650_FT_DF_Time_series_Validation["Model_training_time"] == t
        ].copy()
        smoothed = apply_loess_to_finetune_models(subset)
        smoothed_parts.append(smoothed)

    df_650_FT_DF_Time_series_Validation_LOESS = pd.concat(smoothed_parts, ignore_index=True)
    return (df_650_FT_DF_Time_series_Validation_LOESS,)


@app.cell(hide_code=True)
def _(pd, spearmanr):
    def calculate_time_series_cross_df(df, ll_column: str = 'log_likelihood'):
        time_bins = {
            "1990": [(1990, 2000), (2000, 2010), (2010, 2020)],
            "1995": [(1995, 2005), (2005, 2015), (2015, None)],
            "2000": [(2000, 2010), (2010, 2020), (2020, None)],
            "2005": [(2005, 2015), (2015, None)],
            "2010": [(2010, 2020), (2020, None)]
        }

        results = []

        for start_year, ranges in time_bins.items():
            for idx, (start, end) in enumerate(ranges, 1):
                if end is None:
                    df_bin = df[
                    (df['time'] >= start) &
                    (df['Model_training_time'] == int(start_year))
                    ]
                else:
                    df_bin = df[
                        (df['time'] >= start) &
                        (df['time'] <= end) &
                        (df['Model_training_time'] == int(start_year))
                    ]


                corr, _ = spearmanr(df_bin['max_frequency'], df_bin[ll_column])


                results.append({
                    "start_year": start_year,
                    "bin_index": idx,
                    "range": f"{start}-{end if end else '2025'}",
                    "spearman_corr": corr
                })

        spearman_df = pd.DataFrame(results)
        return spearman_df
    return (calculate_time_series_cross_df,)


@app.cell(hide_code=True)
def _(
    calculate_time_series_cross_df,
    df_650_FT_DF_Time_series_Validation_LOESS,
):
    spearman_df = calculate_time_series_cross_df(df_650_FT_DF_Time_series_Validation_LOESS)
    spearman_df_LOESS = calculate_time_series_cross_df(df_650_FT_DF_Time_series_Validation_LOESS, "corrected_log_likelihood")
    return spearman_df, spearman_df_LOESS


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Generate time series cross validation for fine tune models no LOESS
    """)
    return


@app.cell(hide_code=True)
def _(plt, sns):
    def create_time_series_plot(spearman_df):
        unique_years = spearman_df['start_year'].unique()
        n_years = len(unique_years)

        fig, axes = plt.subplots(1, n_years, figsize=(5 * n_years, 5), sharey=True)

        if n_years == 1:
            axes = [axes]

        for ax, year in zip(axes, unique_years):
            subset = spearman_df[spearman_df['start_year'] == year]
            sns.barplot(data=subset, x='range', y='spearman_corr', hue='range', palette="viridis", errorbar=None, ax=ax)
            ax.set_title(f"Model Trained up to: {int(year)}")
            ax.set_xlabel("Time Range")
            ax.set_ylabel("Spearman Correlation Coefficient")

        plt.tight_layout()

        plt.show()
    return (create_time_series_plot,)


@app.cell(hide_code=True)
def _(create_time_series_plot, spearman_df):
    create_time_series_plot(spearman_df)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    _This figure is a time series cross validation for the smaller (base tree) with no LOESS correction applied. EX for the first box, a model was trained up to 1990, then spearman CC was calculated for small windows after the training period._

    We would expect spearman CC to drop off over time past the training period, where in this figure it looks more sporadic and it is difficult to discern a pattern.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Generate time series cross validation for fine tune models with LOESS correction
    """)
    return


@app.cell(hide_code=True)
def _(create_time_series_plot, spearman_df_LOESS):
    create_time_series_plot(spearman_df_LOESS)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    _This figure takes the same dataset from the previous time series plot and adds LOESS correction before binning and calculating spearman CC._

    Here we see spearman CC drop off over time the futher the time window is from the training dataset.
    """)
    return


if __name__ == "__main__":
    app.run()
