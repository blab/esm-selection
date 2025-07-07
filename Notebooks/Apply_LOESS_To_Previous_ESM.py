import marimo

__generated_with = "0.14.9"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""# Applying LOESS to previous ESM work""")
    return


@app.cell
def _(mo):
    mo.md(r"""In this notebook I will be redoing some of my previous findings with LOESS correction applied""")
    return


@app.cell
def _(mo):
    mo.md(r"""## Setup""")
    return


@app.cell
def _(mo):
    mo.md(r"""### Install Libraries""")
    return


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    import seaborn as sns
    from scipy.stats import spearmanr
    import colorsys
    import matplotlib.cm as cm
    from matplotlib.ticker import ScalarFormatter
    from matplotlib.ticker import FormatStrFormatter
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    return ScalarFormatter, cm, colorsys, mo, np, pd, plt, sns, spearmanr


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Patch to fix LOESS package 

    Areas where data similarity is high there is a division by zero error
    """
    )
    return


@app.cell
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
        """
        Rotates points counter-clockwise by an angle ANG in degrees.
        Michele cappellari, Paranal, 10 November 2013

        """
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
                        mad = np.finfo(float).tiny
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


@app.cell
def _(mo):
    mo.md(r"""## Regenerating ESM vs Time Plots""")
    return


@app.cell
def _(mo):
    mo.md(r"""Load dataframes trained up to 1990""")
    return


@app.cell
def _(pd):
    df_650_FT_DF = pd.read_csv('Dataframes/650M_Fine_Tune_Up_To_1990.csv', keep_default_na=False)
    df_3B_FT_DF = pd.read_csv('Dataframes/3B_Fine_Tune_Up_To_1990.csv', keep_default_na=False)
    return df_3B_FT_DF, df_650_FT_DF


@app.cell
def _(mo):
    mo.md(r"""Rewrite LOESS application to work with any dataframe and store as a function""")
    return


@app.cell
def _(loess_1d, pd):
    def apply_loess_to_segment(df, x_col="time", y_col="log_likelihood", degree=2, frac=0.15):
        x = df[x_col].values
        y = df[y_col].values

        xout, yout, wout = loess_1d(
            x=x,
            y=y,
            xnew=x,
            degree=degree,
            frac=frac
        )

        df.loc[:, f"{y_col}_LOESS"] = yout
        df.loc[:, "loess_weight"] = wout

        return df


    def apply_loess_to_finetune_models(df, x_col="time", y_col="log_likelihood", degree=2, frac=0.15):
        fine_tune_df = df[df['Model'].str.startswith("Fine_Tune")].copy()

        smoothed_dfs = []

        for (segment, model), group_df in fine_tune_df.groupby(["Segment", "Model"]):
            group_df = group_df.copy()
            group_df = apply_loess_to_segment(group_df, x_col=x_col, y_col=y_col, degree=degree, frac=frac)
            smoothed_dfs.append(group_df[["Segment", "Model", x_col, f"{y_col}_LOESS", "loess_weight"]])

        smoothed_df = pd.concat(smoothed_dfs, ignore_index=True)

        df_merged = df.merge(
            smoothed_df,
            on=["Segment", "Model", x_col],
            how="left"
        )

        df_merged["corrected_log_likelihood"] = df_merged["log_likelihood"] - df_merged["log_likelihood_LOESS"]

        return df_merged

    return (apply_loess_to_finetune_models,)


@app.cell
def _(apply_loess_to_finetune_models, df_3B_FT_DF, df_650_FT_DF):
    df_650_FT_DF_with_loess = apply_loess_to_finetune_models(df_650_FT_DF)
    df_3B_FT_DF_with_loess = apply_loess_to_finetune_models(df_3B_FT_DF)
    return df_3B_FT_DF_with_loess, df_650_FT_DF_with_loess


@app.cell
def _(ScalarFormatter, cm, colorsys, np, plt, sns):
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set(style='ticks', palette='Set2')

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

        high_freq_df = df[df["max_frequency"] >= 1].sort_values("time")
        ax.plot(
            high_freq_df["time"],
            high_freq_df[ll_col],
            linestyle='-',
            color='black',
            linewidth=3,
            alpha=0.6,
            label='Max Freq ≥ 0.99',
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
        plt.show()
    return (esm_vs_time_3x8_grid,)


@app.cell
def _(df_650_FT_DF_with_loess, esm_vs_time_3x8_grid):
    esm_vs_time_3x8_grid(df_650_FT_DF_with_loess, "650M")
    return


@app.cell
def _(df_3B_FT_DF_with_loess, esm_vs_time_3x8_grid):
    esm_vs_time_3x8_grid(df_3B_FT_DF_with_loess, "3B")
    return


@app.cell
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

      print("____________________________")
      print(f"Summary Statistics for {base_name} Model - {time_frame}")
      print(results_df.groupby('Model')['Spearman Correlation Coefficient between Max Frequency and LL'].mean())

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


@app.cell
def _(
    df_3B_FT_DF_Time_Above_1990_Results_DF,
    df_3B_FT_DF_Time_Below_1990_Results_DF,
    df_650_FT_DF_Time_Above_1990_Results_DF,
    df_650_FT_DF_Time_Below_1990_Results_DF,
    pd,
    plt,
    sns,
):
    # Combine all result plots into one 4x4 figure, easier to view

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


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
