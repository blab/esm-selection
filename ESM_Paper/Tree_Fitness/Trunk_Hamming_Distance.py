# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "altair==6.0.0",
#     "duckdb==1.4.3",
#     "marimo",
#     "matplotlib==3.10.7",
#     "nbformat==5.10.4",
#     "numpy==2.3.4",
#     "openai==2.13.0",
#     "pandas==2.3.3",
#     "polars[pyarrow]==1.36.1",
#     "pyarrow==22.0.0",
#     "pytest==9.0.2",
#     "python-lsp-ruff==2.3.0",
#     "python-lsp-server==1.14.0",
#     "ruff==0.14.9",
#     "scikit-learn==1.8.0",
#     "scipy==1.16.3",
#     "seaborn==0.13.2",
#     "sqlglot==28.4.1",
#     "vegafusion==2.0.3",
#     "vl-convert-python==1.8.0",
#     "websockets==15.0.1",
# ]
# ///

import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from scipy.stats import spearmanr
    import colorsys
    import matplotlib.cm as cm
    from matplotlib.ticker import ScalarFormatter
    from sklearn.linear_model import LinearRegression
    import re
    return (
        LinearRegression,
        ScalarFormatter,
        cm,
        colorsys,
        mo,
        mpl,
        np,
        pd,
        pl,
        plt,
        spearmanr,
    )


@app.cell
def _():
    import pyarrow as pa
    pa.__version__
    return


@app.cell
def _():
    import json

    # Load the h3n2/ha.json file for node time mapping
    with open(
        "/Users/cavendan/Desktop/esm-selection/ESM_Paper/Tree_Fitness/h3n2/ha.json",
        "r",
    ) as f:
        tree_data = json.load(f)


    def extract_node_times(node, node_times_dict=None):
        """Recursively extract node names and their num_date values from the tree."""
        if node_times_dict is None:
            node_times_dict = {}

        # Extract time from current node
        if "node_attrs" in node and "num_date" in node["node_attrs"]:
            time_value = node["node_attrs"]["num_date"]["value"]
            node_times_dict[node["name"]] = time_value

        # Recursively process children
        if "children" in node:
            for child in node["children"]:
                extract_node_times(child, node_times_dict)

        return node_times_dict


    # Extract all node times from the tree
    node_times = extract_node_times(tree_data["tree"])
    print(f"Loaded {len(node_times)} node-time mappings from h3n2/ha.json")

    # Show some example NODE_ entries
    node_examples = {
        name: time
        for name, time in list(node_times.items())[:10]
        if name.startswith("NODE_")
    }
    print("Example NODE_ time mappings:", node_examples)
    return json, node_times


@app.cell
def _(pl):
    ha_data = pl.read_csv(
        "/Users/cavendan/Desktop/esm-selection/ESM_Paper/Tree_Fitness/next_tree~h3n2/epochs~1/learning_rate~5e-05/model~esm2_t33_650M_UR50D/time~2000/H3N2_Dataset_1965_Full/Max_Freq_Fasta_LL_Fine_Tune_ha.csv"
    )
    ha_data.head()
    return (ha_data,)


@app.cell
def _(node_times, np):
    def get_node_time(node_name):
        """Get time value for a node from the phylogenetic tree mapping."""
        if node_name in node_times:
            return node_times[node_name]

        return None


    # LOESS implementation from reference notebook
    class polyfit1d:
        def __init__(self, x, y, degree, weights):
            sqw = np.sqrt(weights)
            a = x[:, None] ** np.arange(degree + 1)
            self.degree = degree
            self.coeff = np.linalg.lstsq(a * sqw[:, None], y * sqw, rcond=None)[0]
            self.yfit = a @ self.coeff

        def eval(self, x):
            a = x ** np.arange(self.degree + 1)
            yout = a @ self.coeff
            return yout


    def biweight_sigma(y, zero=False):
        y = np.ravel(y)
        if zero:
            d = y
        else:
            d = y - np.median(y)

        mad = np.median(np.abs(d))
        u2 = (d / (9.0 * mad)) ** 2  # c = 9
        good = u2 < 1.0
        u1 = 1.0 - u2[good]
        num = y.size * ((d[good] * u1**2) ** 2).sum()
        den = (u1 * (1.0 - 5.0 * u2[good])).sum()
        sigma = np.sqrt(num / (den * (den - 1.0)))  # see note in above reference

        return sigma


    def rotate_points(x, y, ang):
        theta = np.radians(ang)
        xNew = x * np.cos(theta) - y * np.sin(theta)
        yNew = x * np.sin(theta) + y * np.cos(theta)
        return xNew, yNew


    def loess_1d(
        x, y, xnew=None, degree=1, frac=0.5, npoints=None, rotate=False, sigy=None
    ):
        if frac == 0:
            return y, np.ones_like(y)

        assert x.size == y.size, "Input vectors (X, Y) must have the same size"

        if npoints is None:
            npoints = int(np.ceil(frac * x.size))

        if rotate:
            assert xnew is None, "`rotate` not supported with `xnew`"
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
            dist_weights = (
                1 - (dist[w] / dist[w[-1]]) ** 3
            ) ** 3  # tricube function distance weights
            yfit = polyfit1d(x[w], y[w], degree, dist_weights).yfit

            # Robust fit from Sec.2 of Cleveland (1979)
            bad = None
            for p in range(10):  # do at most 10 iterations
                if sigy is None:  # Errors are unknown
                    aerr = np.abs(yfit - y[w])  # Note ABS()
                    mad = np.median(aerr)  # Characteristic scale

                    if mad == 0:
                        mad = np.maximum(mad, 1e-10)
                    uu = (
                        aerr / (6 * mad)
                    ) ** 2  # For a Gaussian: sigma=1.4826*MAD
                else:  # Errors are assumed known
                    uu = ((yfit - y[w]) / (4 * sigy[w])) ** 2  # 4*sig ~ 6*mad

                uu = uu.clip(0, 1)
                biweights = (1 - uu) ** 2
                tot_weights = dist_weights * biweights
                poly = polyfit1d(x[w], y[w], degree, tot_weights)
                yfit = poly.yfit
                badOld = bad
                bad = biweights < 0.34  # 99% confidence outliers
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
    return get_node_time, loess_1d


@app.cell
def _(get_node_time, ha_data):
    # Convert to pandas for compatibility with reference notebook functions
    ha_df = ha_data.to_pandas()

    # Extract time information from phylogenetic tree - no arbitrary filters
    ha_df["time"] = ha_df["node"].apply(get_node_time)
    ha_df = ha_df.dropna(subset=["time"])
    ha_df = ha_df[ha_df["time"] >= 1968]

    # Filter out sequences with gaps (- characters)
    initial_count = len(ha_df)
    ha_df = ha_df[~ha_df["sequence"].str.contains("-", na=False)]
    final_count = len(ha_df)

    print(f"Filtered out {initial_count - final_count} sequences with gaps")
    print(f"Total sequences after filtering: {final_count}")

    ha_df.head()
    return (ha_df,)


@app.cell
def _(loess_1d, pd):
    # LOESS application function from reference notebook
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

        df = df.copy()
        df[f"{y_col}_LOESS"] = y_smoothed
        df["loess_weight"] = w_smoothed

        return df
    return (apply_loess_to_segment,)


@app.cell
def _(apply_loess_to_segment, ha_df):
    # Apply LOESS to HA data
    ha_df_with_loess = apply_loess_to_segment(ha_df)
    ha_df_with_loess["corrected_log_likelihood"] = (
        ha_df_with_loess["log_likelihood"]
        - ha_df_with_loess["log_likelihood_LOESS"]
    )

    # Add trunk labeling - nodes with max frequency = 1.0 (100%) and contain "NODE_"
    ha_df_with_loess["is_trunk"] = (ha_df_with_loess["max_frequency"] >= 0.99) & (
        ha_df_with_loess["node"].str.contains("NODE_")
    )

    trunk_count = ha_df_with_loess["is_trunk"].sum()
    print(f"Identified {trunk_count} trunk nodes")

    ha_df_with_loess
    return (ha_df_with_loess,)


@app.cell
def _(ha_df_with_loess):
    # Filter sequences to keep only those with the most common length
    sequence_lengths = ha_df_with_loess["sequence"].str.len()
    most_common_length = sequence_lengths.mode().iloc[0]

    print(f"Sequence Length Distribution:")
    length_counts = sequence_lengths.value_counts().sort_index()
    print(length_counts)
    print(f"\nMost common sequence length: {most_common_length}")

    # Find sequences that don't match the most common length
    non_standard_mask = sequence_lengths != most_common_length
    non_standard_seqs = ha_df_with_loess[non_standard_mask]

    print(f"\nSEQUENCE LENGTH FILTERING:")
    print(f"Mode (most common) sequence length: {most_common_length}")
    print(f"Sequences with non-standard lengths: {len(non_standard_seqs)}")

    if len(non_standard_seqs) > 0:
        print(f"\nNODE NAMES WITH NON-STANDARD SEQUENCE LENGTHS:")
        print(f"Mode (most common) sequence length: {most_common_length}")
        print(f"Total sequences to exclude: {len(non_standard_seqs)}")
        print()

        for count, (idx, row) in enumerate(non_standard_seqs.iterrows(), 1):
            seq_len = len(row["sequence"])
            trunk_status = "TRUNK" if row["is_trunk"] else "BRANCH"
            length_diff = seq_len - most_common_length
            diff_str = f"+{length_diff}" if length_diff > 0 else str(length_diff)
            print(
                f"  {row['node']} -> Length: {seq_len} ({diff_str} from mode), Time: {row['time']:.1f}, Type: {trunk_status}"
            )

            if (
                len(non_standard_seqs) <= 5
            ):  # Show sequence for small number of exclusions
                print(f"    Sequence: {row['sequence']}")
            elif count >= 10:  # Limit output for very large numbers
                remaining = len(non_standard_seqs) - count
                if remaining > 0:
                    print(f"  ... and {remaining} more excluded nodes")
                break
    else:
        print(
            f"✓ All sequences match the mode length ({most_common_length}). No exclusions needed."
        )

    # Create filtered dataframe with only standard-length sequences
    ha_df_length_filtered = ha_df_with_loess[
        sequence_lengths == most_common_length
    ].copy()

    print(f"\nFiltering Results:")
    print(f"Original sequences: {len(ha_df_with_loess)}")
    print(f"Filtered sequences: {len(ha_df_length_filtered)}")
    print(
        f"Excluded sequences: {len(ha_df_with_loess) - len(ha_df_length_filtered)}"
    )

    # Verify all sequences now have the same length
    filtered_lengths = ha_df_length_filtered["sequence"].str.len()
    print(
        f"All remaining sequences have length {filtered_lengths.iloc[0]}: {filtered_lengths.nunique() == 1}"
    )
    return (ha_df_length_filtered,)


@app.cell
def _(ScalarFormatter, cm, colorsys, plt):
    # Plotting functions from reference notebook
    def darken_color(rgb, factor=0.7):
        h, l, s = colorsys.rgb_to_hls(*rgb)
        r, g, b = colorsys.hls_to_rgb(h, max(0, l * factor), s)
        return (r, g, b, 1.0)


    def plot_esm_score(ax, df, title, Fine_Tune=False, LOESS=False):
        if LOESS == False:
            ll_col = "log_likelihood"
        else:
            ll_col = "corrected_log_likelihood"

        norm = plt.Normalize(df[ll_col].min(), df[ll_col].max())
        cmap = plt.get_cmap("viridis")
        colors = cmap(norm(df[ll_col]))
        edgecolors = [darken_color(c[:3], factor=0.7) for c in colors]

        # Add grid behind data points
        ax.grid(True, color="lightgray", linestyle="-", linewidth=0.75, zorder=0)

        ax.scatter(
            df["time"],
            df[ll_col],
            c=colors,
            edgecolors=edgecolors,
            linewidths=0.5,
            alpha=0.7,
            zorder=3,
        )

        # Connect trunk nodes (max frequency >= 1.0 and contains NODE_)
        trunk_df = df[
            (df["max_frequency"] >= 1.0)
            & (df["node"].str.contains("NODE_", na=False))
        ].sort_values("time")
        if len(trunk_df) > 0:
            ax.plot(
                trunk_df["time"],
                trunk_df[ll_col],
                linestyle="-",
                color="black",
                linewidth=3,
                alpha=0.6,
                label="Trunk (Max Freq ≥ 1 & NODE_)",
                zorder=4,
            )

        ax.yaxis.offsetText.set_visible(False)

        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = plt.colorbar(
            sm, ax=ax, orientation="vertical", pad=0.02, extend="both"
        )

        cbar.ax.yaxis.offsetText.set_visible(False)

        ax.set_title(title, fontsize=10)

        if Fine_Tune:
            ax.axvline(1990, color="gray", linestyle="--", linewidth=1.5, zorder=1)

        ax.set_ylabel("ESM Score", fontsize=8)
        ax.spines[["right", "top"]].set_visible(False)
        ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        ax.ticklabel_format(style="plain", axis="x")
        ax.set_xlim(1965, 2030)

        y_min, y_max = df[ll_col].min(), df[ll_col].max()
        pad = (y_max - y_min) * 0.05 if y_max != y_min else 1.0
        ax.set_ylim(y_min - pad, y_max + pad)

        return ax
    return (plot_esm_score,)


@app.cell
def _(ha_df_length_filtered, mpl, np, plot_esm_score, plt):
    # Set light theme from reference notebook
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.edgecolor": "black",
            "axes.labelcolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "text.color": "black",
        }
    )

    # Create ESM vs Time plots using reference notebook approach
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Raw ESM scores
    plot_esm_score(axes[0], ha_df_length_filtered, "HA • Raw ESM Scores vs Time")

    # LOESS corrected scores
    plot_esm_score(
        axes[1],
        ha_df_length_filtered,
        "HA • LOESS Corrected ESM Scores",
        LOESS=True,
    )

    # Set x-axis labels with extended range to include 2025+ data
    _years = np.arange(1967, 2026, 20)
    for _ax in axes:
        _ax.set_xticks(_years)
        _ax.set_xticklabels(_years, rotation=0, ha="right", fontsize=10)
        _ax.tick_params(axis="x", which="major", labelbottom=True)
        _ax.set_xlabel("Year", fontsize=8)

    plt.tight_layout(h_pad=2, w_pad=1)
    plt.gca()
    return


@app.cell
def _(ha_df_length_filtered):
    # Hamming Distance Analysis using filtered sequences
    print("=== HAMMING DISTANCE ANALYSIS ===")


    def hamming_distance(seq1, seq2):
        """Calculate Hamming distance between two equal-length sequences."""
        return sum(c1 != c2 for c1, c2 in zip(seq1, seq2))


    def find_nearest_trunk_node(target_time, trunk_df):
        """Find trunk node closest in time to target_time."""
        time_diffs = abs(trunk_df["time"] - target_time)
        nearest_idx = time_diffs.idxmin()
        return trunk_df.loc[nearest_idx]


    # Separate trunk and branch nodes using existing is_trunk column
    trunk_nodes = ha_df_length_filtered[ha_df_length_filtered["is_trunk"]].copy()
    branch_nodes = ha_df_length_filtered[~ha_df_length_filtered["is_trunk"]].copy()

    print(f"Working with filtered data:")
    print(f"Total nodes: {len(ha_df_length_filtered)}")
    print(f"Trunk nodes: {len(trunk_nodes)}")
    print(f"Branch nodes: {len(branch_nodes)}")
    print(
        f"All sequences have length: {len(ha_df_length_filtered['sequence'].iloc[0])}"
    )

    # Calculate hamming distances (no length checking needed since all sequences are filtered)
    hamming_distances = []
    for i, node_row in ha_df_length_filtered.iterrows():
        if node_row["is_trunk"]:
            hamming_distances.append(0)  # Trunk nodes get 0
        else:
            nearest_trunk = find_nearest_trunk_node(node_row["time"], trunk_nodes)
            distance = hamming_distance(
                node_row["sequence"], nearest_trunk["sequence"]
            )
            hamming_distances.append(distance)

    # Add the column to the dataframe
    ha_df_final = ha_df_length_filtered.copy()
    ha_df_final["hamming_distance_to_trunk"] = hamming_distances

    # Summary statistics
    branch_distances = [
        d
        for d, is_trunk in zip(hamming_distances, ha_df_final["is_trunk"])
        if not is_trunk
    ]
    if branch_distances:
        print(f"\nHamming Distance Statistics for Branch Nodes:")
        print(f"Mean: {sum(branch_distances) / len(branch_distances):.2f}")
        print(f"Min: {min(branch_distances)}")
        print(f"Max: {max(branch_distances)}")
        print(f"Total branch nodes with distances: {len(branch_distances)}")

    # Show some examples
    print(f"\nExample branch nodes and their nearest trunk matches:")
    branch_examples = ha_df_final[~ha_df_final["is_trunk"]].head(3)
    for j, example_row in branch_examples.iterrows():
        nearest_trunk = find_nearest_trunk_node(example_row["time"], trunk_nodes)
        print(
            f"Branch: {example_row['node']} (time: {example_row['time']:.1f}) -> Nearest trunk: {nearest_trunk['node']} (time: {nearest_trunk['time']:.1f}), Distance: {example_row['hamming_distance_to_trunk']}"
        )
    return (ha_df_final,)


@app.cell
def _(ha_df_final):
    ha_df_final
    return


@app.cell
def _(ha_df_final, np, plt):
    # 2x2 Correlation analysis with training/testing split
    import scipy.stats as stats

    # Filter out any rows with missing data
    correlation_data = ha_df_final.dropna(
        subset=["max_frequency", "log_likelihood", "corrected_log_likelihood"]
    )

    # Split data by time: Training (< 2000) and Testing (>= 2000)
    training_data = correlation_data[correlation_data["time"] < 2000]
    testing_data = correlation_data[correlation_data["time"] >= 2000]


    def plot_correlation_subplot(
        ax, x_data, y_data, title, color="steelblue", colormap_data=None
    ):
        """Helper function to create correlation subplot."""
        if len(x_data) == 0 or len(y_data) == 0:
            ax.text(
                0.5,
                0.5,
                "No data",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
            ax.set_title(title, fontsize=9)
            return None, None

        # Create scatter plot
        if colormap_data is not None:
            # Color by hamming distance
            norm = plt.Normalize(colormap_data.min(), colormap_data.max())
            cmap = plt.get_cmap("viridis")
            colors = cmap(norm(colormap_data))
            scatter = ax.scatter(x_data, y_data, c=colors, alpha=0.6, s=15)
        else:
            scatter = ax.scatter(x_data, y_data, alpha=0.6, s=15, color=color)

        # Add trend line
        if len(x_data) > 1:
            z = np.polyfit(x_data, y_data, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(x_data.min(), x_data.max(), 100)
            ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=1)

        # Calculate correlations
        try:
            spearman_r, spearman_p = stats.spearmanr(x_data, y_data)
            pearson_r, pearson_p = stats.pearsonr(x_data, y_data)
        except:
            spearman_r = spearman_p = pearson_r = pearson_p = np.nan

        # Add correlation text
        corr_text = f"ρ={spearman_r:.2f}\nr={pearson_r:.2f}"
        ax.text(
            0.05,
            0.95,
            corr_text,
            transform=ax.transAxes,
            fontsize=7,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        ax.set_title(title, fontsize=9)
        ax.grid(True, alpha=0.3)

        return spearman_r, pearson_r


    # Create 2x2 figure
    grid_fig, grid_axes = plt.subplots(2, 2, figsize=(12, 8))

    correlation_results = {}

    # Top Left: Training - Raw ESM
    spearman_r, pearson_r = plot_correlation_subplot(
        grid_axes[0, 0],
        training_data["max_frequency"],
        training_data["log_likelihood"],
        f"Training Raw ESM - 650M (Pre 2000)",
        color="steelblue",
        colormap_data=None,
    )
    correlation_results["train_raw"] = (spearman_r, pearson_r)

    # Top Right: Training - LOESS
    spearman_r, pearson_r = plot_correlation_subplot(
        grid_axes[0, 1],
        training_data["max_frequency"],
        training_data["corrected_log_likelihood"],
        f"Training LOESS - 650M (Pre 2000)",
        color="steelblue",
        colormap_data=None,
    )
    correlation_results["train_loess"] = (spearman_r, pearson_r)

    # Bottom Left: Testing - Raw ESM
    spearman_r, pearson_r = plot_correlation_subplot(
        grid_axes[1, 0],
        testing_data["max_frequency"],
        testing_data["log_likelihood"],
        f"Testing Raw ESM - 650M (Post 2000)",
        color="darkgreen",
        colormap_data=None,
    )
    correlation_results["test_raw"] = (spearman_r, pearson_r)

    # Bottom Right: Testing - LOESS
    spearman_r, pearson_r = plot_correlation_subplot(
        grid_axes[1, 1],
        testing_data["max_frequency"],
        testing_data["corrected_log_likelihood"],
        f"Testing LOESS - 650M (Post 2000)",
        color="darkgreen",
        colormap_data=None,
    )
    correlation_results["test_loess"] = (spearman_r, pearson_r)

    # Set axis labels
    grid_axes[0, 0].set_ylabel("ESM Score", fontsize=12)
    grid_axes[1, 0].set_ylabel("ESM Score", fontsize=12)
    grid_axes[1, 0].set_xlabel("Maximum Frequency", fontsize=12)
    grid_axes[1, 1].set_xlabel("Maximum Frequency", fontsize=12)

    plt.tight_layout(pad=1.5)
    plt.show()

    # Create second 2x2 figure comparing ESM scores with Hamming distance
    hamming_fig, hamming_axes = plt.subplots(2, 2, figsize=(12, 8))

    hamming_correlation_results = {}

    # Top Left: Training - Raw ESM vs Hamming Distance
    spearman_r, pearson_r = plot_correlation_subplot(
        hamming_axes[0, 0],
        training_data["hamming_distance_to_trunk"],
        training_data["log_likelihood"],
        f"Training Raw ESM vs Hamming - 650M (Pre 2000)",
        color="steelblue",
        colormap_data=None,
    )
    hamming_correlation_results["train_raw"] = (spearman_r, pearson_r)

    # Top Right: Training - LOESS vs Hamming Distance
    spearman_r, pearson_r = plot_correlation_subplot(
        hamming_axes[0, 1],
        training_data["hamming_distance_to_trunk"],
        training_data["corrected_log_likelihood"],
        f"Training LOESS vs Hamming - 650M (Pre 2000)",
        color="steelblue",
        colormap_data=None,
    )
    hamming_correlation_results["train_loess"] = (spearman_r, pearson_r)

    # Bottom Left: Testing - Raw ESM vs Hamming Distance
    spearman_r, pearson_r = plot_correlation_subplot(
        hamming_axes[1, 0],
        testing_data["hamming_distance_to_trunk"],
        testing_data["log_likelihood"],
        f"Testing Raw ESM vs Hamming - 650M (Post 2000)",
        color="darkgreen",
        colormap_data=None,
    )
    hamming_correlation_results["test_raw"] = (spearman_r, pearson_r)

    # Bottom Right: Testing - LOESS vs Hamming Distance
    spearman_r, pearson_r = plot_correlation_subplot(
        hamming_axes[1, 1],
        testing_data["hamming_distance_to_trunk"],
        testing_data["corrected_log_likelihood"],
        f"Testing LOESS vs Hamming - 650M (Post 2000)",
        color="darkgreen",
        colormap_data=None,
    )
    hamming_correlation_results["test_loess"] = (spearman_r, pearson_r)

    # Set axis labels for hamming distance plots
    hamming_axes[0, 0].set_ylabel("ESM Score", fontsize=12)
    hamming_axes[1, 0].set_ylabel("ESM Score", fontsize=12)
    hamming_axes[1, 0].set_xlabel("Hamming Distance to Trunk", fontsize=12)
    hamming_axes[1, 1].set_xlabel("Hamming Distance to Trunk", fontsize=12)

    plt.tight_layout(pad=1.5)
    plt.show()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Comprehensive HA Analysis from Apply_LOESS Notebook
    Reproducing key analyses from Apply_LOESS_To_Previous_ESM.py focused on HA segment only.
    """)
    return


@app.cell
def _(get_node_time, pl):
    # Load 650M and 3B data for both Base and Fine-Tune models

    # Base directory paths
    base_dir = "/Users/cavendan/Desktop/esm-selection/ESM_Paper/Tree_Fitness/next_tree~h3n2/epochs~1/learning_rate~5e-05"

    # Load 650M Base (ESM without fine-tuning)
    ha_650M_base = pl.read_csv(
        f"{base_dir}/model~esm2_t33_650M_UR50D/time~2000/ESM_1965_Full/Max_Freq_Fasta_LL_Fine_Tune_ha.csv"
    ).to_pandas()
    ha_650M_base["time"] = ha_650M_base["node"].apply(get_node_time)
    ha_650M_base = ha_650M_base.dropna(subset=["time"])
    ha_650M_base = ha_650M_base[ha_650M_base["log_likelihood"] >= -250]
    ha_650M_base["Model"] = "650M"
    ha_650M_base["Segment"] = "HA"

    # Load 650M Fine-Tune
    ha_650M_ft = pl.read_csv(
        f"{base_dir}/model~esm2_t33_650M_UR50D/time~2000/H3N2_Dataset_1965_Full/Max_Freq_Fasta_LL_Fine_Tune_ha.csv"
    ).to_pandas()
    ha_650M_ft["time"] = ha_650M_ft["node"].apply(get_node_time)
    ha_650M_ft = ha_650M_ft.dropna(subset=["time"])
    ha_650M_ft = ha_650M_ft[ha_650M_ft["log_likelihood"] >= -250]
    ha_650M_ft["Model"] = "Fine_Tune_650M"
    ha_650M_ft["Segment"] = "HA"

    # Load 3B Base (ESM without fine-tuning)
    ha_3B_base = pl.read_csv(
        f"{base_dir}/model~esm2_t36_3B_UR50D/time~2000/ESM_1965_Full/Max_Freq_Fasta_LL_Fine_Tune_ha.csv"
    ).to_pandas()
    ha_3B_base["time"] = ha_3B_base["node"].apply(get_node_time)
    ha_3B_base = ha_3B_base.dropna(subset=["time"])
    ha_3B_base = ha_3B_base[ha_3B_base["log_likelihood"] >= -250]
    ha_3B_base["Model"] = "3B"
    ha_3B_base["Segment"] = "HA"

    # Load 3B Fine-Tune
    ha_3B_ft = pl.read_csv(
        f"{base_dir}/model~esm2_t36_3B_UR50D/time~2000/H3N2_Dataset_1965_Full/Max_Freq_Fasta_LL_Fine_Tune_ha.csv"
    ).to_pandas()
    ha_3B_ft["time"] = ha_3B_ft["node"].apply(get_node_time)
    ha_3B_ft = ha_3B_ft.dropna(subset=["time"])
    ha_3B_ft = ha_3B_ft[ha_3B_ft["log_likelihood"] >= -250]
    ha_3B_ft["Model"] = "Fine_Tune_3B"
    ha_3B_ft["Segment"] = "HA"

    print(f"Loaded datasets (filtered log_likelihood >= -250):")
    print(f"  650M Base: {len(ha_650M_base)} rows")
    print(f"  650M Fine-Tune: {len(ha_650M_ft)} rows")
    print(f"  3B Base: {len(ha_3B_base)} rows")
    print(f"  3B Fine-Tune: {len(ha_3B_ft)} rows")
    return ha_3B_base, ha_3B_ft, ha_650M_base, ha_650M_ft


@app.cell
def _(apply_loess_to_segment, ha_3B_base, ha_3B_ft, ha_650M_base, ha_650M_ft):
    # Apply LOESS to all datasets
    ha_650M_base_loess = apply_loess_to_segment(ha_650M_base.copy())
    ha_650M_base_loess["corrected_log_likelihood"] = (
        ha_650M_base_loess["log_likelihood"] - ha_650M_base_loess["log_likelihood_LOESS"]
    )

    ha_650M_ft_loess = apply_loess_to_segment(ha_650M_ft.copy())
    ha_650M_ft_loess["corrected_log_likelihood"] = (
        ha_650M_ft_loess["log_likelihood"] - ha_650M_ft_loess["log_likelihood_LOESS"]
    )

    ha_3B_base_loess = apply_loess_to_segment(ha_3B_base.copy())
    ha_3B_base_loess["corrected_log_likelihood"] = (
        ha_3B_base_loess["log_likelihood"] - ha_3B_base_loess["log_likelihood_LOESS"]
    )

    ha_3B_ft_loess = apply_loess_to_segment(ha_3B_ft.copy())
    ha_3B_ft_loess["corrected_log_likelihood"] = (
        ha_3B_ft_loess["log_likelihood"] - ha_3B_ft_loess["log_likelihood_LOESS"]
    )

    print("LOESS correction applied to all datasets")
    return (
        ha_3B_base_loess,
        ha_3B_ft_loess,
        ha_650M_base_loess,
        ha_650M_ft_loess,
    )


@app.cell
def _(ha_650M_base_loess, ha_650M_ft_loess, mpl, np, plot_esm_score, plt):
    # ESM vs Time 3-column figure for 650M
    mpl.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.edgecolor": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "text.color": "black",
    })

    fig_650M, axes_650M = plt.subplots(1, 3, figsize=(18, 6))

    # Base model
    plot_esm_score(axes_650M[0], ha_650M_base_loess, "HA • 650M Base", Fine_Tune=False, LOESS=False)

    # Fine-Tune model
    plot_esm_score(axes_650M[1], ha_650M_ft_loess, "HA • 650M Fine-Tune", Fine_Tune=True, LOESS=False)

    # LOESS corrected
    plot_esm_score(axes_650M[2], ha_650M_ft_loess, "HA • 650M LOESS", Fine_Tune=True, LOESS=True)

    # Set x-axis labels
    _years_650M = np.arange(1965, 2026, 20)
    for _ax in axes_650M:
        _ax.set_xticks(_years_650M)
        _ax.set_xticklabels(_years_650M, rotation=0, ha="right", fontsize=10)
        _ax.tick_params(axis="x", which="major", labelbottom=True)
        _ax.set_xlabel("Year", fontsize=8)

    plt.tight_layout(h_pad=2, w_pad=1)
    plt.gca()
    return


@app.cell
def _(ha_3B_base_loess, ha_3B_ft_loess, np, plot_esm_score, plt):
    # ESM vs Time 3-column figure for 3B
    fig_3B, axes_3B = plt.subplots(1, 3, figsize=(18, 6))

    # Base model
    plot_esm_score(axes_3B[0], ha_3B_base_loess, "HA • 3B Base", Fine_Tune=False, LOESS=False)

    # Fine-Tune model
    plot_esm_score(axes_3B[1], ha_3B_ft_loess, "HA • 3B Fine-Tune", Fine_Tune=True, LOESS=False)

    # LOESS corrected
    plot_esm_score(axes_3B[2], ha_3B_ft_loess, "HA • 3B LOESS", Fine_Tune=True, LOESS=True)

    # Set x-axis labels
    _years_3B = np.arange(1965, 2026, 20)
    for _ax in axes_3B:
        _ax.set_xticks(_years_3B)
        _ax.set_xticklabels(_years_3B, rotation=0, ha="right", fontsize=10)
        _ax.tick_params(axis="x", which="major", labelbottom=True)
        _ax.set_xlabel("Year", fontsize=8)

    plt.tight_layout(h_pad=2, w_pad=1)
    plt.gca()
    return


@app.cell
def _(
    ScalarFormatter,
    cm,
    colorsys,
    ha_3B_ft_loess,
    ha_650M_ft_loess,
    np,
    plt,
):
    # 2-column LOESS comparison (650M vs 3B)
    def darken_color_2col(rgb, factor=0.7):
        h, l, s = colorsys.rgb_to_hls(*rgb)
        r, g, b = colorsys.hls_to_rgb(h, max(0, l * factor), s)
        return (r, g, b, 1.0)

    def plot_loess_finetune_2col(ax, df, title):
        ll_col = "corrected_log_likelihood"
        norm = plt.Normalize(df[ll_col].min(), df[ll_col].max())
        cmap = plt.get_cmap("viridis")
        colors = cmap(norm(df[ll_col]))
        edgecolors = [darken_color_2col(c[:3], factor=0.7) for c in colors]

        ax.scatter(
            df["time"],
            df[ll_col],
            c=colors,
            edgecolors=edgecolors,
            linewidths=0.5,
            alpha=0.7,
            zorder=1,
        )

        high_freq = df[df["max_frequency"] >= 1].sort_values("time")
        ax.plot(
            high_freq["time"],
            high_freq[ll_col],
            linestyle="-",
            color="black",
            linewidth=3,
            alpha=0.6,
            label="Max Freq ≥ 1",
            zorder=2,
        )

        ax.yaxis.offsetText.set_visible(False)
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = plt.colorbar(
            sm, ax=ax, orientation="vertical", pad=0.02, extend="both"
        )
        cbar.ax.yaxis.offsetText.set_visible(False)

        ax.set_title(title, fontsize=10)
        ax.axvline(2000, color="gray", linestyle="--", linewidth=1.5, zorder=1)
        ax.set_ylabel("ESM Score", fontsize=8)
        ax.grid(True, color="lightgray", linestyle="-", linewidth=0.75)
        ax.spines[["right", "top"]].set_visible(False)
        ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        ax.ticklabel_format(style="plain", axis="x")
        ax.set_xlim(1965, 2025)
        y_min, y_max = df[ll_col].min(), df[ll_col].max()
        pad = (y_max - y_min) * 0.05 if y_max != y_min else 1.0
        ax.set_ylim(y_min - pad, y_max + pad)
        return ax

    fig_2col, axes_2col = plt.subplots(1, 2, figsize=(14, 6))

    plot_loess_finetune_2col(axes_2col[0], ha_650M_ft_loess, "HA • Fine_Tune_650M")
    plot_loess_finetune_2col(axes_2col[1], ha_3B_ft_loess, "HA • Fine_Tune_3B")

    _years_2col = np.arange(1960, 2021, 20)
    for _ax in axes_2col:
        _ax.set_xticks(_years_2col)
        _ax.set_xticklabels(_years_2col, rotation=0, ha="right", fontsize=10)
        _ax.set_xlabel("Year", fontsize=8)

    plt.tight_layout(h_pad=2, w_pad=1)
    plt.gca()
    return


@app.cell
def _(
    ha_3B_base_loess,
    ha_3B_ft_loess,
    ha_650M_base_loess,
    ha_650M_ft_loess,
    pd,
    spearmanr,
):
    # Calculate correlation statistics
    def summary_stats_ha(model_df, base_name, time_frame):
        results = []

        for model in model_df["Model"].unique():
            df = model_df[model_df["Model"] == model].copy()

            df_below_01 = df[df["max_frequency"] < 0.1]
            df_above_1 = df[df["max_frequency"] >= 0.99]

            spearman_corr, p_value = spearmanr(df["max_frequency"], df["log_likelihood"])

            results.append(
                {
                    "Model": model,
                    "Segment": "HA",
                    "Spearman Correlation Coefficient between Max Frequency and LL": spearman_corr,
                    "P-value": p_value,
                    "Mean ESM LL below 0.1": df_below_01["log_likelihood"].mean(),
                    "Mean ESM LL above 0.99": df_above_1["log_likelihood"].mean(),
                    "Difference in LL ESM Means": df_above_1["log_likelihood"].mean()
                    - df_below_01["log_likelihood"].mean(),
                    "Time Frame": time_frame,
                }
            )

            if model in ["Fine_Tune_3B", "Fine_Tune_650M"]:
                spearman_corr, p_value = spearmanr(
                    df["max_frequency"], df["corrected_log_likelihood"]
                )

                results.append(
                    {
                        "Model": f"LOESS_{model}",
                        "Segment": "HA",
                        "Spearman Correlation Coefficient between Max Frequency and LL": spearman_corr,
                        "P-value": p_value,
                        "Mean ESM LL below 0.1": df_below_01[
                            "corrected_log_likelihood"
                        ].mean(),
                        "Mean ESM LL above 0.99": df_above_1[
                            "corrected_log_likelihood"
                        ].mean(),
                        "Difference in LL ESM Means": df_above_1[
                            "corrected_log_likelihood"
                        ].mean()
                        - df_below_01["corrected_log_likelihood"].mean(),
                        "Time Frame": time_frame,
                    }
                )

        return pd.DataFrame(results)

    # Combine all datasets
    ha_all = pd.concat(
        [ha_650M_base_loess, ha_650M_ft_loess, ha_3B_base_loess, ha_3B_ft_loess],
        ignore_index=True,
    )

    # Split by time
    ha_all_above_2000 = ha_all[ha_all["time"] >= 2001]
    ha_all_below_2000 = ha_all[ha_all["time"] <= 2000]

    # Calculate stats
    ha_3B_above_2000_results = summary_stats_ha(
        ha_all_above_2000[ha_all_above_2000["Model"].str.contains("3B")], "3B", "Post 2000"
    )
    ha_650M_above_2000_results = summary_stats_ha(
        ha_all_above_2000[ha_all_above_2000["Model"].str.contains("650M")],
        "650M",
        "Post 2000",
    )
    ha_3B_below_2000_results = summary_stats_ha(
        ha_all_below_2000[ha_all_below_2000["Model"].str.contains("3B")], "3B", "Pre 2000"
    )
    ha_650M_below_2000_results = summary_stats_ha(
        ha_all_below_2000[ha_all_below_2000["Model"].str.contains("650M")], "650M", "Pre 2000"
    )

    combined_results_ha = pd.concat(
        [
            ha_3B_above_2000_results,
            ha_650M_above_2000_results,
            ha_3B_below_2000_results,
            ha_650M_below_2000_results,
        ],
        ignore_index=True,
    )
    return (
        combined_results_ha,
        ha_3B_below_2000_results,
        ha_650M_above_2000_results,
        ha_650M_below_2000_results,
        ha_all,
    )


@app.cell
def _(
    combined_results_ha,
    ha_3B_below_2000_results,
    ha_650M_above_2000_results,
    ha_650M_below_2000_results,
    pd,
    plt,
):
    # Spearman CC bar chart (2x2 grid)
    def plot_spearman_barplot_ha(ax, df, model_order, palette, title, xaxis=""):
        df_plot = df.copy()
        df_plot["Model"] = pd.Categorical(
            df_plot["Model"], categories=model_order, ordered=True
        )
        df_plot = df_plot.sort_values("Model")

        import seaborn as sns

        sns.barplot(
            data=df_plot,
            x="Segment",
            y="Spearman Correlation Coefficient between Max Frequency and LL",
            hue="Model",
            hue_order=model_order,
            errorbar=None,
            palette=palette,
            ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel(xaxis, weight="bold")
        ax.set_ylabel("Spearman CC (Max Freq. vs LL)", weight="bold")
        ax.legend(title="Model", frameon=False, loc="lower left")

    model_order_3B = ["3B", "Fine_Tune_3B", "LOESS_Fine_Tune_3B"]
    model_order_650M = ["650M", "Fine_Tune_650M", "LOESS_Fine_Tune_650M"]

    palette_3B = {
        "3B": "#0a2463",
        "Fine_Tune_3B": "#f4d35e",
        "LOESS_Fine_Tune_3B": "#890304",
    }

    palette_650M = {
        "650M": "#0a2463",
        "Fine_Tune_650M": "#f4d35e",
        "LOESS_Fine_Tune_650M": "#890304",
    }

    fig_corr, axes_corr = plt.subplots(2, 2, figsize=(10, 10), sharey=True)

    # Get the subsets for each plot
    df_3B_post = combined_results_ha[
        (combined_results_ha["Time Frame"] == "Post 2000")
        & (combined_results_ha["Model"].str.contains("3B"))
    ]
    df_650M_post = ha_650M_above_2000_results
    df_3B_pre = ha_3B_below_2000_results
    df_650M_pre = ha_650M_below_2000_results

    plot_spearman_barplot_ha(
        axes_corr[0, 0],
        df_3B_post,
        model_order_3B,
        palette_3B,
        "3B - Fine Tune vs LOESS (Post-2000)",
        xaxis="",
    )
    plot_spearman_barplot_ha(
        axes_corr[0, 1],
        df_650M_post,
        model_order_650M,
        palette_650M,
        "650M - Fine Tune vs LOESS (Post-2000)",
        xaxis="",
    )
    plot_spearman_barplot_ha(
        axes_corr[1, 0],
        df_3B_pre,
        model_order_3B,
        palette_3B,
        "3B - Fine Tune vs LOESS (Pre-2000)",
        xaxis="Segment",
    )
    plot_spearman_barplot_ha(
        axes_corr[1, 1],
        df_650M_pre,
        model_order_650M,
        palette_650M,
        "650M - Fine Tune vs LOESS (Pre-2000)",
        xaxis="Segment",
    )

    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(LinearRegression, ha_all, np, pd, plt, spearmanr):
    # Time vs Max Frequency scatter
    import seaborn as _sns_time

    ha_all_for_time = ha_all[ha_all["Model"] == "Fine_Tune_650M"].copy()

    rho, pval = spearmanr(ha_all_for_time["time"], ha_all_for_time["max_frequency"])
    print(f"Spearman ρ = {rho:.3f}, p = {pval:.3g}")

    ha_all_for_time["time_rank"] = ha_all_for_time["time"].rank()
    ha_all_for_time["freq_rank"] = ha_all_for_time["max_frequency"].rank()

    _X = ha_all_for_time[["time_rank"]].values
    _y_time = ha_all_for_time["freq_rank"].values
    lm = LinearRegression().fit(_X, _y_time)

    fig_time_freq = plt.figure(figsize=(8, 6))
    _sns_time.scatterplot(x="time", y="max_frequency", data=ha_all_for_time)

    x_line = np.linspace(
        ha_all_for_time["time"].min(), ha_all_for_time["time"].max(), 100
    )
    x_line_rank = pd.Series(x_line).rank(method="first", pct=False).values
    y_line_rank = lm.predict(x_line_rank.reshape(-1, 1))
    y_line = np.percentile(
        ha_all_for_time["max_frequency"],
        100 * (y_line_rank - 1) / (len(ha_all_for_time) - 1),
    )

    plt.plot(
        x_line,
        y_line,
        color="red",
        linestyle="--",
        label=f"Spearman fit (ρ={rho:.2f})",
    )
    plt.legend()
    plt.title("HA time vs maximum frequency")
    plt.gca()
    return


@app.cell
def _(ha_all, pd, spearmanr):
    # Calculate sliding window correlations
    def spearman_correlation_ha(df):
        results = []

        df_650M = df[df["Model"] == "Fine_Tune_650M"].copy()
        df_3B = df[df["Model"] == "Fine_Tune_3B"].copy()

        time_ranges = [
            (1970, 1990, "1980"),
            (1980, 2000, "1990"),
            (1990, 2010, "2000"),
            (2000, 2020, "2010"),
            (2010, None, "2020"),
        ]

        for start, end, label in time_ranges:
            if end is None:
                df_650M_label = df_650M[df_650M["time"] >= start]
                df_3B_label = df_3B[df_3B["time"] >= start]
            else:
                df_650M_label = df_650M[
                    (df_650M["time"] >= start) & (df_650M["time"] <= end)
                ]
                df_3B_label = df_3B[(df_3B["time"] >= start) & (df_3B["time"] <= end)]

            # 650M correlations
            if len(df_650M_label) > 0:
                spearman_corr, p_value = spearmanr(
                    df_650M_label["max_frequency"], df_650M_label["log_likelihood"]
                )
                results.append(
                    {
                        "Model": "Fine_Tune_650M",
                        "Segment": "HA",
                        "Time_Range": label,
                        "Spearman_Correlation": spearman_corr,
                        "P_Value": p_value,
                    }
                )

                spearman_corr, p_value = spearmanr(
                    df_650M_label["max_frequency"],
                    df_650M_label["corrected_log_likelihood"],
                )
                results.append(
                    {
                        "Model": "LOESS_Fine_Tune_650M",
                        "Segment": "HA",
                        "Time_Range": label,
                        "Spearman_Correlation": spearman_corr,
                        "P_Value": p_value,
                    }
                )

            # 3B correlations
            if len(df_3B_label) > 0:
                spearman_corr, p_value = spearmanr(
                    df_3B_label["max_frequency"], df_3B_label["log_likelihood"]
                )
                results.append(
                    {
                        "Model": "Fine_Tune_3B",
                        "Segment": "HA",
                        "Time_Range": label,
                        "Spearman_Correlation": spearman_corr,
                        "P_Value": p_value,
                    }
                )

                spearman_corr, p_value = spearmanr(
                    df_3B_label["max_frequency"],
                    df_3B_label["corrected_log_likelihood"],
                )
                results.append(
                    {
                        "Model": "LOESS_Fine_Tune_3B",
                        "Segment": "HA",
                        "Time_Range": label,
                        "Spearman_Correlation": spearman_corr,
                        "P_Value": p_value,
                    }
                )

        return pd.DataFrame(results)

    ha_spearman_time = spearman_correlation_ha(ha_all)
    return (ha_spearman_time,)


@app.cell
def _(ha_spearman_time, np, plt):
    # Plot sliding window results
    import seaborn as _sns_sliding

    _sns_sliding.set_style("whitegrid")
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    _sns_sliding.set_theme(style="ticks", rc=custom_params)

    fig_sliding, ax_sliding = plt.subplots(figsize=(10, 5))

    ax_sliding = _sns_sliding.lineplot(
        data=ha_spearman_time,
        x="Time_Range",
        y="Spearman_Correlation",
        hue="Model",
        marker="o",
        legend=False,
        zorder=1,
        ax=ax_sliding,
        errorbar=None,
    )

    ax_sliding.set_title("Spearman CC Summary HA Segment")
    ax_sliding.set_xlabel("Time Range")
    ax_sliding.set_ylabel("Spearman CC")

    _label_positions = []
    for _line, _model in zip(ax_sliding.lines, ha_spearman_time["Model"].unique()):
        _y = _line.get_ydata()[-1]
        _x = _line.get_xdata()[-1]

        if not np.isfinite(_y) or not np.isfinite(_x):
            continue

        while any(abs(_y - pos) < 0.025 for pos in _label_positions):
            _y += 0.007

        _label_positions.append(_y)

        ax_sliding.annotate(
            _model,
            xy=(_x, _y),
            xytext=(5, 0),
            textcoords="offset points",
            color=_line.get_color(),
            fontsize=12,
            weight="bold",
            ha="left",
            va="center",
            zorder=2,
        )

    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Internal Node Analysis (≥10 Descendants)
    Filtering to internal nodes with at least 10 descendant terminals to focus on more stable phylogenetic positions.
    """)
    return


@app.function
def count_descendant_terminals(tree_file, json):
    """
    Count the number of descendant terminal nodes for each node in the tree.

    Args:
        tree_file: Path to the tree JSON file
        json: json module

    Returns:
        dict: Mapping of node_id -> number of descendant terminals
    """
    with open(tree_file, "r") as f:
        data = json.load(f)

    tree_root = data.get("tree", data)

    def count_terminals(node):
        """Recursively count terminal descendants."""
        children = node.get("children", [])

        if not children:
            # Leaf node - return 1
            return {node.get("name"): 1}

        # Internal node - sum counts from all descendants
        counts = {}
        total = 0
        for child in children:
            child_counts = count_terminals(child)
            counts.update(child_counts)
            # Only add the immediate child's terminal count (not all subtree counts)
            total += child_counts[child.get("name")]

        # Add this node's total count
        counts[node.get("name")] = total
        return counts

    return count_terminals(tree_root)


@app.cell
def _(json):
    # Count descendant terminals for all nodes in the HA tree
    ha_descendant_counts = count_descendant_terminals(
        "/Users/cavendan/Desktop/esm-selection/ESM_Paper/Tree_Fitness/h3n2/ha.json",
        json
    )

    print(f"Total nodes in tree: {len(ha_descendant_counts)}")

    # Show some example counts for internal nodes
    internal_examples = {name: count for name, count in list(ha_descendant_counts.items())[:5] if name.startswith("NODE_")}
    print(f"Example internal node descendant counts: {internal_examples}")
    return (ha_descendant_counts,)


@app.cell
def _(
    ha_3B_base_loess,
    ha_3B_ft_loess,
    ha_650M_base_loess,
    ha_650M_ft_loess,
    ha_descendant_counts,
):
    # Filter each dataset to internal nodes with ≥10 descendants

    # 650M Base
    ha_650M_base_loess_internal = ha_650M_base_loess[
        ha_650M_base_loess['node'].str.contains("NODE_", na=False)
    ].copy()
    ha_650M_base_loess_internal['descendant_count'] = ha_650M_base_loess_internal['node'].map(ha_descendant_counts)
    ha_650M_base_loess_internal = ha_650M_base_loess_internal[
        ha_650M_base_loess_internal['descendant_count'] >= 10
    ].copy()

    # 650M Fine-Tune
    ha_650M_ft_loess_internal = ha_650M_ft_loess[
        ha_650M_ft_loess['node'].str.contains("NODE_", na=False)
    ].copy()
    ha_650M_ft_loess_internal['descendant_count'] = ha_650M_ft_loess_internal['node'].map(ha_descendant_counts)
    ha_650M_ft_loess_internal = ha_650M_ft_loess_internal[
        ha_650M_ft_loess_internal['descendant_count'] >= 10
    ].copy()

    # 3B Base
    ha_3B_base_loess_internal = ha_3B_base_loess[
        ha_3B_base_loess['node'].str.contains("NODE_", na=False)
    ].copy()
    ha_3B_base_loess_internal['descendant_count'] = ha_3B_base_loess_internal['node'].map(ha_descendant_counts)
    ha_3B_base_loess_internal = ha_3B_base_loess_internal[
        ha_3B_base_loess_internal['descendant_count'] >= 10
    ].copy()

    # 3B Fine-Tune
    ha_3B_ft_loess_internal = ha_3B_ft_loess[
        ha_3B_ft_loess['node'].str.contains("NODE_", na=False)
    ].copy()
    ha_3B_ft_loess_internal['descendant_count'] = ha_3B_ft_loess_internal['node'].map(ha_descendant_counts)
    ha_3B_ft_loess_internal = ha_3B_ft_loess_internal[
        ha_3B_ft_loess_internal['descendant_count'] >= 10
    ].copy()

    print("Internal nodes (≥10 descendants) filtered datasets:")
    print(f"  650M Base: {len(ha_650M_base_loess_internal)} nodes (from {len(ha_650M_base_loess)})")
    print(f"  650M Fine-Tune: {len(ha_650M_ft_loess_internal)} nodes (from {len(ha_650M_ft_loess)})")
    print(f"  3B Base: {len(ha_3B_base_loess_internal)} nodes (from {len(ha_3B_base_loess)})")
    print(f"  3B Fine-Tune: {len(ha_3B_ft_loess_internal)} nodes (from {len(ha_3B_ft_loess)})")
    return (
        ha_3B_base_loess_internal,
        ha_3B_ft_loess_internal,
        ha_650M_base_loess_internal,
        ha_650M_ft_loess_internal,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ### ESM vs Time Plots - Internal Nodes Only (≥10 descendants)
    """)
    return


@app.cell
def _(
    ha_650M_base_loess_internal,
    ha_650M_ft_loess_internal,
    np,
    plot_esm_score,
    plt,
):
    # ESM vs Time 3-column figure for 650M (Internal nodes only)
    fig_650M_internal, axes_650M_internal = plt.subplots(1, 3, figsize=(18, 6))

    # Base model
    plot_esm_score(axes_650M_internal[0], ha_650M_base_loess_internal, "HA • 650M Base (Internal ≥10)", Fine_Tune=False, LOESS=False)

    # Fine-Tune model
    plot_esm_score(axes_650M_internal[1], ha_650M_ft_loess_internal, "HA • 650M Fine-Tune (Internal ≥10)", Fine_Tune=True, LOESS=False)

    # LOESS corrected
    plot_esm_score(axes_650M_internal[2], ha_650M_ft_loess_internal, "HA • 650M LOESS (Internal ≥10)", Fine_Tune=True, LOESS=True)

    # Set x-axis labels
    _years_650M_internal = np.arange(1965, 2026, 20)
    for _ax in axes_650M_internal:
        _ax.set_xticks(_years_650M_internal)
        _ax.set_xticklabels(_years_650M_internal, rotation=0, ha="right", fontsize=10)
        _ax.tick_params(axis="x", which="major", labelbottom=True)
        _ax.set_xlabel("Year", fontsize=8)

    plt.tight_layout(h_pad=2, w_pad=1)
    plt.gca()
    return


@app.cell
def _(
    ha_3B_base_loess_internal,
    ha_3B_ft_loess_internal,
    np,
    plot_esm_score,
    plt,
):
    # ESM vs Time 3-column figure for 3B (Internal nodes only)
    fig_3B_internal, axes_3B_internal = plt.subplots(1, 3, figsize=(18, 6))

    # Base model
    plot_esm_score(axes_3B_internal[0], ha_3B_base_loess_internal, "HA • 3B Base (Internal ≥10)", Fine_Tune=False, LOESS=False)

    # Fine-Tune model
    plot_esm_score(axes_3B_internal[1], ha_3B_ft_loess_internal, "HA • 3B Fine-Tune (Internal ≥10)", Fine_Tune=True, LOESS=False)

    # LOESS corrected
    plot_esm_score(axes_3B_internal[2], ha_3B_ft_loess_internal, "HA • 3B LOESS (Internal ≥10)", Fine_Tune=True, LOESS=True)

    # Set x-axis labels
    _years_3B_internal = np.arange(1965, 2026, 20)
    for _ax in axes_3B_internal:
        _ax.set_xticks(_years_3B_internal)
        _ax.set_xticklabels(_years_3B_internal, rotation=0, ha="right", fontsize=10)
        _ax.tick_params(axis="x", which="major", labelbottom=True)
        _ax.set_xlabel("Year", fontsize=8)

    plt.tight_layout(h_pad=2, w_pad=1)
    plt.gca()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 2-Column LOESS Comparison - Internal Nodes Only
    """)
    return


@app.cell
def _(
    ScalarFormatter,
    cm,
    colorsys,
    ha_3B_ft_loess_internal,
    ha_650M_ft_loess_internal,
    np,
    plt,
):
    # 2-column LOESS comparison (650M vs 3B) - Internal nodes only
    def darken_color_2col_internal(rgb, factor=0.7):
        h, l, s = colorsys.rgb_to_hls(*rgb)
        r, g, b = colorsys.hls_to_rgb(h, max(0, l * factor), s)
        return (r, g, b, 1.0)

    def plot_loess_finetune_2col_internal(ax, df, title):
        ll_col = "corrected_log_likelihood"
        norm = plt.Normalize(df[ll_col].min(), df[ll_col].max())
        cmap = plt.get_cmap("viridis")
        colors = cmap(norm(df[ll_col]))
        edgecolors = [darken_color_2col_internal(c[:3], factor=0.7) for c in colors]

        ax.scatter(
            df["time"],
            df[ll_col],
            c=colors,
            edgecolors=edgecolors,
            linewidths=0.5,
            alpha=0.7,
            zorder=1,
        )

        high_freq = df[df["max_frequency"] >= 1].sort_values("time")
        ax.plot(
            high_freq["time"],
            high_freq[ll_col],
            linestyle="-",
            color="black",
            linewidth=3,
            alpha=0.6,
            label="Max Freq ≥ 1",
            zorder=2,
        )

        ax.yaxis.offsetText.set_visible(False)
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = plt.colorbar(
            sm, ax=ax, orientation="vertical", pad=0.02, extend="both"
        )
        cbar.ax.yaxis.offsetText.set_visible(False)

        ax.set_title(title, fontsize=10)
        ax.axvline(2000, color="gray", linestyle="--", linewidth=1.5, zorder=1)
        ax.set_ylabel("ESM Score", fontsize=8)
        ax.grid(True, color="lightgray", linestyle="-", linewidth=0.75)
        ax.spines[["right", "top"]].set_visible(False)
        ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        ax.ticklabel_format(style="plain", axis="x")
        ax.set_xlim(1965, 2025)
        y_min, y_max = df[ll_col].min(), df[ll_col].max()
        pad = (y_max - y_min) * 0.05 if y_max != y_min else 1.0
        ax.set_ylim(y_min - pad, y_max + pad)
        return ax

    fig_2col_internal, axes_2col_internal = plt.subplots(1, 2, figsize=(14, 6))

    plot_loess_finetune_2col_internal(axes_2col_internal[0], ha_650M_ft_loess_internal, "HA • Fine_Tune_650M (Internal ≥10)")
    plot_loess_finetune_2col_internal(axes_2col_internal[1], ha_3B_ft_loess_internal, "HA • Fine_Tune_3B (Internal ≥10)")

    _years_2col_internal = np.arange(1960, 2021, 20)
    for _ax in axes_2col_internal:
        _ax.set_xticks(_years_2col_internal)
        _ax.set_xticklabels(_years_2col_internal, rotation=0, ha="right", fontsize=10)
        _ax.set_xlabel("Year", fontsize=8)

    plt.tight_layout(h_pad=2, w_pad=1)
    plt.gca()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Spearman CC Bar Charts - Internal Nodes Only (≥10 descendants)
    """)
    return


@app.cell
def _(
    ha_3B_base_loess_internal,
    ha_3B_ft_loess_internal,
    ha_650M_base_loess_internal,
    ha_650M_ft_loess_internal,
    pd,
    spearmanr,
):
    # Calculate correlation statistics for internal-only datasets
    def summary_stats_ha_internal(model_df, base_name, time_frame):
        results = []

        for model in model_df["Model"].unique():
            df = model_df[model_df["Model"] == model].copy()

            df_below_01 = df[df["max_frequency"] < 0.1]
            df_above_1 = df[df["max_frequency"] >= 0.99]

            spearman_corr, p_value = spearmanr(df["max_frequency"], df["log_likelihood"])

            results.append(
                {
                    "Model": model,
                    "Segment": "HA",
                    "Spearman Correlation Coefficient between Max Frequency and LL": spearman_corr,
                    "P-value": p_value,
                    "Mean ESM LL below 0.1": df_below_01["log_likelihood"].mean(),
                    "Mean ESM LL above 0.99": df_above_1["log_likelihood"].mean(),
                    "Difference in LL ESM Means": df_above_1["log_likelihood"].mean()
                    - df_below_01["log_likelihood"].mean(),
                    "Time Frame": time_frame,
                }
            )

            if model in ["Fine_Tune_3B", "Fine_Tune_650M"]:
                spearman_corr, p_value = spearmanr(
                    df["max_frequency"], df["corrected_log_likelihood"]
                )

                results.append(
                    {
                        "Model": f"LOESS_{model}",
                        "Segment": "HA",
                        "Spearman Correlation Coefficient between Max Frequency and LL": spearman_corr,
                        "P-value": p_value,
                        "Mean ESM LL below 0.1": df_below_01[
                            "corrected_log_likelihood"
                        ].mean(),
                        "Mean ESM LL above 0.99": df_above_1[
                            "corrected_log_likelihood"
                        ].mean(),
                        "Difference in LL ESM Means": df_above_1[
                            "corrected_log_likelihood"
                        ].mean()
                        - df_below_01["corrected_log_likelihood"].mean(),
                        "Time Frame": time_frame,
                    }
                )

        return pd.DataFrame(results)

    # Combine all internal datasets
    ha_all_internal = pd.concat(
        [ha_650M_base_loess_internal, ha_650M_ft_loess_internal, ha_3B_base_loess_internal, ha_3B_ft_loess_internal],
        ignore_index=True,
    )

    # Split by time
    ha_all_internal_above_2000 = ha_all_internal[ha_all_internal["time"] >= 2001]
    ha_all_internal_below_2000 = ha_all_internal[ha_all_internal["time"] <= 2000]

    # Calculate stats for internal-only datasets
    ha_3B_above_2000_results_internal = summary_stats_ha_internal(
        ha_all_internal_above_2000[ha_all_internal_above_2000["Model"].str.contains("3B")], "3B", "Post 2000"
    )
    ha_650M_above_2000_results_internal = summary_stats_ha_internal(
        ha_all_internal_above_2000[ha_all_internal_above_2000["Model"].str.contains("650M")],
        "650M",
        "Post 2000",
    )
    ha_3B_below_2000_results_internal = summary_stats_ha_internal(
        ha_all_internal_below_2000[ha_all_internal_below_2000["Model"].str.contains("3B")], "3B", "Pre 2000"
    )
    ha_650M_below_2000_results_internal = summary_stats_ha_internal(
        ha_all_internal_below_2000[ha_all_internal_below_2000["Model"].str.contains("650M")], "650M", "Pre 2000"
    )

    combined_results_ha_internal = pd.concat(
        [
            ha_3B_above_2000_results_internal,
            ha_650M_above_2000_results_internal,
            ha_3B_below_2000_results_internal,
            ha_650M_below_2000_results_internal,
        ],
        ignore_index=True,
    )
    return (
        combined_results_ha_internal,
        ha_3B_below_2000_results_internal,
        ha_650M_above_2000_results_internal,
        ha_650M_below_2000_results_internal,
        ha_all_internal,
    )


@app.cell
def _(
    combined_results_ha_internal,
    ha_3B_below_2000_results_internal,
    ha_650M_above_2000_results_internal,
    ha_650M_below_2000_results_internal,
    pd,
    plt,
):
    # Spearman CC bar chart (2x2 grid) - Internal nodes only
    def plot_spearman_barplot_ha_internal(ax, df, model_order, palette, title, xaxis=""):
        df_plot = df.copy()
        df_plot["Model"] = pd.Categorical(
            df_plot["Model"], categories=model_order, ordered=True
        )
        df_plot = df_plot.sort_values("Model")

        import seaborn as sns

        sns.barplot(
            data=df_plot,
            x="Segment",
            y="Spearman Correlation Coefficient between Max Frequency and LL",
            hue="Model",
            hue_order=model_order,
            errorbar=None,
            palette=palette,
            ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel(xaxis, weight="bold")
        ax.set_ylabel("Spearman CC (Max Freq. vs LL)", weight="bold")
        ax.legend(title="Model", frameon=False, loc="lower left")

    model_order_3B_internal = ["3B", "Fine_Tune_3B", "LOESS_Fine_Tune_3B"]
    model_order_650M_internal = ["650M", "Fine_Tune_650M", "LOESS_Fine_Tune_650M"]

    palette_3B_internal = {
        "3B": "#0a2463",
        "Fine_Tune_3B": "#f4d35e",
        "LOESS_Fine_Tune_3B": "#890304",
    }

    palette_650M_internal = {
        "650M": "#0a2463",
        "Fine_Tune_650M": "#f4d35e",
        "LOESS_Fine_Tune_650M": "#890304",
    }

    fig_corr_internal, axes_corr_internal = plt.subplots(2, 2, figsize=(10, 10), sharey=True)

    # Get the subsets for each plot
    df_3B_post_internal = combined_results_ha_internal[
        (combined_results_ha_internal["Time Frame"] == "Post 2000")
        & (combined_results_ha_internal["Model"].str.contains("3B"))
    ]
    df_650M_post_internal = ha_650M_above_2000_results_internal
    df_3B_pre_internal = ha_3B_below_2000_results_internal
    df_650M_pre_internal = ha_650M_below_2000_results_internal

    plot_spearman_barplot_ha_internal(
        axes_corr_internal[0, 0],
        df_3B_post_internal,
        model_order_3B_internal,
        palette_3B_internal,
        "3B - Internal ≥10 (Post-2000)",
        xaxis="",
    )
    plot_spearman_barplot_ha_internal(
        axes_corr_internal[0, 1],
        df_650M_post_internal,
        model_order_650M_internal,
        palette_650M_internal,
        "650M - Internal ≥10 (Post-2000)",
        xaxis="",
    )
    plot_spearman_barplot_ha_internal(
        axes_corr_internal[1, 0],
        df_3B_pre_internal,
        model_order_3B_internal,
        palette_3B_internal,
        "3B - Internal ≥10 (Pre-2000)",
        xaxis="Segment",
    )
    plot_spearman_barplot_ha_internal(
        axes_corr_internal[1, 1],
        df_650M_pre_internal,
        model_order_650M_internal,
        palette_650M_internal,
        "650M - Internal ≥10 (Pre-2000)",
        xaxis="Segment",
    )

    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Time vs Max Frequency - Internal Nodes Only (≥10 descendants)
    """)
    return


@app.cell
def _(LinearRegression, ha_all_internal, np, pd, plt, spearmanr):
    # Time vs Max Frequency scatter - Internal nodes only
    import seaborn as _sns_time_internal

    ha_all_for_time_internal = ha_all_internal[ha_all_internal["Model"] == "Fine_Tune_650M"].copy()

    rho_internal, pval_internal = spearmanr(ha_all_for_time_internal["time"], ha_all_for_time_internal["max_frequency"])
    print(f"Internal Nodes - Spearman ρ = {rho_internal:.3f}, p = {pval_internal:.3g}")

    ha_all_for_time_internal["time_rank"] = ha_all_for_time_internal["time"].rank()
    ha_all_for_time_internal["freq_rank"] = ha_all_for_time_internal["max_frequency"].rank()

    _X_internal = ha_all_for_time_internal[["time_rank"]].values
    _y_time_internal = ha_all_for_time_internal["freq_rank"].values
    lm_internal = LinearRegression().fit(_X_internal, _y_time_internal)

    fig_time_freq_internal = plt.figure(figsize=(8, 6))
    _sns_time_internal.scatterplot(x="time", y="max_frequency", data=ha_all_for_time_internal)

    x_line_internal = np.linspace(
        ha_all_for_time_internal["time"].min(), ha_all_for_time_internal["time"].max(), 100
    )
    x_line_rank_internal = pd.Series(x_line_internal).rank(method="first", pct=False).values
    y_line_rank_internal = lm_internal.predict(x_line_rank_internal.reshape(-1, 1))
    y_line_internal = np.percentile(
        ha_all_for_time_internal["max_frequency"],
        100 * (y_line_rank_internal - 1) / (len(ha_all_for_time_internal) - 1),
    )

    plt.plot(
        x_line_internal,
        y_line_internal,
        color="red",
        linestyle="--",
        label=f"Spearman fit (ρ={rho_internal:.2f})",
    )
    plt.legend()
    plt.title("HA time vs maximum frequency (Internal ≥10)")
    plt.gca()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Max Frequency vs ESM Score Comparison: All Nodes vs Internal ≥10
    Comparing correlations split by training (≤2000) and test (>2000) periods for 650M and 3B models.
    """)
    return


@app.cell
def _(ha_all, ha_all_internal, np, plt, spearmanr):
    # 650M Model: 2x2 grid (Training vs Test, All vs Internal)
    import seaborn as _sns_650M

    _all_650M = ha_all[ha_all["Model"] == "Fine_Tune_650M"].copy()
    _internal_650M = ha_all_internal[ha_all_internal["Model"] == "Fine_Tune_650M"].copy()

    # Split by time period
    _all_650M_train = _all_650M[_all_650M["time"] <= 2000]
    _all_650M_test = _all_650M[_all_650M["time"] > 2000]
    _internal_650M_train = _internal_650M[_internal_650M["time"] <= 2000]
    _internal_650M_test = _internal_650M[_internal_650M["time"] > 2000]

    # Create 2x2 grid
    _fig_650M, _axes_650M = plt.subplots(2, 2, figsize=(14, 12))
    _fig_650M.suptitle("650M Model: Max Frequency vs LOESS-Corrected ESM Score", fontsize=14, weight="bold", y=0.995)

    # Define plotting function
    def _plot_scatter_panel(ax, data, title, color, edge_color):
        if len(data) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title, fontsize=10)
            return

        rho, pval = spearmanr(data["max_frequency"], data["corrected_log_likelihood"])

        ax.scatter(
            data["max_frequency"],
            data["corrected_log_likelihood"],
            alpha=0.5,
            s=20,
            color=color,
            edgecolors=edge_color,
            linewidths=0.5
        )

        # Trend line
        z = np.polyfit(data["max_frequency"], data["corrected_log_likelihood"], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(data["max_frequency"].min(), data["max_frequency"].max(), 100)
        ax.plot(x_trend, p(x_trend), "r--", linewidth=2, alpha=0.8)

        ax.set_xlabel("Maximum Frequency", fontsize=10, weight="bold")
        ax.set_ylabel("LOESS-Corrected ESM Score", fontsize=10, weight="bold")
        ax.set_title(f"{title}\nn={len(data)}, ρ={rho:.3f}, p={pval:.2g}", fontsize=10)
        ax.grid(True, alpha=0.3)

    # Plot all 4 panels
    _plot_scatter_panel(_axes_650M[0, 0], _all_650M_train, "All Nodes - Training (≤2000)", "steelblue", "navy")
    _plot_scatter_panel(_axes_650M[0, 1], _all_650M_test, "All Nodes - Test (>2000)", "steelblue", "navy")
    _plot_scatter_panel(_axes_650M[1, 0], _internal_650M_train, "Internal ≥10 - Training (≤2000)", "darkorange", "darkred")
    _plot_scatter_panel(_axes_650M[1, 1], _internal_650M_test, "Internal ≥10 - Test (>2000)", "darkorange", "darkred")

    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(ha_all, ha_all_internal, np, plt, spearmanr):
    # 3B Model: 2x2 grid (Training vs Test, All vs Internal)
    import seaborn as _sns_3B

    _all_3B = ha_all[ha_all["Model"] == "Fine_Tune_3B"].copy()
    _internal_3B = ha_all_internal[ha_all_internal["Model"] == "Fine_Tune_3B"].copy()

    # Split by time period
    _all_3B_train = _all_3B[_all_3B["time"] <= 2000]
    _all_3B_test = _all_3B[_all_3B["time"] > 2000]
    _internal_3B_train = _internal_3B[_internal_3B["time"] <= 2000]
    _internal_3B_test = _internal_3B[_internal_3B["time"] > 2000]

    # Create 2x2 grid
    _fig_3B, _axes_3B = plt.subplots(2, 2, figsize=(14, 12))
    _fig_3B.suptitle("3B Model: Max Frequency vs LOESS-Corrected ESM Score", fontsize=14, weight="bold", y=0.995)

    # Reuse plotting function
    def _plot_scatter_panel_3B(ax, data, title, color, edge_color):
        if len(data) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title, fontsize=10)
            return

        rho, pval = spearmanr(data["max_frequency"], data["corrected_log_likelihood"])

        ax.scatter(
            data["max_frequency"],
            data["corrected_log_likelihood"],
            alpha=0.5,
            s=20,
            color=color,
            edgecolors=edge_color,
            linewidths=0.5
        )

        # Trend line
        z = np.polyfit(data["max_frequency"], data["corrected_log_likelihood"], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(data["max_frequency"].min(), data["max_frequency"].max(), 100)
        ax.plot(x_trend, p(x_trend), "r--", linewidth=2, alpha=0.8)

        ax.set_xlabel("Maximum Frequency", fontsize=10, weight="bold")
        ax.set_ylabel("LOESS-Corrected ESM Score", fontsize=10, weight="bold")
        ax.set_title(f"{title}\nn={len(data)}, ρ={rho:.3f}, p={pval:.2g}", fontsize=10)
        ax.grid(True, alpha=0.3)

    # Plot all 4 panels
    _plot_scatter_panel_3B(_axes_3B[0, 0], _all_3B_train, "All Nodes - Training (≤2000)", "mediumseagreen", "darkgreen")
    _plot_scatter_panel_3B(_axes_3B[0, 1], _all_3B_test, "All Nodes - Test (>2000)", "mediumseagreen", "darkgreen")
    _plot_scatter_panel_3B(_axes_3B[1, 0], _internal_3B_train, "Internal ≥10 - Training (≤2000)", "mediumpurple", "indigo")
    _plot_scatter_panel_3B(_axes_3B[1, 1], _internal_3B_test, "Internal ≥10 - Test (>2000)", "mediumpurple", "indigo")

    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Sliding Window Analysis - Internal Nodes Only (≥10 descendants)
    """)
    return


@app.cell
def _(ha_all_internal, pd, spearmanr):
    # Calculate sliding window correlations - Internal nodes only
    def spearman_correlation_ha_internal(df):
        results = []

        df_650M = df[df["Model"] == "Fine_Tune_650M"].copy()
        df_3B = df[df["Model"] == "Fine_Tune_3B"].copy()

        time_ranges = [
            (1970, 1990, "1980"),
            (1980, 2000, "1990"),
            (1990, 2010, "2000"),
            (2000, 2020, "2010"),
            (2010, None, "2020"),
        ]

        for start, end, label in time_ranges:
            if end is None:
                df_650M_label = df_650M[df_650M["time"] >= start]
                df_3B_label = df_3B[df_3B["time"] >= start]
            else:
                df_650M_label = df_650M[
                    (df_650M["time"] >= start) & (df_650M["time"] <= end)
                ]
                df_3B_label = df_3B[(df_3B["time"] >= start) & (df_3B["time"] <= end)]

            # 650M correlations
            if len(df_650M_label) > 0:
                spearman_corr, p_value = spearmanr(
                    df_650M_label["max_frequency"], df_650M_label["log_likelihood"]
                )
                results.append(
                    {
                        "Model": "Fine_Tune_650M",
                        "Segment": "HA",
                        "Time_Range": label,
                        "Spearman_Correlation": spearman_corr,
                        "P_Value": p_value,
                    }
                )

                spearman_corr, p_value = spearmanr(
                    df_650M_label["max_frequency"],
                    df_650M_label["corrected_log_likelihood"],
                )
                results.append(
                    {
                        "Model": "LOESS_Fine_Tune_650M",
                        "Segment": "HA",
                        "Time_Range": label,
                        "Spearman_Correlation": spearman_corr,
                        "P_Value": p_value,
                    }
                )

            # 3B correlations
            if len(df_3B_label) > 0:
                spearman_corr, p_value = spearmanr(
                    df_3B_label["max_frequency"], df_3B_label["log_likelihood"]
                )
                results.append(
                    {
                        "Model": "Fine_Tune_3B",
                        "Segment": "HA",
                        "Time_Range": label,
                        "Spearman_Correlation": spearman_corr,
                        "P_Value": p_value,
                    }
                )

                spearman_corr, p_value = spearmanr(
                    df_3B_label["max_frequency"],
                    df_3B_label["corrected_log_likelihood"],
                )
                results.append(
                    {
                        "Model": "LOESS_Fine_Tune_3B",
                        "Segment": "HA",
                        "Time_Range": label,
                        "Spearman_Correlation": spearman_corr,
                        "P_Value": p_value,
                    }
                )

        return pd.DataFrame(results)

    ha_spearman_time_internal = spearman_correlation_ha_internal(ha_all_internal)
    return (ha_spearman_time_internal,)


@app.cell
def _(ha_spearman_time_internal, np, plt):
    # Plot sliding window results - Internal nodes only
    import seaborn as _sns_sliding_internal

    _sns_sliding_internal.set_style("whitegrid")
    custom_params_internal = {"axes.spines.right": False, "axes.spines.top": False}
    _sns_sliding_internal.set_theme(style="ticks", rc=custom_params_internal)

    fig_sliding_internal, ax_sliding_internal = plt.subplots(figsize=(10, 5))

    ax_sliding_internal = _sns_sliding_internal.lineplot(
        data=ha_spearman_time_internal,
        x="Time_Range",
        y="Spearman_Correlation",
        hue="Model",
        marker="o",
        legend=False,
        zorder=1,
        ax=ax_sliding_internal,
        errorbar=None,
    )

    ax_sliding_internal.set_title("Spearman CC Summary HA Segment (Internal ≥10)")
    ax_sliding_internal.set_xlabel("Time Range")
    ax_sliding_internal.set_ylabel("Spearman CC")

    _label_positions_internal = []
    for _line_internal, _model_internal in zip(ax_sliding_internal.lines, ha_spearman_time_internal["Model"].unique()):
        _y_internal = _line_internal.get_ydata()[-1]
        _x_internal = _line_internal.get_xdata()[-1]

        if not np.isfinite(_y_internal) or not np.isfinite(_x_internal):
            continue

        while any(abs(_y_internal - pos) < 0.025 for pos in _label_positions_internal):
            _y_internal += 0.007

        _label_positions_internal.append(_y_internal)

        ax_sliding_internal.annotate(
            _model_internal,
            xy=(_x_internal, _y_internal),
            xytext=(5, 0),
            textcoords="offset points",
            color=_line_internal.get_color(),
            fontsize=12,
            weight="bold",
            ha="left",
            va="center",
            zorder=2,
        )

    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Comparison: All Nodes vs Internal Nodes (≥10 descendants)

    ### Key Findings

    **Dataset Sizes:**
    - Filtering to internal nodes with ≥10 descendants significantly reduces dataset size
    - This focuses analysis on more stable phylogenetic positions with greater sampling support
    - Internal nodes represent ancestral sequences with multiple descendant lineages

    **ESM vs Time Plots:**
    - Both all-nodes and internal-filtered datasets show similar temporal trends
    - LOESS correction effectively removes time-dependent bias in both cases
    - Trunk visualization (black line) is clearer in internal-filtered plots due to reduced noise

    **Spearman Correlations:**
    - Compare the 2x2 grid bar charts between all-nodes and internal-filtered datasets
    - Internal nodes may show stronger or weaker correlations depending on dataset characteristics
    - Pre/Post 2000 patterns should be examined for both filtering approaches

    **Sliding Window Analysis:**
    - Both datasets show similar trends across time windows
    - LOESS correction stabilizes correlations in both cases
    - Internal-only filtering may reduce variance in some time periods

    ### Biological Interpretation

    **Why filter to internal nodes ≥10?**
    - Internal nodes represent ancestral sequences inferred from multiple descendants
    - Nodes with ≥10 descendants are more robustly inferred
    - Reduces impact of sampling bias and terminal node variability
    - Focuses on trunk and major branches of the phylogenetic tree

    **Trade-offs:**
    - **All nodes**: Maximum data coverage, includes terminal branches
    - **Internal ≥10**: Higher confidence positions, focuses on major lineages

    Both analyses are valuable and provide complementary insights into ESM score dynamics across the HA phylogeny.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Summary
    This analysis shows that LOESS correction successfully removes temporal trends in ESM scores for HA segment, resulting in more stable correlations with maximum frequency across time periods.

    The comprehensive HA analysis includes:
    1. **All-nodes analysis** (lines 788-1413): Complete dataset with all terminal and internal nodes
    2. **Internal-nodes (≥10) analysis** (new cells): Filtered to high-confidence ancestral positions

    Key visualizations for both approaches:
    - ESM vs Time plots (3-column: Base, Fine-Tune, LOESS)
    - 2-column LOESS comparisons (650M vs 3B)
    - Spearman CC bar charts (2x2: Pre/Post 2000)
    - Time vs Max Frequency scatter
    - Sliding window analysis

    Both filtering strategies demonstrate that LOESS correction is effective at removing temporal bias and improving the stability of correlations between ESM scores and maximum frequency.
    """)
    return


if __name__ == "__main__":
    app.run()
