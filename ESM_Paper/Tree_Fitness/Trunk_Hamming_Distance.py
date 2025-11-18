# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo",
#     "matplotlib==3.10.7",
#     "numpy==2.3.4",
#     "pandas==2.3.3",
#     "polars==1.35.2",
#     "scipy==1.16.3",
# ]
# ///

import marimo

__generated_with = "0.17.8"
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
    import re
    return ScalarFormatter, cm, colorsys, mo, mpl, np, pd, pl, plt


@app.cell
def _(mo):
    mo.md("""
    # HA ESM Scores vs Time Analysis

    Analysis of ESM fitness scores for hemagglutinin (HA) sequences over time with LOESS correction.
    Using the LOESS implementation and plotting approach from the reference notebook.
    """)
    return


@app.cell
def _():
    import json

    # Load the h3n2/ha.json file for node time mapping
    with open('/Users/cavendan/Desktop/esm-selection/ESM_Paper/Tree_Fitness/h3n2/ha.json', 'r') as f:
        tree_data = json.load(f)

    def extract_node_times(node, node_times_dict=None):
        """Recursively extract node names and their num_date values from the tree."""
        if node_times_dict is None:
            node_times_dict = {}

        # Extract time from current node
        if 'node_attrs' in node and 'num_date' in node['node_attrs']:
            time_value = node['node_attrs']['num_date']['value']
            node_times_dict[node['name']] = time_value

        # Recursively process children
        if 'children' in node:
            for child in node['children']:
                extract_node_times(child, node_times_dict)

        return node_times_dict

    # Extract all node times from the tree
    node_times = extract_node_times(tree_data['tree'])
    print(f"Loaded {len(node_times)} node-time mappings from h3n2/ha.json")

    # Show some example NODE_ entries
    node_examples = {name: time for name, time in list(node_times.items())[:10] if name.startswith('NODE_')}
    print("Example NODE_ time mappings:", node_examples)
    return (node_times,)


@app.cell
def _(pl):
    ha_data = pl.read_csv("/Users/cavendan/Desktop/esm-selection/ESM_Paper/Tree_Fitness/next_tree~h3n2/epochs~1/learning_rate~5e-05/model~esm2_t33_650M_UR50D/time~2000/H3N2_Dataset_1965_Full/Max_Freq_Fasta_LL_Fine_Tune_ha.csv")
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
            bad = None
            for p in range(10):  # do at most 10 iterations
                if sigy is None:                # Errors are unknown
                    aerr = np.abs(yfit - y[w])  # Note ABS()
                    mad = np.median(aerr)       # Characteristic scale

                    if mad == 0:
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
    return get_node_time, loess_1d


@app.cell
def _(get_node_time, ha_data):
    # Convert to pandas for compatibility with reference notebook functions
    ha_df = ha_data.to_pandas()

    # Extract time information from phylogenetic tree - no arbitrary filters
    ha_df['time'] = ha_df['node'].apply(get_node_time)
    ha_df = ha_df.dropna(subset=['time'])
    ha_df = ha_df[ha_df['time'] >= 1968]

    # Filter out sequences with gaps (- characters)
    initial_count = len(ha_df)
    ha_df = ha_df[~ha_df['sequence'].str.contains('-', na=False)]
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
    ha_df_with_loess['corrected_log_likelihood'] = ha_df_with_loess['log_likelihood'] - ha_df_with_loess['log_likelihood_LOESS']

    # Add trunk labeling - nodes with max frequency = 1.0 (100%) and contain "NODE_"
    ha_df_with_loess['is_trunk'] = (
        (ha_df_with_loess['max_frequency'] >= .99) & 
        (ha_df_with_loess['node'].str.contains('NODE_'))
    )

    trunk_count = ha_df_with_loess['is_trunk'].sum()
    print(f"Identified {trunk_count} trunk nodes")

    ha_df_with_loess
    return (ha_df_with_loess,)


@app.cell
def _(ha_df_with_loess):
    # Filter sequences to keep only those with the most common length
    sequence_lengths = ha_df_with_loess['sequence'].str.len()
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
            seq_len = len(row['sequence'])
            trunk_status = "TRUNK" if row['is_trunk'] else "BRANCH"
            length_diff = seq_len - most_common_length
            diff_str = f"+{length_diff}" if length_diff > 0 else str(length_diff)
            print(f"  {row['node']} -> Length: {seq_len} ({diff_str} from mode), Time: {row['time']:.1f}, Type: {trunk_status}")

            if len(non_standard_seqs) <= 5:  # Show sequence for small number of exclusions
                print(f"    Sequence: {row['sequence']}")
            elif count >= 10:  # Limit output for very large numbers
                remaining = len(non_standard_seqs) - count
                if remaining > 0:
                    print(f"  ... and {remaining} more excluded nodes")
                break
    else:
        print(f"✓ All sequences match the mode length ({most_common_length}). No exclusions needed.")

    # Create filtered dataframe with only standard-length sequences
    ha_df_length_filtered = ha_df_with_loess[sequence_lengths == most_common_length].copy()

    print(f"\nFiltering Results:")
    print(f"Original sequences: {len(ha_df_with_loess)}")
    print(f"Filtered sequences: {len(ha_df_length_filtered)}")
    print(f"Excluded sequences: {len(ha_df_with_loess) - len(ha_df_length_filtered)}")

    # Verify all sequences now have the same length
    filtered_lengths = ha_df_length_filtered['sequence'].str.len()
    print(f"All remaining sequences have length {filtered_lengths.iloc[0]}: {filtered_lengths.nunique() == 1}")
    return (ha_df_length_filtered,)


@app.cell
def _(ScalarFormatter, cm, colorsys, plt):
    # Plotting functions from reference notebook
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

        # Add grid behind data points
        ax.grid(True, color='lightgray', linestyle='-', linewidth=0.75, zorder=0)

        ax.scatter(
            df["time"],
            df[ll_col],
            c=colors,
            edgecolors=edgecolors,
            linewidths=0.5,
            alpha=0.7,
            zorder=3
        )

        # Connect trunk nodes (max frequency >= 1.0 and contains NODE_)
        trunk_df = (
            df[
                (df["max_frequency"] >= 1.0) &
                (df["node"].str.contains("NODE_", na=False))
            ]
            .sort_values("time")
        )
        if len(trunk_df) > 0:
            ax.plot(
                trunk_df["time"],
                trunk_df[ll_col],
                linestyle='-',
                color='black',
                linewidth=3,
                alpha=0.6,
                label='Trunk (Max Freq ≥ 1 & NODE_)',
                zorder=4
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
            ax.axvline(1990, color='gray', linestyle='--', linewidth=1.5, zorder=1)

        ax.set_ylabel("ESM Score", fontsize=8)
        ax.spines[['right', 'top']].set_visible(False)
        ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        ax.ticklabel_format(style='plain', axis='x')
        ax.set_xlim(1965, 2030)

        y_min, y_max = df[ll_col].min(), df[ll_col].max()
        pad = (y_max - y_min) * 0.05 if y_max != y_min else 1.0
        ax.set_ylim(y_min - pad, y_max + pad)

        return ax
    return (plot_esm_score,)


@app.cell
def _(ha_df_length_filtered, mpl, np, plot_esm_score, plt):
    # Set light theme from reference notebook
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

    # Create ESM vs Time plots using reference notebook approach
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Raw ESM scores
    plot_esm_score(axes[0], ha_df_length_filtered, "HA • Raw ESM Scores vs Time")

    # LOESS corrected scores
    plot_esm_score(axes[1], ha_df_length_filtered, "HA • LOESS Corrected ESM Scores", LOESS=True)

    # Set x-axis labels with extended range to include 2025+ data
    years = np.arange(1967, 2026, 20)    
    for ax in axes:                    
        ax.set_xticks(years)               
        ax.set_xticklabels(years,         
                           rotation=0,    
                           ha='right',
                           fontsize=10
                          )
        ax.tick_params(axis='x',
                       which='major',
                       labelbottom=True)
        ax.set_xlabel("Year", fontsize=8)

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
        time_diffs = abs(trunk_df['time'] - target_time)
        nearest_idx = time_diffs.idxmin()
        return trunk_df.loc[nearest_idx]

    # Separate trunk and branch nodes using existing is_trunk column
    trunk_nodes = ha_df_length_filtered[ha_df_length_filtered['is_trunk']].copy()
    branch_nodes = ha_df_length_filtered[~ha_df_length_filtered['is_trunk']].copy()

    print(f"Working with filtered data:")
    print(f"Total nodes: {len(ha_df_length_filtered)}")
    print(f"Trunk nodes: {len(trunk_nodes)}")
    print(f"Branch nodes: {len(branch_nodes)}")
    print(f"All sequences have length: {len(ha_df_length_filtered['sequence'].iloc[0])}")

    # Calculate hamming distances (no length checking needed since all sequences are filtered)
    hamming_distances = []
    for i, node_row in ha_df_length_filtered.iterrows():
        if node_row['is_trunk']:
            hamming_distances.append(0)  # Trunk nodes get 0
        else:
            nearest_trunk = find_nearest_trunk_node(node_row['time'], trunk_nodes)
            distance = hamming_distance(node_row['sequence'], nearest_trunk['sequence'])
            hamming_distances.append(distance)

    # Add the column to the dataframe
    ha_df_final = ha_df_length_filtered.copy()
    ha_df_final['hamming_distance_to_trunk'] = hamming_distances

    # Summary statistics
    branch_distances = [d for d, is_trunk in zip(hamming_distances, ha_df_final['is_trunk']) if not is_trunk]
    if branch_distances:
        print(f"\nHamming Distance Statistics for Branch Nodes:")
        print(f"Mean: {sum(branch_distances)/len(branch_distances):.2f}")
        print(f"Min: {min(branch_distances)}")
        print(f"Max: {max(branch_distances)}")
        print(f"Total branch nodes with distances: {len(branch_distances)}")

    # Show some examples
    print(f"\nExample branch nodes and their nearest trunk matches:")
    branch_examples = ha_df_final[~ha_df_final['is_trunk']].head(3)
    for j, example_row in branch_examples.iterrows():
        nearest_trunk = find_nearest_trunk_node(example_row['time'], trunk_nodes)
        print(f"Branch: {example_row['node']} (time: {example_row['time']:.1f}) -> Nearest trunk: {nearest_trunk['node']} (time: {nearest_trunk['time']:.1f}), Distance: {example_row['hamming_distance_to_trunk']}")
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
    correlation_data = ha_df_final.dropna(subset=['max_frequency', 'log_likelihood', 'corrected_log_likelihood'])

    # Split data by time: Training (< 2000) and Testing (>= 2000)
    training_data = correlation_data[correlation_data['time'] < 2000]
    testing_data = correlation_data[correlation_data['time'] >= 2000]

    def plot_correlation_subplot(ax, x_data, y_data, title, color='steelblue', colormap_data=None):
        """Helper function to create correlation subplot."""
        if len(x_data) == 0 or len(y_data) == 0:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center', va='center')
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
        corr_text = f'ρ={spearman_r:.2f}\nr={pearson_r:.2f}'
        ax.text(0.05, 0.95, corr_text, transform=ax.transAxes, fontsize=7,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_title(title, fontsize=9)
        ax.grid(True, alpha=0.3)

        return spearman_r, pearson_r

    # Create 2x2 figure
    grid_fig, grid_axes = plt.subplots(2, 2, figsize=(12, 8))

    correlation_results = {}

    # Top Left: Training - Raw ESM
    spearman_r, pearson_r = plot_correlation_subplot(
        grid_axes[0, 0], 
        training_data['max_frequency'], 
        training_data['log_likelihood'],
        f"Training Raw ESM - 650M (Pre 2000)",
        color='steelblue',
        colormap_data=None
    )
    correlation_results['train_raw'] = (spearman_r, pearson_r)

    # Top Right: Training - LOESS
    spearman_r, pearson_r = plot_correlation_subplot(
        grid_axes[0, 1], 
        training_data['max_frequency'], 
        training_data['corrected_log_likelihood'],
        f"Training LOESS - 650M (Pre 2000)",
        color='steelblue',
        colormap_data=None
    )
    correlation_results['train_loess'] = (spearman_r, pearson_r)

    # Bottom Left: Testing - Raw ESM  
    spearman_r, pearson_r = plot_correlation_subplot(
        grid_axes[1, 0], 
        testing_data['max_frequency'], 
        testing_data['log_likelihood'],
        f"Testing Raw ESM - 650M (Post 2000)",
        color='darkgreen',
        colormap_data=None
    )
    correlation_results['test_raw'] = (spearman_r, pearson_r)

    # Bottom Right: Testing - LOESS
    spearman_r, pearson_r = plot_correlation_subplot(
        grid_axes[1, 1], 
        testing_data['max_frequency'], 
        testing_data['corrected_log_likelihood'],
        f"Testing LOESS - 650M (Post 2000)",
        color='darkgreen',
        colormap_data=None
    )
    correlation_results['test_loess'] = (spearman_r, pearson_r)

    # Set axis labels
    grid_axes[0, 0].set_ylabel('ESM Score', fontsize=12)
    grid_axes[1, 0].set_ylabel('ESM Score', fontsize=12)
    grid_axes[1, 0].set_xlabel('Maximum Frequency', fontsize=12)
    grid_axes[1, 1].set_xlabel('Maximum Frequency', fontsize=12)

    plt.tight_layout(pad=1.5)
    plt.show()

    # Create second 2x2 figure comparing ESM scores with Hamming distance
    hamming_fig, hamming_axes = plt.subplots(2, 2, figsize=(12, 8))

    hamming_correlation_results = {}

    # Top Left: Training - Raw ESM vs Hamming Distance
    spearman_r, pearson_r = plot_correlation_subplot(
        hamming_axes[0, 0], 
        training_data['hamming_distance_to_trunk'], 
        training_data['log_likelihood'],
        f"Training Raw ESM vs Hamming - 650M (Pre 2000)",
        color='steelblue',
        colormap_data=None
    )
    hamming_correlation_results['train_raw'] = (spearman_r, pearson_r)

    # Top Right: Training - LOESS vs Hamming Distance
    spearman_r, pearson_r = plot_correlation_subplot(
        hamming_axes[0, 1], 
        training_data['hamming_distance_to_trunk'], 
        training_data['corrected_log_likelihood'],
        f"Training LOESS vs Hamming - 650M (Pre 2000)",
        color='steelblue',
        colormap_data=None
    )
    hamming_correlation_results['train_loess'] = (spearman_r, pearson_r)

    # Bottom Left: Testing - Raw ESM vs Hamming Distance
    spearman_r, pearson_r = plot_correlation_subplot(
        hamming_axes[1, 0], 
        testing_data['hamming_distance_to_trunk'], 
        testing_data['log_likelihood'],
        f"Testing Raw ESM vs Hamming - 650M (Post 2000)",
        color='darkgreen',
        colormap_data=None
    )
    hamming_correlation_results['test_raw'] = (spearman_r, pearson_r)

    # Bottom Right: Testing - LOESS vs Hamming Distance
    spearman_r, pearson_r = plot_correlation_subplot(
        hamming_axes[1, 1], 
        testing_data['hamming_distance_to_trunk'], 
        testing_data['corrected_log_likelihood'],
        f"Testing LOESS vs Hamming - 650M (Post 2000)",
        color='darkgreen',
        colormap_data=None
    )
    hamming_correlation_results['test_loess'] = (spearman_r, pearson_r)

    # Set axis labels for hamming distance plots
    hamming_axes[0, 0].set_ylabel('ESM Score', fontsize=12)
    hamming_axes[1, 0].set_ylabel('ESM Score', fontsize=12)
    hamming_axes[1, 0].set_xlabel('Hamming Distance to Trunk', fontsize=12)
    hamming_axes[1, 1].set_xlabel('Hamming Distance to Trunk', fontsize=12)

    plt.tight_layout(pad=1.5)
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
