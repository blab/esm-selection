# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "altair==5.5.0",
#     "duckdb==1.3.2",
#     "marimo",
#     "nbformat==5.10.4",
#     "openai==1.107.2",
#     "polars[pyarrow]==1.33.1",
#     "pytest==8.4.2",
#     "sqlglot==27.14.0",
#     "vegafusion==2.0.2",
#     "vl-convert-python==1.8.0",
#     "pandas==2.3.2",
#     "matplotlib==3.9.4",
#     "seaborn==0.13.2",
#     "scikit-learn==1.7.2",
#     "scipy==1.16.2",
#     "numpy==2.3.3",
#     "scikit-misc>=0.5.1; python_version < '3.13'",
# ]
# ///

import marimo

__generated_with = "0.16.2"
app = marimo.App(
    width="medium",
    app_title="ESM2 Validation Pipeline with LOESS",
)


@app.cell
def _(mo):
    mo.md(r"""# ESM2 Validation Pipeline with LOESS Correction - 650M vs 3B Model Comparison""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    This notebook imports datasets from ESM2_Validation_Pipeline/results/Log_Likelihoods/ 
    and applies LOESS correction to analyze ESM model performance across different training datasets.

    This version compares both 650M (esm2_t33_650M_UR50D) and 3B (esm2_t36_3B_UR50D) models.
    LOESS correction is applied separately to each model size with independent caching.

    The LOESS correction removes the negative trend of ESM score vs Time to provide more accurate fitness measurements.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## Setup and Imports""")
    return


@app.cell
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
    from sklearn.linear_model import LinearRegression
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    np.seterr(divide='ignore', over='ignore', invalid='ignore')
    import glob
    from pathlib import Path
    return (
        ScalarFormatter,
        colorsys,
        glob,
        mo,
        mpl,
        np,
        os,
        pd,
        plt,
        sns,
        spearmanr,
    )


@app.cell
def _(np):
    # LOESS implementation from Apply_LOESS_To_Previous_ESM.py
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


@app.cell
def _(mo):
    mo.md(r"""## Data Loading Functions""")
    return


@app.cell
def _(glob, os, pd):
    def load_datasets_from_directory(base_path: str) -> pd.DataFrame:
        """
        Load all CSV datasets from the ESM2_Validation_Pipeline/results/Log_Likelihoods/ directory
        and combine them into a single DataFrame with metadata extracted from file paths.
        """
        csv_files = glob.glob(os.path.join(base_path, "**/*.csv"), recursive=True)

        all_dataframes = []

        for file_path in csv_files:
            # Skip files from h3n2-Small directory to avoid duplicates
            if "h3n2-Small" in file_path:
                continue
            try:
                df = pd.read_csv(file_path)

                # Extract metadata from file path
                path_parts = file_path.split(os.sep)

                # Find relevant path components
                metadata = {}
                for part in path_parts:
                    if part.startswith("epochs~"):
                        metadata["epochs"] = part.split("~")[1]
                    elif part.startswith("learning_rate~"):
                        metadata["learning_rate"] = part.split("~")[1]
                    elif part.startswith("model~"):
                        metadata["model_type"] = part.split("~")[1]
                    elif part.startswith("time~"):
                        metadata["training_time"] = int(part.split("~")[1])
                    elif part.startswith("lr~"):
                        metadata["lr"] = part.split("~")[1]

                # Determine dataset type and model configuration
                if "base" in file_path:
                    metadata["model_config"] = "Base"
                    metadata["training_dataset"] = "None"
                elif "Fine_Tune" in file_path:
                    # Extract training dataset from path
                    for part in path_parts:
                        if part in ["ESM_1965_Full", "H3N2_Dataset_1965_Full", "pan_flu_1965_Full"] or part.startswith("mix_"):
                            metadata["training_dataset"] = part
                            break
                    metadata["model_config"] = "Fine_Tune"

                # Extract segment from filename
                filename = os.path.basename(file_path)
                if filename.endswith('.csv'):
                    segment = filename.split('_')[-1].replace('.csv', '')
                    metadata["segment"] = segment.upper()

                # Add metadata to dataframe
                for key, value in metadata.items():
                    df[key] = value

                # Add file path for reference
                df["source_file"] = file_path

                all_dataframes.append(df)

            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue

        if all_dataframes:
            combined_df = pd.concat(all_dataframes, ignore_index=True)
            return combined_df
        else:
            return pd.DataFrame()

    def extract_node_times(tree_data, segment):
        """Extract node times from tree JSON data."""
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
        """Process all JSON files in a directory to extract node times."""
        import json
        all_data = []

        if not os.path.exists(directory):
            print(f"Directory {directory} does not exist")
            return pd.DataFrame()

        for filename in os.listdir(directory):
            if filename.endswith('.json'):
                segment = filename[:-5]  # remove the '.json' suffix
                file_path = os.path.join(directory, filename)
                try:
                    with open(file_path, 'r') as f:
                        tree_data = json.load(f)

                    segment_data = extract_node_times(tree_data, segment)
                    all_data.extend(segment_data)
                    print(f"Processed {filename}: found {len(segment_data)} nodes")
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
                    continue

        if all_data:
            df = pd.DataFrame(all_data)
            print(f"Total nodes extracted: {len(df)}")
            print(f"Segments found: {df['Segment'].unique()}")
            return df
        else:
            print("No data extracted from tree files")
            return pd.DataFrame()

    def merge_time(models_df, tree_dir=""):
        """Merge time data from tree JSON files with models dataframe."""
        directory = f"../ESM2_Validation_Pipeline/input/trees/{tree_dir}/" if tree_dir else "../ESM2_Validation_Pipeline/input/trees/"

        # Check if directory exists
        if not os.path.exists(directory):
            print(f"Warning: Tree directory {directory} not found")
            return models_df

        try:
            df = process_directory(directory)
            df['Segment'] = df['Segment'].str.upper()

            # Ensure models_df has the right column names for merging
            segment_col = 'segment' if 'segment' in models_df.columns else 'Segment'

            # Create a copy and standardize column names for merging
            models_copy = models_df.copy()
            if segment_col == 'segment':
                models_copy['Segment'] = models_copy['segment'].str.upper()
                merge_cols = ['Segment', 'node']
            else:
                merge_cols = ['Segment', 'node']

            models_merged = models_copy.merge(df, on=merge_cols, how='left')
            return models_merged
        except Exception as e:
            print(f"Error loading tree data: {e}")
            print(f"Available columns in models_df: {list(models_df.columns)}")
            return models_df

    def extract_time_from_node(node_name: str) -> float:
        """
        Placeholder function - time should be loaded via merge_time() function.
        Returns NaN to indicate missing time data.
        """
        import numpy as np
        return np.nan  # Return NaN instead of default year
    return load_datasets_from_directory, merge_time


@app.cell
def _(mo):
    mo.md(r"""## Apply Custom LOESS to All Segments and Datasets""")
    return


@app.cell
def _(mo):
    mo.md(r"""## Load and Process Data""")
    return


@app.cell
def _(load_datasets_from_directory):
    # Load all datasets from the Log_Likelihoods directory
    base_path = "../ESM2_Validation_Pipeline/results/Log_Likelihoods/"
    datasets_df = load_datasets_from_directory(base_path)

    print(f"Loaded {len(datasets_df)} records from {datasets_df['source_file'].nunique()} files")
    print(f"Segments: {sorted(datasets_df['segment'].unique())}")
    print(f"Model configurations: {sorted(datasets_df['model_config'].unique())}")
    print(f"Training datasets: {sorted(datasets_df['training_dataset'].unique())}")
    return (datasets_df,)


@app.cell
def _(datasets_df, merge_time):
    # Merge time data from tree JSON files
    datasets_with_time = merge_time(datasets_df, tree_dir="h3n2")

    print(f"Added time data. Records with time: {datasets_with_time['time'].notna().sum()}")
    print(f"Records without time: {datasets_with_time['time'].isna().sum()}")
    print(f"Time range: {datasets_with_time['time'].min()} - {datasets_with_time['time'].max()}")
    return (datasets_with_time,)


@app.cell
def _(datasets_with_time):
    datasets_with_time
    return


@app.cell
def _(datasets_with_time):
    filtered_datasets_with_time = datasets_with_time[~datasets_with_time["sequence"].str.contains("-", na=False)]
    filtered_datasets_with_time
    return (filtered_datasets_with_time,)


@app.cell
def apply_loess(filtered_datasets_with_time, loess_1d, np, os, pd):
    def apply_loess_for_model(model_type):
        """Apply LOESS correction with model-specific caching."""
        def apply_custom_loess_to_group(group_df, x_col="time", y_col="log_likelihood", degree=2, frac=0.15):
            """Apply custom LOESS to a single group."""
            if len(group_df) < 3:  # Need at least 3 points for LOESS
                group_df = group_df.copy()
                group_df['loess_trend'] = group_df[y_col]
                group_df['loess_weights'] = 1.0
                group_df['corrected_log_likelihood'] = 0.0
                return group_df

            # Sort by time
            group_df = group_df.sort_values(x_col).copy()

            x = group_df[x_col].values
            y = group_df[y_col].values

            # Apply custom LOESS function
            x_smooth, y_smooth, weights = loess_1d(
                x=x, 
                y=y, 
                xnew=x, 
                degree=degree, 
                frac=frac
            )

            # Add results to dataframe
            group_df.loc[:, 'loess_trend'] = y_smooth
            group_df.loc[:, 'loess_weights'] = weights
            group_df.loc[:, 'corrected_log_likelihood'] = y - y_smooth

            return group_df

        # Filter data for specific model type
        data_to_process = filtered_datasets_with_time[filtered_datasets_with_time['model_type'] == model_type]
        cache_suffix = f"_{model_type.replace('esm2_t33_650M_UR50D', '650M').replace('esm2_t36_3B_UR50D', '3B')}"
        print(f"Applying custom LOESS correction to {model_type} model...")

        # Define Parquet cache file path with model-specific suffix
        cache_file = f"Dataframes/datasets_with_loess{cache_suffix}_cache.parquet"

        # Check if cached results exist
        if os.path.exists(cache_file):
            print(f"Loading cached LOESS results from {cache_file}...")
            datasets_with_loess = pd.read_parquet(cache_file)
            print(f"Loaded {len(datasets_with_loess)} records from cache")
            print(f"Records with LOESS correction: {datasets_with_loess['corrected_log_likelihood'].notna().sum()}")
            print(f"Records without time data: {datasets_with_loess['time'].isna().sum()}")
        else:
            print("No cache found, computing LOESS correction...")

            # Filter for records with valid time data
            valid_data = data_to_process[data_to_process['time'].notna()].copy()

            processed_groups = []
            group_count = 0

            for (segment, model_config, training_dataset, model_type_group), group in valid_data.groupby(["segment", "model_config", "training_dataset", "model_type"], sort=False):
                group_count += 1
                if group_count % 5 == 0:
                    print(f"Processing group {group_count}: {segment}_{model_config}_{training_dataset}_{model_type_group}")

                processed_group = apply_custom_loess_to_group(group)
                processed_groups.append(processed_group)

            # Combine all processed groups
            if processed_groups:
                datasets_with_loess = pd.concat(processed_groups, ignore_index=True)

                # Merge back with original data (for records without time data)
                no_time_data = data_to_process[data_to_process['time'].isna()].copy()
                if len(no_time_data) > 0:
                    no_time_data['loess_trend'] = np.nan
                    no_time_data['loess_weights'] = np.nan  
                    no_time_data['corrected_log_likelihood'] = np.nan
                    datasets_with_loess = pd.concat([datasets_with_loess, no_time_data], ignore_index=True)
            else:
                datasets_with_loess = data_to_process.copy()
                datasets_with_loess['loess_trend'] = np.nan
                datasets_with_loess['loess_weights'] = np.nan
                datasets_with_loess['corrected_log_likelihood'] = np.nan

            print(f"Custom LOESS applied to {group_count} segment+model_config+training_dataset+model_type combinations")
            print(f"Records with LOESS correction: {datasets_with_loess['corrected_log_likelihood'].notna().sum()}")
            print(f"Records without time data: {datasets_with_loess['time'].isna().sum()}")

            # Save results to cache
            print(f"Saving LOESS results to cache file: {cache_file}")
            datasets_with_loess.to_parquet(cache_file, index=False)
            print("Cache saved successfully")
        return datasets_with_loess
    return (apply_loess_for_model,)


@app.cell
def _(mo):
    mo.md(r"""## Apply LOESS Correction to 650M Model Data""")
    return


@app.cell
def _(apply_loess_for_model):
    # Apply LOESS correction specifically to 650M model data
    datasets_with_loess_650m = apply_loess_for_model("esm2_t33_650M_UR50D")

    print(f"650M Model - Total records: {len(datasets_with_loess_650m)}")
    print(f"650M Model - Records with LOESS correction: {datasets_with_loess_650m['corrected_log_likelihood'].notna().sum()}")
    print(f"650M Model - Available model configurations: {sorted(datasets_with_loess_650m['model_config'].unique())}")
    print(f"650M Model - Available training datasets: {sorted(datasets_with_loess_650m['training_dataset'].unique())}")
    return (datasets_with_loess_650m,)


@app.cell
def _(datasets_with_loess_650m):
    datasets_with_loess_650m
    return


@app.cell
def _(mo):
    mo.md(r"""## Plotting Functions""")
    return


@app.cell
def _(ScalarFormatter, colorsys, mpl, np, plt, sns):
    # Set up plotting style
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

    def darken_color(rgb, factor=0.7):
        """Darken a color by the specified factor."""
        h, l, s = colorsys.rgb_to_hls(*rgb)
        r, g, b = colorsys.hls_to_rgb(h, max(0, l * factor), s)
        return (r, g, b, 1.0)

    def remove_outliers_simple(df, column, threshold=200):
        """Remove outliers that are more than threshold away from the mean."""
        mean_val = df[column].mean()
        return df[np.abs(df[column] - mean_val) <= threshold]

    def plot_segment_esm_score(ax, df, segment, title_prefix, use_loess=False, 
                             remove_outliers_flag=False, outlier_threshold=200):
        """Plot ESM score vs time for a single segment."""
        if use_loess:
            ll_col = "corrected_log_likelihood"
            title_suffix = "LOESS Corrected"
        else:
            ll_col = "log_likelihood"
            title_suffix = "Raw"

        # Filter for the specific segment
        df_segment = df[df['segment'].str.upper() == segment.upper()].copy()

        if len(df_segment) == 0:
            ax.text(0.5, 0.5, f'No data for {segment}', 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title(f"{segment.upper()} • {title_prefix} ({title_suffix})")
            return ax

        # Remove outliers if requested
        original_count = len(df_segment)
        if remove_outliers_flag and len(df_segment) > 0:
            df_segment = remove_outliers_simple(df_segment, ll_col, threshold=outlier_threshold)
            removed_count = original_count - len(df_segment)
            if removed_count > 0:
                title_suffix += f" (Outliers: -{removed_count})"

        # Create color mapping based on log likelihood values
        norm = plt.Normalize(df_segment[ll_col].min(), df_segment[ll_col].max())
        cmap = plt.get_cmap("viridis")
        colors = cmap(norm(df_segment[ll_col]))
        edgecolors = [darken_color(c[:3], factor=0.7) for c in colors]

        # Main scatter plot
        ax.scatter(
            df_segment["time"],
            df_segment[ll_col],
            c=colors,
            edgecolors=edgecolors,
            linewidths=0.5,
            alpha=0.7,
            zorder=1
        )

        # Highlight high frequency nodes if available
        if 'max_frequency' in df_segment.columns:
            high_freq_df = (
                df_segment[
                    (df_segment["max_frequency"] > 1) &
                    (df_segment["node"].str.contains("NODE_", na=False))
                ]
                .sort_values("time")
            )
            if len(high_freq_df) > 0:
                ax.plot(
                    high_freq_df["time"],
                    high_freq_df[ll_col],
                    linestyle='-',
                    color='black',
                    linewidth=2,
                    alpha=0.8,
                    label='Max Freq > 1 & NODE_',
                    zorder=2
                )

        # Styling
        ax.yaxis.offsetText.set_visible(False)
        ax.set_title(f"{segment.upper()} • {title_prefix} ({title_suffix})", fontsize=10)
        ax.set_ylabel("ESM Score", fontsize=8)
        ax.grid(True, color='lightgray', linestyle='-', linewidth=0.75)
        ax.spines[['right', 'top']].set_visible(False)
        ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        ax.ticklabel_format(style='plain', axis='x')
        ax.set_xlim(1965, 2025)

        # Set y-axis limits with padding
        y_min, y_max = df_segment[ll_col].min(), df_segment[ll_col].max()
        pad = (y_max - y_min) * 0.05 if y_max != y_min else 1.0
        ax.set_ylim(y_min - pad, y_max + pad)

        return ax

    def create_model_comparison_plots(df, model_config, training_dataset, model_type, 
                                     remove_outliers_flag=False, outlier_threshold=200):
        """Create multipanel plots comparing raw and LOESS corrected ESM scores for all segments in a model configuration."""
        # Filter data for this specific model configuration
        model_df = df[
            (df['model_config'] == model_config) & 
            (df['training_dataset'] == training_dataset) & 
            (df['model_type'] == model_type) &
            (df['time'].notna())  # Only include records with time data
        ].copy()

        if len(model_df) == 0:
            print(f"No data found for {model_config} + {training_dataset} + {model_type}")
            return None

        # Get unique segments
        segments = sorted(model_df['segment'].unique())
        n_segments = len(segments)

        if n_segments == 0:
            print(f"No segments found for {model_config} + {training_dataset} + {model_type}")
            return None

        # Create figure with 2 columns (raw and LOESS) and n_segments rows
        fig, axes = plt.subplots(n_segments, 2, figsize=(15, 4*n_segments), sharex=True)

        # Handle case with only one segment
        if n_segments == 1:
            axes = axes.reshape(1, -1)

        # Create title for the entire figure
        if model_config == "Base":
            main_title = f"{model_type} Base Model"
        else:
            main_title = f"{model_type} Fine-Tuned on {training_dataset}"

        if remove_outliers_flag:
            main_title += f" (Outliers >±{outlier_threshold} from mean removed)"

        fig.suptitle(main_title, fontsize=16, y=0.98)

        # Plot each segment
        for i, segment in enumerate(segments):
            # Raw log likelihood (left column)
            plot_segment_esm_score(axes[i, 0], model_df, segment, 
                                 f"{model_type}", use_loess=False,
                                 remove_outliers_flag=remove_outliers_flag,
                                 outlier_threshold=outlier_threshold)

            # LOESS corrected (right column)
            plot_segment_esm_score(axes[i, 1], model_df, segment, 
                                 f"{model_type}", use_loess=True,
                                 remove_outliers_flag=remove_outliers_flag,
                                 outlier_threshold=outlier_threshold)

            # Add x-axis labels only to bottom row
            if i == n_segments - 1:
                axes[i, 0].set_xlabel("Year", fontsize=10)
                axes[i, 1].set_xlabel("Year", fontsize=10)

        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        return fig
    return (create_model_comparison_plots,)


@app.cell
def _(mo):
    mo.md(r"""## Data Exploration""")
    return


@app.cell
def _(mo):
    mo.md(r"""### 650M Model Time Series Plots (Pre vs Post LOESS)""")
    return


@app.cell
def _(mo):
    mo.md(r"""### Generate plots for all model configurations""")
    return


@app.cell
def _(datasets_with_loess_650m):
    # Get all unique model configurations for 650M model
    model_configs_650m = []

    for (model_config_650m, training_dataset_650m, model_type_650m), group_650m in datasets_with_loess_650m.groupby(
        ['model_config', 'training_dataset', 'model_type']
    ):
        if group_650m['time'].notna().sum() > 0:  # Only include configs with time data
            model_configs_650m.append((model_config_650m, training_dataset_650m, model_type_650m))

    print(f"Found {len(model_configs_650m)} 650M model configurations with time data:")
    for config_650m in model_configs_650m:
        print(f"  - {config_650m[0]} + {config_650m[1]} + {config_650m[2]}")
    return (model_configs_650m,)


@app.cell
def _(
    create_model_comparison_plots,
    datasets_with_loess_650m,
    model_configs_650m,
):
    def generate_650m_base_plots():
        # Generate plots for 650M base models
        print("Generating plots for 650M Base models...")
        base_configs_650m = [config for config in model_configs_650m if config[0] == "Base"]

        base_model_figures_650m = []

        for model_config_650m_base, training_dataset_650m_base, model_type_650m_base in base_configs_650m:
            print(f"\nCreating plot for: {model_config_650m_base} + {training_dataset_650m_base} + {model_type_650m_base}")
            fig_1 = create_model_comparison_plots(datasets_with_loess_650m, model_config_650m_base, training_dataset_650m_base, model_type_650m_base)

            if fig_1:
                base_model_figures_650m.append(fig_1)

        return base_model_figures_650m

    base_model_figures_650m = generate_650m_base_plots()
    return (base_model_figures_650m,)


@app.cell
def _(base_model_figures_650m):
    base_model_figures_650m
    return


@app.cell
def _(
    create_model_comparison_plots,
    datasets_with_loess_650m,
    model_configs_650m,
):
    def generate_650m_specific_dataset_plots(training_dataset_name, remove_outliers_flag=False, 
                                      outlier_threshold=200):
        """Generate plots for a specific training dataset using 650M model."""
        fine_tune_configs_650m = [config for config in model_configs_650m 
                           if config[0] == "Fine_Tune" and config[1] == training_dataset_name]

        print(f"Generating 650M model plots for dataset: {training_dataset_name}")
        if remove_outliers_flag:
            print(f"Outlier removal enabled: removing points >±{outlier_threshold} from mean")

        figures = []
        for model_config_650m_ft, training_dataset_650m_ft, model_type_650m_ft in fine_tune_configs_650m:
            print(f"\nCreating plot for: {model_config_650m_ft} + {training_dataset_650m_ft} + {model_type_650m_ft}")
            fig = create_model_comparison_plots(datasets_with_loess_650m, model_config_650m_ft, training_dataset_650m_ft, model_type_650m_ft,
                                              remove_outliers_flag=remove_outliers_flag,
                                              outlier_threshold=outlier_threshold)
            if fig:
                figures.append(fig)

        return figures
    return (generate_650m_specific_dataset_plots,)


@app.cell
def _(generate_650m_specific_dataset_plots):
    # Generate plots for H3N2 dataset using 650M model
    h3n2_figures_650m = generate_650m_specific_dataset_plots("H3N2_Dataset_1965_Full")
    h3n2_figures_650m
    return


@app.cell
def _(generate_650m_specific_dataset_plots):
    # Generate 650M model plots for all individual datasets
    esm_dataset_figures_650m = generate_650m_specific_dataset_plots("ESM_1965_Full")
    pan_flu_figures_650m = generate_650m_specific_dataset_plots("pan_flu_1965_Full") 
    mix_ab_figures_650m = generate_650m_specific_dataset_plots("mix_AB_50_ESM_1965_Full_50_H3N2_Dataset_1965_Full")
    mix_ac_figures_650m = generate_650m_specific_dataset_plots("mix_AC_50_ESM_1965_Full_50_pan_flu_1965_Full")
    mix_bc_figures_650m = generate_650m_specific_dataset_plots("mix_BC_50_H3N2_Dataset_1965_Full_50_pan_flu_1965_Full")
    mix_abc_figures_650m = generate_650m_specific_dataset_plots("mix_ABC_33_ESM_1965_Full_33_H3N2_Dataset_1965_Full_33_pan_flu_1965_Full")
    return (
        esm_dataset_figures_650m,
        mix_ab_figures_650m,
        mix_abc_figures_650m,
        mix_ac_figures_650m,
        mix_bc_figures_650m,
        pan_flu_figures_650m,
    )


@app.cell
def _(esm_dataset_figures_650m):
    esm_dataset_figures_650m
    return


@app.cell
def _(pan_flu_figures_650m):
    pan_flu_figures_650m
    return


@app.cell
def _(mix_ab_figures_650m):
    mix_ab_figures_650m
    return


@app.cell
def _(mix_ac_figures_650m):
    mix_ac_figures_650m
    return


@app.cell
def _(mix_bc_figures_650m):
    mix_bc_figures_650m
    return


@app.cell
def _(mix_abc_figures_650m):
    mix_abc_figures_650m
    return


@app.cell
def _(mo):
    mo.md(r"""## 650M Model Analysis""")
    return


@app.cell
def _(mo):
    mo.md(r"""### Training Dataset Spearman Correlation Comparison (650M Model)""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    Compare Spearman correlations between log likelihood and maximum frequency 
    across different training datasets for the 650M model using LOESS-corrected data.

    This analysis compares all models fine-tuned on different training datasets rather than 
    comparing base vs fine-tune models.
    """
    )
    return


@app.cell
def _(datasets_with_loess_650m):
    # Filter for 650M model and Fine_Tune configuration only
    model_650m_ft = datasets_with_loess_650m[
        (datasets_with_loess_650m['model_type'] == 'esm2_t33_650M_UR50D') & 
        (datasets_with_loess_650m['model_config'] == 'Fine_Tune') &
        (datasets_with_loess_650m['training_dataset'] != 'None') &  # Exclude base model (no training dataset)
        (datasets_with_loess_650m['corrected_log_likelihood'].notna())  # Only include records with LOESS correction
    ].copy()

    print(f"650M Model - Filtered data shape: {model_650m_ft.shape}")
    print(f"650M Model - Available training datasets: {sorted(model_650m_ft['training_dataset'].unique())}")
    print(f"650M Model - Available segments: {sorted(model_650m_ft['segment'].unique())}")

    # Count records per training dataset
    training_dataset_counts_650m = model_650m_ft['training_dataset'].value_counts()
    print(f"\n650M Model - Records per training dataset:")
    for dataset_key_650m, count_650m in training_dataset_counts_650m.items():
        print(f"  {dataset_key_650m}: {count_650m}")
    return (model_650m_ft,)


@app.cell
def _(pd, spearmanr):
    def calculate_spearman_by_training_dataset(df):
        """Calculate Spearman correlation using LOESS corrected log likelihood."""
        results = []

        ll_col = 'corrected_log_likelihood'

        for dataset_name in df['training_dataset'].unique():
            dataset_df = df[df['training_dataset'] == dataset_name]

            for segment_name in dataset_df['segment'].unique():
                seg_df = dataset_df[dataset_df['segment'] == segment_name]

                seg_df_clean = seg_df[~seg_df['sequence'].str.contains('-', na=False)]

                if len(seg_df_clean) < 3:
                    continue

                try:
                    corr, p_value = spearmanr(seg_df_clean['max_frequency'], seg_df_clean[ll_col])

                    results.append({
                        'training_dataset': dataset_name,
                        'segment': segment_name,
                        'spearman_correlation': corr,
                        'p_value': p_value,
                        'n_points': len(seg_df_clean),
                        'correlation_type': 'LOESS_Corrected'
                    })
                except Exception as e:
                    print(f"Error calculating correlation for {dataset_name} - {segment_name}: {e}")
                    continue

        return pd.DataFrame(results)
    return (calculate_spearman_by_training_dataset,)


@app.cell
def _(calculate_spearman_by_training_dataset, model_650m_ft, pd):
    # Split 650M data by time period
    model_650m_ft_training = model_650m_ft[model_650m_ft['time'] < 2000].copy()
    model_650m_ft_testing = model_650m_ft[model_650m_ft['time'] >= 2000].copy()

    print(f"650M Model:")
    print(f"  Training period data (before 2000): {model_650m_ft_training.shape[0]} records")
    print(f"  Testing period data (2000+): {model_650m_ft_testing.shape[0]} records")

    # Calculate correlations for each period - 650M model
    spearman_results_training_650m = calculate_spearman_by_training_dataset(model_650m_ft_training)
    spearman_results_testing_650m = calculate_spearman_by_training_dataset(model_650m_ft_testing)

    # Add time period labels
    spearman_results_training_650m['time_period'] = 'Training (< 2000)'
    spearman_results_testing_650m['time_period'] = 'Testing (≥ 2000)'

    # Combine results for 650M model
    spearman_results_combined_650m = pd.concat([spearman_results_training_650m, spearman_results_testing_650m], ignore_index=True)

    print(f"\n650M Training period correlations: {spearman_results_training_650m.shape[0]} dataset-segment combinations")
    print(f"650M Testing period correlations: {spearman_results_testing_650m.shape[0]} dataset-segment combinations")
    return (
        spearman_results_combined_650m,
        spearman_results_testing_650m,
        spearman_results_training_650m,
    )


@app.cell
def _(np, pd, plt, sns, spearmanr):
    def create_time_period_comparison_plot(spearman_df, title="Spearman Correlation by Training Dataset and Time Period"):
        training_dataset_mapping = {
            'ESM_1965_Full': 'ESM Only',
            'H3N2_Dataset_1965_Full': 'H3N2 Only', 
            'pan_flu_1965_Full': 'Pan-Flu Only',
            'mix_AB_50_ESM_1965_Full_50_H3N2_Dataset_1965_Full': 'ESM + H3N2 (50/50)',
            'mix_AC_50_ESM_1965_Full_50_pan_flu_1965_Full': 'ESM + Pan-Flu (50/50)',
            'mix_BC_50_H3N2_Dataset_1965_Full_50_pan_flu_1965_Full': 'H3N2 + Pan-Flu (50/50)',
            'mix_ABC_33_ESM_1965_Full_33_H3N2_Dataset_1965_Full_33_pan_flu_1965_Full': 'ESM + H3N2 + Pan-Flu (33/33/33)'
        }

        spearman_plot_df = spearman_df.copy()
        spearman_plot_df['dataset_label'] = spearman_plot_df['training_dataset'].map(
            lambda x: training_dataset_mapping.get(x, x)
        )

        fig, ax = plt.subplots(figsize=(18, 10))

        sns.barplot(
            data=spearman_plot_df,
            x='segment',
            y='spearman_correlation',
            hue='time_period',
            ax=ax,
            palette=['#1f77b4', '#ff7f0e']  # Blue for training, orange for testing
        )

        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Segment', fontsize=12, fontweight='bold')
        ax.set_ylabel('Spearman Correlation Coefficient', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        plt.xticks(rotation=45)
        ax.legend(title='Time Period', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)

        plt.tight_layout()
        return fig

    def create_time_period_summary_plot(spearman_df, title="Average Spearman Correlation by Training Dataset and Time Period"):
        # Calculate averages by both training dataset and time period
        summary_time_correlations = spearman_df.groupby(['training_dataset', 'time_period'])['spearman_correlation'].agg(['mean', 'std']).reset_index()

        time_summary_dataset_mapping = {
            'ESM_1965_Full': 'ESM Only',
            'H3N2_Dataset_1965_Full': 'H3N2 Only', 
            'pan_flu_1965_Full': 'Pan-Flu Only',
            'mix_AB_50_ESM_1965_Full_50_H3N2_Dataset_1965_Full': 'ESM + H3N2',
            'mix_AC_50_ESM_1965_Full_50_pan_flu_1965_Full': 'ESM + Pan-Flu',
            'mix_BC_50_H3N2_Dataset_1965_Full_50_pan_flu_1965_Full': 'H3N2 + Pan-Flu',
            'mix_ABC_33_ESM_1965_Full_33_H3N2_Dataset_1965_Full_33_pan_flu_1965_Full': 'ESM + H3N2 + Pan-Flu'
        }

        summary_time_correlations['dataset_label'] = summary_time_correlations['training_dataset'].map(
            lambda x: time_summary_dataset_mapping.get(x, x)
        )

        # Create grouped bar plot
        fig, ax = plt.subplots(figsize=(14, 8))

        sns.barplot(
            data=summary_time_correlations,
            x='dataset_label',
            y='mean',
            hue='time_period',
            hue_order=['Training (< 2000)', 'Testing (≥ 2000)'],  # Training first, then testing
            ax=ax,
            palette=['#f4d35e', '#890304'],  # Tan for training, red for testing
            capsize=0.1
        )

        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Training Dataset', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average Spearman Correlation Coefficient', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)

        plt.xticks(rotation=45, ha='right')
        ax.legend(title='Time Period')

        plt.tight_layout()
        return fig

    def create_training_dataset_comparison_plot(spearman_df, title="Spearman Correlation by Training Dataset"):
        training_dataset_mapping = {
            'ESM_1965_Full': 'ESM Only',
            'H3N2_Dataset_1965_Full': 'H3N2 Only', 
            'pan_flu_1965_Full': 'Pan-Flu Only',
            'mix_AB_50_ESM_1965_Full_50_H3N2_Dataset_1965_Full': 'ESM + H3N2 (50/50)',
            'mix_AC_50_ESM_1965_Full_50_pan_flu_1965_Full': 'ESM + Pan-Flu (50/50)',
            'mix_BC_50_H3N2_Dataset_1965_Full_50_pan_flu_1965_Full': 'H3N2 + Pan-Flu (50/50)',
            'mix_ABC_33_ESM_1965_Full_33_H3N2_Dataset_1965_Full_33_pan_flu_1965_Full': 'ESM + H3N2 + Pan-Flu (33/33/33)'
        }

        spearman_plot_df = spearman_df.copy()
        spearman_plot_df['dataset_label'] = spearman_plot_df['training_dataset'].map(
            lambda x: training_dataset_mapping.get(x, x)
        )

        fig, ax = plt.subplots(figsize=(16, 8))

        sns.barplot(
            data=spearman_plot_df,
            x='segment',
            y='spearman_correlation',
            hue='dataset_label',
            ax=ax,
            palette='Set2'
        )

        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Segment', fontsize=12, fontweight='bold')
        ax.set_ylabel('Spearman Correlation Coefficient', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        plt.xticks(rotation=45)
        ax.legend(title='Training Dataset', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)

        plt.tight_layout()
        return fig

    def create_training_dataset_heatmap(spearman_df, title="Spearman Correlation Heatmap"):
        heatmap_pivot_data = spearman_df.pivot(index='training_dataset', columns='segment', values='spearman_correlation')

        heatmap_dataset_mapping = {
            'ESM_1965_Full': 'ESM Only',
            'H3N2_Dataset_1965_Full': 'H3N2 Only', 
            'pan_flu_1965_Full': 'Pan-Flu Only',
            'mix_AB_50_ESM_1965_Full_50_H3N2_Dataset_1965_Full': 'ESM + H3N2',
            'mix_AC_50_ESM_1965_Full_50_pan_flu_1965_Full': 'ESM + Pan-Flu',
            'mix_BC_50_H3N2_Dataset_1965_Full_50_pan_flu_1965_Full': 'H3N2 + Pan-Flu',
            'mix_ABC_33_ESM_1965_Full_33_H3N2_Dataset_1965_Full_33_pan_flu_1965_Full': 'ESM + H3N2 + Pan-Flu'
        }

        heatmap_pivot_data.index = heatmap_pivot_data.index.map(lambda x: heatmap_dataset_mapping.get(x, x))

        fig, ax = plt.subplots(figsize=(12, 8))

        sns.heatmap(
            heatmap_pivot_data,
            annot=True,
            cmap='RdBu_r',
            center=0,
            fmt='.3f',
            ax=ax,
            cbar_kws={'label': 'Spearman Correlation'}
        )

        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Segment', fontsize=12, fontweight='bold')
        ax.set_ylabel('Training Dataset', fontsize=12, fontweight='bold')

        plt.tight_layout()
        return fig

    def create_summary_comparison_plot(spearman_df, title="Average Spearman Correlation by Training Dataset"):
        summary_avg_correlations = spearman_df.groupby('training_dataset')['spearman_correlation'].agg(['mean', 'std']).reset_index()

        summary_dataset_mapping = {
            'ESM_1965_Full': 'ESM Only',
            'H3N2_Dataset_1965_Full': 'H3N2 Only', 
            'pan_flu_1965_Full': 'Pan-Flu Only',
            'mix_AB_50_ESM_1965_Full_50_H3N2_Dataset_1965_Full': 'ESM + H3N2',
            'mix_AC_50_ESM_1965_Full_50_pan_flu_1965_Full': 'ESM + Pan-Flu',
            'mix_BC_50_H3N2_Dataset_1965_Full_50_pan_flu_1965_Full': 'H3N2 + Pan-Flu',
            'mix_ABC_33_ESM_1965_Full_33_H3N2_Dataset_1965_Full_33_pan_flu_1965_Full': 'ESM + H3N2 + Pan-Flu'
        }

        summary_avg_correlations['dataset_label'] = summary_avg_correlations['training_dataset'].map(
            lambda x: summary_dataset_mapping.get(x, x)
        )

        summary_avg_correlations = summary_avg_correlations.sort_values('mean', ascending=True)

        fig, ax = plt.subplots(figsize=(12, 8))

        bars = ax.barh(
            summary_avg_correlations['dataset_label'],
            summary_avg_correlations['mean'],
            xerr=summary_avg_correlations['std'],
            capsize=5,
            color='skyblue',
            edgecolor='navy',
            alpha=0.7
        )

        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Average Spearman Correlation Coefficient', fontsize=12, fontweight='bold')
        ax.set_ylabel('Training Dataset', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)

        for i, (bar, val) in enumerate(zip(bars, summary_avg_correlations['mean'])):
            ax.text(val + 0.01 if val >= 0 else val - 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{val:.3f}', ha='left' if val >= 0 else 'right', va='center', fontweight='bold')

        plt.tight_layout()
        return fig

    def create_spearman_segment_comparison_plot(spearman_df, title="Spearman Correlation by Training Dataset and Segment"):
        """
        Create a plot comparing Spearman correlation coefficients across training datasets,
        showing each segment as a separate bar similar to create_time_period_summary_plot.
        """
        # Dataset mapping for cleaner labels
        dataset_mapping = {
            'ESM_1965_Full': 'ESM Only',
            'H3N2_Dataset_1965_Full': 'H3N2 Only', 
            'pan_flu_1965_Full': 'Pan-Flu Only',
            'mix_AB_50_ESM_1965_Full_50_H3N2_Dataset_1965_Full': 'ESM + H3N2',
            'mix_AC_50_ESM_1965_Full_50_pan_flu_1965_Full': 'ESM + Pan-Flu',
            'mix_BC_50_H3N2_Dataset_1965_Full_50_pan_flu_1965_Full': 'H3N2 + Pan-Flu',
            'mix_ABC_33_ESM_1965_Full_33_H3N2_Dataset_1965_Full_33_pan_flu_1965_Full': 'ESM + H3N2 + Pan-Flu'
        }

        # Add dataset labels
        plot_df = spearman_df.copy()
        plot_df['dataset_label'] = plot_df['training_dataset'].map(
            lambda x: dataset_mapping.get(x, x)
        )

        # Create the plot
        fig, ax = plt.subplots(figsize=(16, 10))

        # Create grouped bar plot with segments as hue
        custom_palette = ['#B84432', '#F8C3A0', '#D8AD4B', '#1A5B23', '#DEE8E5', '#3F5B8D', '#C9A8CD', '#211603']
        sns.barplot(
            data=plot_df,
            x='dataset_label',
            y='spearman_correlation',
            hue='segment',
            ax=ax,
            palette=custom_palette
        )

        # Styling
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Training Dataset', fontsize=12, fontweight='bold')
        ax.set_ylabel('Spearman Correlation Coefficient', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')

        # Position legend outside the plot
        ax.legend(title='Segment', bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        return fig

    def create_spearman_segment_detailed_plot(spearman_df, title="Spearman Correlation with Individual Segments"):
        """
        Create a plot similar to create_time_period_summary_plot but showing individual segments
        as separate bars along with training dataset grouping.
        """
        # Dataset mapping for cleaner labels
        dataset_mapping = {
            'ESM_1965_Full': 'ESM Only',
            'H3N2_Dataset_1965_Full': 'H3N2 Only', 
            'pan_flu_1965_Full': 'Pan-Flu Only',
            'mix_AB_50_ESM_1965_Full_50_H3N2_Dataset_1965_Full': 'ESM + H3N2',
            'mix_AC_50_ESM_1965_Full_50_pan_flu_1965_Full': 'ESM + Pan-Flu',
            'mix_BC_50_H3N2_Dataset_1965_Full_50_pan_flu_1965_Full': 'H3N2 + Pan-Flu',
            'mix_ABC_33_ESM_1965_Full_33_H3N2_Dataset_1965_Full_33_pan_flu_1965_Full': 'ESM + H3N2 + Pan-Flu'
        }

        # Add dataset labels
        plot_df = spearman_df.copy()
        plot_df['dataset_label'] = plot_df['training_dataset'].map(
            lambda x: dataset_mapping.get(x, x)
        )

        # Create the plot with segments on x-axis and training datasets as hue
        fig, ax = plt.subplots(figsize=(14, 8))

        sns.barplot(
            data=plot_df,
            x='segment',
            y='spearman_correlation',
            hue='dataset_label',
            ax=ax,
            palette='Set2'
        )

        # Styling similar to create_time_period_summary_plot
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Segment', fontsize=12, fontweight='bold')
        ax.set_ylabel('Spearman Correlation Coefficient', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)

        plt.xticks(rotation=45, ha='right')
        ax.legend(title='Training Dataset')

        plt.tight_layout()
        return fig
    def create_time_binned_spearman_line_plot(df, title="Time-Binned Spearman Correlation Analysis"):
        """
        Create a line plot showing LOESS corrected Spearman correlations across sliding time windows,
        similar to the one in Apply_LOESS_To_Previous_ESM.py
        """
        # Define time ranges (sliding windows)
        time_ranges = [
            (1970, 1990, "1980"),
            (1980, 2000, "1990"), 
            (1990, 2010, "2000"),
            (2000, 2020, "2010"),
            (2010, None, "2020"),
        ]

        results = []

        # Calculate spearman correlation for each time window and training dataset
        for (start, end, label) in time_ranges:
            if end is None:
                time_filtered_df = df[df['time'] >= start]
            else:
                time_filtered_df = df[(df['time'] >= start) & (df['time'] <= end)]

            if len(time_filtered_df) == 0:
                continue

            # Calculate correlations for each training dataset
            for training_dataset in df['training_dataset'].unique():
                dataset_df = time_filtered_df[time_filtered_df['training_dataset'] == training_dataset]

                if len(dataset_df) < 3:  # Need minimum points for correlation
                    continue

                # Calculate LOESS-corrected correlations only
                try:
                    if 'corrected_log_likelihood' in dataset_df.columns:
                        corr_loess, _ = spearmanr(dataset_df['max_frequency'], dataset_df['corrected_log_likelihood'])
                        results.append({
                            'Time_Range': label,
                            'Training_Dataset': training_dataset, 
                            'Spearman_Correlation': corr_loess,
                            'Model_Type': 'LOESS_Corrected',
                            'N_Points': len(dataset_df)
                        })
                except Exception as e:
                    continue

        if not results:
            print("No data available for time-binned analysis")
            return None

        results_df = pd.DataFrame(results)

        # Create dataset labels for cleaner display
        dataset_mapping = {
            'ESM_1965_Full': 'ESM Only',
            'H3N2_Dataset_1965_Full': 'H3N2 Only', 
            'pan_flu_1965_Full': 'Pan-Flu Only',
            'mix_AB_50_ESM_1965_Full_50_H3N2_Dataset_1965_Full': 'ESM + H3N2',
            'mix_AC_50_ESM_1965_Full_50_pan_flu_1965_Full': 'ESM + Pan-Flu',
            'mix_BC_50_H3N2_Dataset_1965_Full_50_pan_flu_1965_Full': 'H3N2 + Pan-Flu',
            'mix_ABC_33_ESM_1965_Full_33_H3N2_Dataset_1965_Full_33_pan_flu_1965_Full': 'ESM + H3N2 + Pan-Flu'
        }

        results_df['Dataset_Label'] = results_df['Training_Dataset'].map(
            lambda x: dataset_mapping.get(x, x)
        )

        # Create model labels (only LOESS corrected)
        results_df['Model'] = results_df['Dataset_Label']

        # Set up plotting style similar to Apply_LOESS file
        sns.set_style("whitegrid")
        custom_params = {"axes.spines.right": False, "axes.spines.top": False}
        sns.set_theme(style="ticks", rc=custom_params)

        fig, ax = plt.subplots(figsize=(12, 8))

        # Create line plot
        ax = sns.lineplot(
            data=results_df,
            x='Time_Range',
            y='Spearman_Correlation', 
            hue='Model',
            marker="o",
            legend=False,
            zorder=1,
            ax=ax,
            errorbar=None,
        )

        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel("Time Range (Center Year)", fontsize=12, fontweight='bold')
        ax.set_ylabel("Spearman Correlation Coefficient", fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)

        # Add labels at the end of each line similar to Apply_LOESS style
        label_positions = []
        for line, model in zip(ax.lines, results_df['Model'].unique()):
            y = line.get_ydata()[-1]
            x = line.get_xdata()[-1]

            if not np.isfinite(y) or not np.isfinite(x):
                continue

            # Adjust label positions to avoid overlap
            while any(abs(y - pos) < 0.025 for pos in label_positions):
                y += 0.01

            label_positions.append(y)

            ax.annotate(
                model,
                xy=(x, y),
                xytext=(5, 0),
                textcoords="offset points",
                color=line.get_color(),
                fontsize=10,
                weight='bold',
                ha='left',
                va='center',
                zorder=2,
            )

        plt.tight_layout()
        return fig
    return (
        create_spearman_segment_comparison_plot,
        create_spearman_segment_detailed_plot,
        create_time_binned_spearman_line_plot,
        create_time_period_summary_plot,
        create_training_dataset_heatmap,
    )


@app.cell
def _(create_time_period_summary_plot, spearman_results_combined_650m):
    # Create time period summary comparison for 650M model
    time_period_summary_fig_650m = create_time_period_summary_plot(
        spearman_results_combined_650m,
        "Average Spearman Correlation by Training Dataset - Training vs Testing Periods (650M Model)"
    )
    time_period_summary_fig_650m
    return


@app.cell
def _(create_training_dataset_heatmap, spearman_results_training_650m):
    # Create heatmap for 650M training period
    training_heatmap_fig_650m = create_training_dataset_heatmap(
        spearman_results_training_650m,
        "Spearman Correlation Heatmap - Training Period (< 2000) - 650M Model"
    )
    training_heatmap_fig_650m
    return


@app.cell
def _(create_training_dataset_heatmap, spearman_results_testing_650m):
    # Create heatmap for 650M testing period
    testing_heatmap_fig_650m = create_training_dataset_heatmap(
        spearman_results_testing_650m,
        "Spearman Correlation Heatmap - Testing Period (≥ 2000) - 650M Model"
    )
    testing_heatmap_fig_650m
    return


@app.cell
def _(create_spearman_segment_comparison_plot, spearman_results_training_650m):
    # Create segment comparison plot for 650M training period
    spearman_segment_training_fig_650m = create_spearman_segment_comparison_plot(
        spearman_results_training_650m,
        "Spearman Correlation by Training Dataset and Segment - Training Period (< 2000) - 650M Model"
    )
    spearman_segment_training_fig_650m
    return


@app.cell
def _(create_spearman_segment_comparison_plot, spearman_results_testing_650m):
    # Create segment comparison plot for 650M testing period
    spearman_segment_testing_fig_650m = create_spearman_segment_comparison_plot(
        spearman_results_testing_650m,
        "Spearman Correlation by Training Dataset and Segment - Testing Period (≥ 2000) - 650M Model"
    )
    spearman_segment_testing_fig_650m
    return


@app.cell
def _(create_spearman_segment_detailed_plot, spearman_results_training_650m):
    # Create detailed segment comparison plot for 650M training period
    spearman_segment_detailed_training_fig_650m = create_spearman_segment_detailed_plot(
        spearman_results_training_650m,
        "Spearman Correlation with Individual Segments by Training Dataset - Training Period (< 2000) - 650M Model"
    )
    spearman_segment_detailed_training_fig_650m
    return


@app.cell
def _(create_spearman_segment_detailed_plot, spearman_results_testing_650m):
    # Create detailed segment comparison plot for 650M testing period
    spearman_segment_detailed_testing_fig_650m = create_spearman_segment_detailed_plot(
        spearman_results_testing_650m,
        "Spearman Correlation with Individual Segments by Training Dataset - Testing Period (≥ 2000) - 650M Model"
    )
    spearman_segment_detailed_testing_fig_650m
    return


@app.cell
def _(create_time_binned_spearman_line_plot, model_650m_ft):
    # Create time-binned spearman correlation line plot (LOESS corrected only)
    time_binned_line_fig = create_time_binned_spearman_line_plot(
        model_650m_ft,
        "Time-Binned Spearman Correlation Analysis - 650M Fine-Tuned Models"
    )
    time_binned_line_fig
    return


@app.cell
def _(mo):
    mo.md(r"""## 3B Model Analysis""")
    return


@app.cell
def _(mo):
    mo.md(r"""### Apply LOESS Correction to 3B Model Data""")
    return


@app.cell
def _(apply_loess_for_model):
    # Apply LOESS correction specifically to 3B model data
    datasets_with_loess_3b = apply_loess_for_model("esm2_t36_3B_UR50D")

    print(f"3B Model - Total records: {len(datasets_with_loess_3b)}")
    print(f"3B Model - Records with LOESS correction: {datasets_with_loess_3b['corrected_log_likelihood'].notna().sum()}")
    print(f"3B Model - Available model configurations: {sorted(datasets_with_loess_3b['model_config'].unique())}")
    print(f"3B Model - Available training datasets: {sorted(datasets_with_loess_3b['training_dataset'].unique())}")
    return (datasets_with_loess_3b,)


@app.cell
def _(datasets_with_loess_3b):
    datasets_with_loess_3b
    return


@app.cell
def _(mo):
    mo.md(r"""### 3B Model Time Series Plots""")
    return


@app.cell
def _(datasets_with_loess_3b):
    # Get all unique model configurations for 3B model
    model_configs_3b = []

    for (model_config_3b, training_dataset_3b, model_type_3b), group_3b in datasets_with_loess_3b.groupby(
        ['model_config', 'training_dataset', 'model_type']
    ):
        if group_3b['time'].notna().sum() > 0:  # Only include configs with time data
            model_configs_3b.append((model_config_3b, training_dataset_3b, model_type_3b))

    print(f"Found {len(model_configs_3b)} 3B model configurations with time data:")
    for config_3b in model_configs_3b:
        print(f"  - {config_3b[0]} + {config_3b[1]} + {config_3b[2]}")
    return (model_configs_3b,)


@app.cell
def _(create_model_comparison_plots, datasets_with_loess_3b, model_configs_3b):
    def generate_3b_base_plots():
        # Generate plots for 3B base models
        print("Generating plots for 3B Base models...")
        base_configs_3b = [config for config in model_configs_3b if config[0] == "Base"]

        base_model_figures_3b = []

        for model_config_3b_base, training_dataset_3b_base, model_type_3b_base in base_configs_3b:
            print(f"\nCreating plot for: {model_config_3b_base} + {training_dataset_3b_base} + {model_type_3b_base}")
            fig_1 = create_model_comparison_plots(datasets_with_loess_3b, model_config_3b_base, training_dataset_3b_base, model_type_3b_base)

            if fig_1:
                base_model_figures_3b.append(fig_1)

        return base_model_figures_3b

    base_model_figures_3b = generate_3b_base_plots()
    return (base_model_figures_3b,)


@app.cell
def _(base_model_figures_3b):
    base_model_figures_3b
    return


@app.cell
def _(base_model_figures_3b):
    base_model_figures_3b
    return


@app.cell
def _(create_model_comparison_plots, datasets_with_loess_3b, model_configs_3b):
    def generate_3b_specific_dataset_plots(training_dataset_name, remove_outliers_flag=False, 
                                      outlier_threshold=200):
        """Generate plots for a specific training dataset using 3B model."""
        fine_tune_configs_3b = [config for config in model_configs_3b 
                           if config[0] == "Fine_Tune" and config[1] == training_dataset_name]

        print(f"Generating 3B model plots for dataset: {training_dataset_name}")
        if remove_outliers_flag:
            print(f"Outlier removal enabled: removing points >±{outlier_threshold} from mean")

        figures = []
        for model_config_3b_ft, training_dataset_3b_ft, model_type_3b_ft in fine_tune_configs_3b:
            print(f"\nCreating plot for: {model_config_3b_ft} + {training_dataset_3b_ft} + {model_type_3b_ft}")
            fig = create_model_comparison_plots(datasets_with_loess_3b, model_config_3b_ft, training_dataset_3b_ft, model_type_3b_ft,
                                              remove_outliers_flag=remove_outliers_flag,
                                              outlier_threshold=outlier_threshold)
            if fig:
                figures.append(fig)

        return figures

    # Generate 3B model plots for all individual datasets
    h3n2_figures_3b = generate_3b_specific_dataset_plots("H3N2_Dataset_1965_Full")
    esm_dataset_figures_3b = generate_3b_specific_dataset_plots("ESM_1965_Full")
    pan_flu_figures_3b = generate_3b_specific_dataset_plots("pan_flu_1965_Full") 
    mix_ab_figures_3b = generate_3b_specific_dataset_plots("mix_AB_50_ESM_1965_Full_50_H3N2_Dataset_1965_Full")
    mix_ac_figures_3b = generate_3b_specific_dataset_plots("mix_AC_50_ESM_1965_Full_50_pan_flu_1965_Full")
    mix_bc_figures_3b = generate_3b_specific_dataset_plots("mix_BC_50_H3N2_Dataset_1965_Full_50_pan_flu_1965_Full")
    mix_abc_figures_3b = generate_3b_specific_dataset_plots("mix_ABC_33_ESM_1965_Full_33_H3N2_Dataset_1965_Full_33_pan_flu_1965_Full")
    return (
        esm_dataset_figures_3b,
        h3n2_figures_3b,
        mix_ab_figures_3b,
        mix_abc_figures_3b,
        mix_ac_figures_3b,
        mix_bc_figures_3b,
        pan_flu_figures_3b,
    )


@app.cell
def _(h3n2_figures_3b):
    h3n2_figures_3b
    return


@app.cell
def _(esm_dataset_figures_3b):
    esm_dataset_figures_3b
    return


@app.cell
def _(pan_flu_figures_3b):
    pan_flu_figures_3b
    return


@app.cell
def _(mix_ab_figures_3b):
    mix_ab_figures_3b
    return


@app.cell
def _(mix_ac_figures_3b):
    mix_ac_figures_3b
    return


@app.cell
def _(mix_bc_figures_3b):
    mix_bc_figures_3b
    return


@app.cell
def _(mix_abc_figures_3b):
    mix_abc_figures_3b
    return


@app.cell
def _(mo):
    mo.md(r"""### 3B Model Spearman Correlation Analysis""")
    return


@app.cell
def _(datasets_with_loess_3b):
    # Filter for 3B model and Fine_Tune configuration only
    model_3b_ft = datasets_with_loess_3b[
        (datasets_with_loess_3b['model_type'] == 'esm2_t36_3B_UR50D') & 
        (datasets_with_loess_3b['model_config'] == 'Fine_Tune') &
        (datasets_with_loess_3b['training_dataset'] != 'None') &  # Exclude base model (no training dataset)
        (datasets_with_loess_3b['corrected_log_likelihood'].notna())  # Only include records with LOESS correction
    ].copy()

    print(f"3B Model - Filtered data shape: {model_3b_ft.shape}")
    print(f"3B Model - Available training datasets: {sorted(model_3b_ft['training_dataset'].unique())}")
    print(f"3B Model - Available segments: {sorted(model_3b_ft['segment'].unique())}")

    # Count records per training dataset
    training_dataset_counts_3b = model_3b_ft['training_dataset'].value_counts()
    print(f"\n3B Model - Records per training dataset:")
    for dataset_key_3b, count_3b in training_dataset_counts_3b.items():
        print(f"  {dataset_key_3b}: {count_3b}")
    return (model_3b_ft,)


@app.cell
def _(calculate_spearman_by_training_dataset, model_3b_ft, pd):
    # Split 3B data by time period
    model_3b_ft_training = model_3b_ft[model_3b_ft['time'] < 2000].copy()
    model_3b_ft_testing = model_3b_ft[model_3b_ft['time'] >= 2000].copy()

    print(f"3B Model:")
    print(f"  Training period data (before 2000): {model_3b_ft_training.shape[0]} records")
    print(f"  Testing period data (2000+): {model_3b_ft_testing.shape[0]} records")

    # Calculate correlations for each period - 3B model
    spearman_results_training_3b = calculate_spearman_by_training_dataset(model_3b_ft_training)
    spearman_results_testing_3b = calculate_spearman_by_training_dataset(model_3b_ft_testing)

    # Add time period labels
    spearman_results_training_3b['time_period'] = 'Training (< 2000)'
    spearman_results_testing_3b['time_period'] = 'Testing (≥ 2000)'

    # Combine results for 3B model
    spearman_results_combined_3b = pd.concat([spearman_results_training_3b, spearman_results_testing_3b], ignore_index=True)

    print(f"\n3B Training period correlations: {spearman_results_training_3b.shape[0]} dataset-segment combinations")
    print(f"3B Testing period correlations: {spearman_results_testing_3b.shape[0]} dataset-segment combinations")
    return (
        spearman_results_combined_3b,
        spearman_results_testing_3b,
        spearman_results_training_3b,
    )


@app.cell
def _(create_time_period_summary_plot, spearman_results_combined_3b):
    # Create time period summary comparison for 3B model
    time_period_summary_fig_3b = create_time_period_summary_plot(
        spearman_results_combined_3b,
        "Average Spearman Correlation by Training Dataset - Training vs Testing Periods (3B Model)"
    )
    time_period_summary_fig_3b
    return


@app.cell
def _(create_training_dataset_heatmap, spearman_results_training_3b):
    # Create heatmap for 3B training period
    training_heatmap_fig_3b = create_training_dataset_heatmap(
        spearman_results_training_3b,
        "Spearman Correlation Heatmap - Training Period (< 2000) - 3B Model"
    )
    training_heatmap_fig_3b
    return


@app.cell
def _(create_training_dataset_heatmap, spearman_results_testing_3b):
    # Create heatmap for 3B testing period
    testing_heatmap_fig_3b = create_training_dataset_heatmap(
        spearman_results_testing_3b,
        "Spearman Correlation Heatmap - Testing Period (≥ 2000) - 3B Model"
    )
    testing_heatmap_fig_3b
    return


@app.cell
def _(create_spearman_segment_comparison_plot, spearman_results_training_3b):
    # Create segment comparison plot for 3B training period
    spearman_segment_training_fig_3b = create_spearman_segment_comparison_plot(
        spearman_results_training_3b,
        "Spearman Correlation by Training Dataset and Segment - Training Period (< 2000) - 3B Model"
    )
    spearman_segment_training_fig_3b
    return


@app.cell
def _(create_spearman_segment_comparison_plot, spearman_results_testing_3b):
    # Create segment comparison plot for 3B testing period
    spearman_segment_testing_fig_3b = create_spearman_segment_comparison_plot(
        spearman_results_testing_3b,
        "Spearman Correlation by Training Dataset and Segment - Testing Period (≥ 2000) - 3B Model"
    )
    spearman_segment_testing_fig_3b
    return


@app.cell
def _(create_spearman_segment_detailed_plot, spearman_results_training_3b):
    # Create detailed segment comparison plot for 3B training period
    spearman_segment_detailed_training_fig_3b = create_spearman_segment_detailed_plot(
        spearman_results_training_3b,
        "Spearman Correlation with Individual Segments by Training Dataset - Training Period (< 2000) - 3B Model"
    )
    spearman_segment_detailed_training_fig_3b
    return


@app.cell
def _(create_spearman_segment_detailed_plot, spearman_results_testing_3b):
    # Create detailed segment comparison plot for 3B testing period
    spearman_segment_detailed_testing_fig_3b = create_spearman_segment_detailed_plot(
        spearman_results_testing_3b,
        "Spearman Correlation with Individual Segments by Training Dataset - Testing Period (≥ 2000) - 3B Model"
    )
    spearman_segment_detailed_testing_fig_3b
    return


@app.cell
def _(create_time_binned_spearman_line_plot, model_3b_ft):
    # Create time-binned spearman correlation line plot for 3B model (LOESS corrected only)
    time_binned_line_fig_3b = create_time_binned_spearman_line_plot(
        model_3b_ft,
        "Time-Binned Spearman Correlation Analysis - 3B Fine-Tuned Models"
    )
    time_binned_line_fig_3b
    return


if __name__ == "__main__":
    app.run()
