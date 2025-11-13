# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo",
#     "pandas==2.3.3",
#     "numpy==2.3.4",
#     "matplotlib==3.10.7",
#     "seaborn==0.13.2",
# ]
# ///

import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    return mo, np, pd, plt, sns


@app.cell
def _(mo, pd):
    # Load the data files
    dms_library = pd.read_csv("results/dms_library.csv")
    summary_avgprefs = pd.read_csv("summary_avgprefs.csv")

    mo.md(f"""
    # DMS Score Calculation

    Loaded data:
    - DMS Library: {len(dms_library)} sequences
    - Summary Avg Prefs: {len(summary_avgprefs)} positions
    """)
    return dms_library, summary_avgprefs


@app.cell
def _(mo):
    # Display sample of the library data
    mo.md("""
    ## DMS Library Sample
    """)
    return


@app.cell
def _(mo):
    # Display sample of the preferences data
    mo.md("""
    ## Summary Average Preferences Sample
    """)
    return


@app.cell
def _(mo, pd, summary_avgprefs):
    import glob
    import os

    # Create a lookup dictionary for DMS scores
    # Key: (position, amino_acid), Value: score
    dms_scores = {}

    # Get amino acid columns (skip 'site' and 'site_fix' columns)
    amino_acids = [col for col in summary_avgprefs.columns if col not in ['site', 'site_fix']]

    for _, row in summary_avgprefs.iterrows():
        position = row['site_fix']
        for aa in amino_acids:
            dms_scores[(position, aa)] = row[aa]

    # Load all log likelihood files
    ll_files = glob.glob("results/log_likelihoods/**/*.csv", recursive=True)

    ll_dataframes = []
    for file_path in ll_files:
        # Extract metadata from file path
        path_parts = file_path.split('/')
        model = None
        condition = None

        for part in path_parts:
            if 'esm2_t' in part:
                model = part
            if part in ['base', 'H3N2_HA_2009_Cutoff']:
                condition = 'base' if part == 'base' else 'fine_tuned'

        # Load the file
        df = pd.read_csv(file_path)
        df['model'] = model
        df['condition'] = condition
        df['file_path'] = file_path

        # Select only needed columns
        df_subset = df[['node', 'log_likelihood', 'model', 'condition']].copy()
        ll_dataframes.append(df_subset)

    # Combine all log likelihood dataframes
    combined_ll = pd.concat(ll_dataframes, ignore_index=True)

    mo.md(f"""
    Created DMS score lookup with {len(dms_scores)} entries for amino acids: {', '.join(amino_acids)}

    **Log Likelihood Data Loaded:**
    - Total files: {len(ll_files)}
    - Total records: {len(combined_ll)}
    - Models: {', '.join(combined_ll['model'].unique())}
    - Conditions: {', '.join(combined_ll['condition'].unique())}
    """)
    return combined_ll, dms_scores


@app.cell
def _(combined_ll, dms_library, dms_scores, mo, pd, summary_avgprefs):
    # Get amino acid columns for validation
    aa_cols = [col for col in summary_avgprefs.columns if col not in ['site', 'site_fix']]

    # Calculate DMS score for each sequence
    def calc_dms_score(sequence):
        total_score = 0
        for position, amino_acid in enumerate(sequence, start=1):
            # Only score amino acids that have DMS data
            if amino_acid in aa_cols:
                score = dms_scores.get((position, amino_acid), 0)
                total_score += score
        return total_score

    # Create scored library dataframe
    scored_library = dms_library.copy()
    scored_library['dms_score'] = scored_library['sequence'].apply(calc_dms_score)

    # Create comprehensive comparison dataframe by merging with log likelihood data
    comparison_df = pd.merge(
        scored_library[['node', 'dms_score']], 
        combined_ll, 
        on='node', 
        how='inner'
    )

    mo.md(f"""
    ## DMS Scores Calculated & Combined with Log Likelihoods

    **Combined Dataset:**
    - Total sequences with both DMS and LL data: {len(comparison_df)}
    - Unique sequences: {comparison_df['node'].nunique()}
    - Models compared: {', '.join(comparison_df['model'].unique())}
    - Conditions: {', '.join(comparison_df['condition'].unique())}
    """)
    return comparison_df, scored_library


@app.cell
def _(mo):
    # Display sample of combined data
    mo.md("""
    ## Combined Data Sample
    """)
    return


@app.cell
def _(comparison_df):
    # Show sample of combined dataframe
    comparison_df.head(10)
    return


@app.cell
def _(comparison_df, mo):
    # Create pivot table for easier comparison
    pivot_comparison = comparison_df.pivot_table(
        index='node', 
        columns=['model', 'condition'], 
        values='log_likelihood', 
        aggfunc='first'
    ).reset_index()

    # Flatten the MultiIndex columns
    if hasattr(pivot_comparison.columns, 'levels'):
        # Flatten MultiIndex columns by joining with underscore
        new_columns = []
        for col in pivot_comparison.columns:
            if isinstance(col, tuple):
                if col[0] == 'node':
                    new_columns.append('node')
                else:
                    # Join non-empty parts of the tuple
                    parts = [str(part) for part in col if str(part) != '']
                    new_columns.append('_'.join(parts))
            else:
                new_columns.append(str(col))
        pivot_comparison.columns = new_columns

    # Merge back with DMS scores
    final_comparison = pivot_comparison.merge(
        comparison_df[['node', 'dms_score']].drop_duplicates(), 
        on='node'
    )

    mo.md(f"""
    ## Comparison Analysis

    **Pivot Table Created:**
    - Shape: {final_comparison.shape}
    - Columns: {list(final_comparison.columns)}
    """)
    return (final_comparison,)


@app.cell
def _(final_comparison):
    # Show the final comparison table
    final_comparison.head()
    return


@app.cell
def _(final_comparison, mo, np, plt, sns):
    # Create 1x4 plot comparing DMS to LL scores
    sns.set_style("whitegrid")
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params)

    # Get the log likelihood columns (exclude node and dms_score)
    ll_columns = [col for col in final_comparison.columns if col not in ['node', 'dms_score']]

    # Create 1x4 subplot
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)

    for i, ll_col in enumerate(ll_columns):
        ax = axes[i]

        # Filter out any NaN values
        mask = final_comparison[ll_col].notna() & final_comparison['dms_score'].notna()
        x_data = final_comparison.loc[mask, ll_col]
        y_data = final_comparison.loc[mask, 'dms_score']

        # Create scatter plot
        ax.scatter(x_data, y_data, alpha=0.6, s=20)

        # Calculate and display correlation
        r_value = np.corrcoef(x_data, y_data)[0, 1]

        # Add trend line
        z = np.polyfit(x_data, y_data, 1)
        p = np.poly1d(z)
        ax.plot(x_data, p(x_data), "r--", alpha=0.8, linewidth=1)

        # Format title and labels
        title_parts = ll_col.split('_')
        if len(title_parts) >= 2:
            # Extract model size and format model name
            if '650M' in title_parts:
                model_name = "ESM2-650M"
            elif '3B' in title_parts:
                model_name = "ESM2-3B"
            else:
                model_name = title_parts[0].replace('esm2', 'ESM2')

            condition_type = title_parts[-1].replace('base', 'Base').replace('tuned', 'Fine Tuned')
            title = f"{model_name}\n{condition_type}"
        else:
            title = ll_col

        ax.set_title(title, fontsize=12, weight='bold')
        ax.set_xlabel("Log Likelihood", fontsize=10)

        # Add correlation text
        ax.text(0.05, 0.95, f'r = {r_value:.3f}', 
                transform=ax.transAxes, fontsize=10, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Set y-label only for the first subplot
    axes[0].set_ylabel("DMS Score", fontsize=12)

    plt.suptitle("DMS Score vs Log Likelihood Comparison", fontsize=14, weight='bold', y=1.05)
    plt.tight_layout()
    plt.show()

    mo.md("## DMS vs Log Likelihood Scatter Plots")
    return


@app.cell
def _(final_comparison, mo, scored_library):
    # Save results to CSV files
    output_path_dms = "results/dms_scored_sequences.csv"
    output_path_combined = "results/dms_ll_comparison.csv"

    # Save DMS scores (original functionality)
    scored_library[['node', 'dms_score']].to_csv(output_path_dms, index=False)

    # Save combined comparison data
    final_comparison.to_csv(output_path_combined, index=False)

    export_message = mo.md(f"""
    ## Export

    Results saved to:
    - DMS scores only: `{output_path_dms}`
    - Combined DMS & LL comparison: `{output_path_combined}`

    **Combined file contains:**
    - DMS scores for each sequence
    - Log likelihood values for all model/condition combinations
    - Ready for correlation analysis and visualization
    """)
    return


if __name__ == "__main__":
    app.run()
