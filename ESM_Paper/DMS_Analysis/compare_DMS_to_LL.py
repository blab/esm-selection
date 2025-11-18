# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo",
#     "pandas==2.3.3",
#     "numpy==2.3.4",
#     "matplotlib==3.10.7",
#     "seaborn==0.13.2",
#     "scipy==1.16.3",
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
    wolf_epitopes = pd.read_csv("wolf_epitope_sites.csv")

    mo.md(f"""
    # DMS Score Calculation

    Loaded data:
    - DMS Library: {len(dms_library)} sequences
    - Summary Avg Prefs: {len(summary_avgprefs)} positions
    - Wolf Epitope Sites: {len(wolf_epitopes)} positions
    """)
    return dms_library, summary_avgprefs, wolf_epitopes


@app.cell
def _(mo, pd, summary_avgprefs, wolf_epitopes):
    import glob
    import os

    # Create epitope site lookup using site_fix column
    epitope_sites = set(wolf_epitopes[wolf_epitopes['epi'] == 'epitope']['site_fix'].values)

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

    **Epitope Sites (using site_fix):**
    - Total epitope positions: {len(epitope_sites)}
    - Sample epitope positions: {sorted(list(epitope_sites))[:10]}...

    **Log Likelihood Data Loaded:**
    - Total files: {len(ll_files)}
    - Total records: {len(combined_ll)}
    - Models: {', '.join(combined_ll['model'].unique())}
    - Conditions: {', '.join(combined_ll['condition'].unique())}
    """)
    return combined_ll, dms_scores, epitope_sites


@app.cell
def _(
    combined_ll,
    dms_library,
    dms_scores,
    epitope_sites,
    mo,
    pd,
    summary_avgprefs,
):
    # Get amino acid columns for validation
    aa_cols = [col for col in summary_avgprefs.columns if col not in ['site', 'site_fix']]

    # Calculate DMS score for each sequence (total and epitope-only)
    def calc_dms_scores(sequence):
        total_score = 0
        epitope_score = 0
        for position, amino_acid in enumerate(sequence, start=1):
            # Only score amino acids that have DMS data
            if amino_acid in aa_cols:
                score = dms_scores.get((position, amino_acid), 0)
                total_score += score
                # Add to epitope score if position is an epitope site
                if position in epitope_sites:
                    epitope_score += score
        return total_score, epitope_score

    # Create scored library dataframe
    scored_library = dms_library.copy()
    scores = scored_library['sequence'].apply(calc_dms_scores)
    scored_library['dms_score'] = [s[0] for s in scores]
    scored_library['epitope_dms_score'] = [s[1] for s in scores]

    # Create comprehensive comparison dataframe by merging with log likelihood data
    comparison_df = pd.merge(
        scored_library[['node', 'dms_score', 'epitope_dms_score']], 
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

    **Score Statistics:**
    - Total DMS Score Range: {comparison_df['dms_score'].min():.3f} to {comparison_df['dms_score'].max():.3f}
    - Epitope DMS Score Range: {comparison_df['epitope_dms_score'].min():.3f} to {comparison_df['epitope_dms_score'].max():.3f}
    """)
    return (comparison_df,)


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
        comparison_df[['node', 'dms_score', 'epitope_dms_score']].drop_duplicates(), 
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

    # Get the log likelihood columns (exclude node and dms_score columns)
    ll_columns = [col for col in final_comparison.columns if col not in ['node', 'dms_score', 'epitope_dms_score']]

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

    mo.md("## DMS vs Log Likelihood Scatter Plots (Total DMS Score)")
    return


@app.cell
def _(final_comparison, mo, np, plt, sns, wolf_epitopes):
    import re

    # Create 1x4 plot with points colored by mutation epitope status
    sns.set_style("whitegrid")
    epitope_plot_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=epitope_plot_params)

    # Create epitope site lookup using site_fix column
    mutation_epitope_sites = set(wolf_epitopes[wolf_epitopes['epi'] == 'epitope']['site_fix'].values)

    # Function to parse mutation position from sequence name
    def parse_mutation_position(node_name):
        if 'WT' in node_name:
            return None  # Wildtype, no mutation

        # Extract mutation info from pattern like "Estimated_sequence_from_DMS_M1A"
        pattern = r'Estimated_sequence_from_DMS_[A-Z](\d+)[A-Z]'
        match = re.search(pattern, node_name)
        if match:
            position = int(match.group(1))
            # Convert to site_fix position (add 16 since site starts at -16 with site_fix=1)
            site_fix_position = position + 16
            return site_fix_position
        return None

    # Add epitope status to final_comparison
    mutation_comparison = final_comparison.copy()
    mutation_comparison['mutation_position'] = mutation_comparison['node'].apply(parse_mutation_position)
    mutation_comparison['is_epitope_mutation'] = mutation_comparison['mutation_position'].apply(
        lambda pos: pos in mutation_epitope_sites if pos is not None else False
    )
    mutation_comparison['is_wildtype'] = mutation_comparison['node'].str.contains('WT')

    # Get the log likelihood columns
    mutation_ll_columns = [col for col in final_comparison.columns if col not in ['node', 'dms_score', 'epitope_dms_score']]

    # Create 1x4 subplot layout
    mutation_fig, mutation_axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)

    for idx, ll_column in enumerate(mutation_ll_columns):
        mutation_ax = mutation_axes[idx]

        # Filter data
        data_mask = mutation_comparison[ll_column].notna() & mutation_comparison['dms_score'].notna()
        plot_data = mutation_comparison.loc[data_mask].copy()

        # Separate data by mutation type
        wildtype_data = plot_data[plot_data['is_wildtype']]
        epitope_mutation_data = plot_data[plot_data['is_epitope_mutation'] & ~plot_data['is_wildtype']]
        non_epitope_mutation_data = plot_data[~plot_data['is_epitope_mutation'] & ~plot_data['is_wildtype']]

        # Plot points with different colors
        if len(non_epitope_mutation_data) > 0:
            mutation_ax.scatter(non_epitope_mutation_data[ll_column], non_epitope_mutation_data['dms_score'], 
                              alpha=0.6, s=20, color='blue', label='Non-epitope mutations')

        if len(epitope_mutation_data) > 0:
            mutation_ax.scatter(epitope_mutation_data[ll_column], epitope_mutation_data['dms_score'], 
                              alpha=0.6, s=20, color='red', label='Epitope mutations')

        if len(wildtype_data) > 0:
            mutation_ax.scatter(wildtype_data[ll_column], wildtype_data['dms_score'], 
                              alpha=0.8, s=30, color='gray', label='Wildtype', marker='D')

        # Calculate overall correlation
        mutation_x_data = plot_data[ll_column]
        mutation_y_data = plot_data['dms_score']
        mutation_r_value = np.corrcoef(mutation_x_data, mutation_y_data)[0, 1]

        # Add trend line for all data
        mutation_z = np.polyfit(mutation_x_data, mutation_y_data, 1)
        mutation_p = np.poly1d(mutation_z)
        mutation_ax.plot(mutation_x_data, mutation_p(mutation_x_data), "black", linestyle="--", alpha=0.8, linewidth=1)

        # Format title and labels
        mutation_title_parts = ll_column.split('_')
        if len(mutation_title_parts) >= 2:
            # Extract model size and format model name
            if '650M' in mutation_title_parts:
                mutation_model_name = "ESM2-650M"
            elif '3B' in mutation_title_parts:
                mutation_model_name = "ESM2-3B"
            else:
                mutation_model_name = mutation_title_parts[0].replace('esm2', 'ESM2')

            mutation_condition_type = mutation_title_parts[-1].replace('base', 'Base').replace('tuned', 'Fine Tuned')
            mutation_title = f"{mutation_model_name}\n{mutation_condition_type}"
        else:
            mutation_title = ll_column

        mutation_ax.set_title(mutation_title, fontsize=12, weight='bold')
        mutation_ax.set_xlabel("Log Likelihood", fontsize=10)

        # Add correlation text
        mutation_ax.text(0.05, 0.95, f'r = {mutation_r_value:.3f}', 
                transform=mutation_ax.transAxes, fontsize=10, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Add legend to first plot only
        if idx == 0:
            mutation_ax.legend(loc='upper right', fontsize=8)

    # Set y-label only for the first subplot
    mutation_axes[0].set_ylabel("DMS Score", fontsize=12)

    plt.suptitle("DMS Score vs Log Likelihood: Mutations by Epitope Status", fontsize=14, weight='bold', y=1.05)
    plt.tight_layout()
    plt.show()

    mo.md(f"""
    ## DMS vs Log Likelihood: Mutations Colored by Epitope Status

    **Legend:**
    - **Blue**: Mutations at non-epitope sites
    - **Red**: Mutations at epitope sites  
    - **Gray diamonds**: Wildtype sequences (no mutations)

    **Mutation Summary:**
    - Epitope mutations: {len(mutation_comparison[mutation_comparison['is_epitope_mutation'] & ~mutation_comparison['is_wildtype']])}
    - Non-epitope mutations: {len(mutation_comparison[~mutation_comparison['is_epitope_mutation'] & ~mutation_comparison['is_wildtype']])}
    - Wildtype sequences: {len(mutation_comparison[mutation_comparison['is_wildtype']])}
    """)
    return (mutation_comparison,)


@app.cell
def _(mo, mutation_comparison, pd):
    # Compute average log likelihood and DMS scores by epitope status

    # Filter out wildtype sequences for mutation analysis
    mutation_only_data = mutation_comparison[~mutation_comparison['is_wildtype']].copy()

    # Get log likelihood columns
    stats_ll_cols = [stats_col for stats_col in mutation_comparison.columns if stats_col not in ['node', 'dms_score', 'epitope_dms_score', 'mutation_position', 'is_epitope_mutation', 'is_wildtype']]

    # Initialize results dictionary
    stats_results = {
        'Mutation_Type': [],
        'Count': [],
        'DMS_Score_Mean': [],
        'DMS_Score_Std': []
    }

    # Add log likelihood columns to results
    for stats_col in stats_ll_cols:
        stats_results[f'{stats_col}_Mean'] = []
        stats_results[f'{stats_col}_Std'] = []

    # Calculate stats for epitope mutations
    epitope_mutations = mutation_only_data[mutation_only_data['is_epitope_mutation']]
    stats_results['Mutation_Type'].append('Epitope Sites')
    stats_results['Count'].append(len(epitope_mutations))
    stats_results['DMS_Score_Mean'].append(epitope_mutations['dms_score'].mean())
    stats_results['DMS_Score_Std'].append(epitope_mutations['dms_score'].std())

    for stats_col in stats_ll_cols:
        stats_results[f'{stats_col}_Mean'].append(epitope_mutations[stats_col].mean())
        stats_results[f'{stats_col}_Std'].append(epitope_mutations[stats_col].std())

    # Calculate stats for non-epitope mutations
    non_epitope_mutations = mutation_only_data[~mutation_only_data['is_epitope_mutation']]
    stats_results['Mutation_Type'].append('Non-Epitope Sites')
    stats_results['Count'].append(len(non_epitope_mutations))
    stats_results['DMS_Score_Mean'].append(non_epitope_mutations['dms_score'].mean())
    stats_results['DMS_Score_Std'].append(non_epitope_mutations['dms_score'].std())

    for stats_col in stats_ll_cols:
        stats_results[f'{stats_col}_Mean'].append(non_epitope_mutations[stats_col].mean())
        stats_results[f'{stats_col}_Std'].append(non_epitope_mutations[stats_col].std())

    # Create summary dataframe
    summary_stats = pd.DataFrame(stats_results)

    # Round to 4 decimal places for readability
    stats_numeric_cols = [stats_col for stats_col in summary_stats.columns if stats_col not in ['Mutation_Type', 'Count']]
    for stats_col in stats_numeric_cols:
        summary_stats[stats_col] = summary_stats[stats_col].round(4)

    mo.md(f"""
    ## Average Scores by Mutation Location

    **Summary Statistics for Mutations at Epitope vs Non-Epitope Sites:**

    ### DMS Scores:
    - **Epitope mutations** (n={len(epitope_mutations)}): {epitope_mutations['dms_score'].mean():.4f} ± {epitope_mutations['dms_score'].std():.4f}
    - **Non-epitope mutations** (n={len(non_epitope_mutations)}): {non_epitope_mutations['dms_score'].mean():.4f} ± {non_epitope_mutations['dms_score'].std():.4f}

    **Statistical Test (DMS Scores):**
    """)

    # Perform t-test for DMS scores
    from scipy import stats
    dms_ttest = stats.ttest_ind(epitope_mutations['dms_score'], non_epitope_mutations['dms_score'])

    mo.md(f"""
    - t-statistic: {dms_ttest.statistic:.4f}
    - p-value: {dms_ttest.pvalue:.2e}
    - Significant difference: {'Yes' if dms_ttest.pvalue < 0.05 else 'No'} (α = 0.05)
    """)
    return (
        epitope_mutations,
        non_epitope_mutations,
        stats_ll_cols,
        summary_stats,
    )


@app.cell
def _(mo, summary_stats):
    # Display the complete summary table
    mo.md("### Complete Summary Table:")
    summary_stats
    return


@app.cell
def _(epitope_mutations, mo, non_epitope_mutations, plt, stats_ll_cols):
    # Create comparison plots for log likelihoods
    plot_fig, plot_axes = plt.subplots(2, 2, figsize=(12, 8))
    plot_axes = plot_axes.flatten()

    for plot_i, plot_col in enumerate(stats_ll_cols):
        plot_ax = plot_axes[plot_i]

        # Create box plots
        plot_data_to_plot = [
            non_epitope_mutations[plot_col].dropna(),
            epitope_mutations[plot_col].dropna()
        ]

        plot_box_plot = plot_ax.boxplot(plot_data_to_plot, 
                             labels=['Non-Epitope', 'Epitope'],
                             patch_artist=True)

        # Color the boxes
        plot_box_plot['boxes'][0].set_facecolor('lightblue')
        plot_box_plot['boxes'][1].set_facecolor('lightcoral')

        # Format title
        plot_title_parts = plot_col.split('_')
        if len(plot_title_parts) >= 2:
            if '650M' in plot_title_parts:
                plot_model_name = "ESM2-650M"
            elif '3B' in plot_title_parts:
                plot_model_name = "ESM2-3B"
            else:
                plot_model_name = plot_title_parts[0].replace('esm2', 'ESM2')

            plot_condition = plot_title_parts[-1].replace('base', 'Base').replace('tuned', 'Fine Tuned')
            plot_title = f"{plot_model_name} ({plot_condition})"
        else:
            plot_title = plot_col

        plot_ax.set_title(plot_title, fontsize=11, weight='bold')
        plot_ax.set_ylabel("Log Likelihood", fontsize=10)
        plot_ax.grid(True, alpha=0.3)

    plt.suptitle("Log Likelihood Distributions: Epitope vs Non-Epitope Mutations", 
                fontsize=14, weight='bold', y=0.98)
    plt.tight_layout()
    plt.show()

    mo.md("## Log Likelihood Distributions by Mutation Location")
    return


@app.cell
def _(mo, mutation_comparison, pd, stats_ll_cols):
    # Create dataframe with top 15 highest and lowest scores for each metric

    # Initialize list to store all rankings
    extreme_rankings = []

    # Get top/bottom 15 for DMS scores
    dms_top15 = mutation_comparison.nlargest(15, 'dms_score').copy()
    dms_top15['Rank_Type'] = 'DMS_Score_Top_15'
    dms_top15['Rank'] = range(1, 16)
    extreme_rankings.append(dms_top15)

    dms_bottom15 = mutation_comparison.nsmallest(15, 'dms_score').copy()
    dms_bottom15['Rank_Type'] = 'DMS_Score_Bottom_15'
    dms_bottom15['Rank'] = range(1, 16)
    extreme_rankings.append(dms_bottom15)

    # Get top/bottom 15 for each log likelihood column
    for ll_metric in stats_ll_cols:
        # Top 15 for this metric
        metric_top15 = mutation_comparison.nlargest(15, ll_metric).copy()
        metric_top15['Rank_Type'] = f'{ll_metric}_Top_15'
        metric_top15['Rank'] = range(1, 16)
        extreme_rankings.append(metric_top15)

        # Bottom 15 for this metric  
        metric_bottom15 = mutation_comparison.nsmallest(15, ll_metric).copy()
        metric_bottom15['Rank_Type'] = f'{ll_metric}_Bottom_15'
        metric_bottom15['Rank'] = range(1, 16)
        extreme_rankings.append(metric_bottom15)

    # Combine all rankings
    extreme_scores_df = pd.concat(extreme_rankings, ignore_index=True)

    # Reorder columns for better readability
    column_order = ['Rank_Type', 'Rank', 'node', 'dms_score', 'is_epitope_mutation', 'is_wildtype'] + stats_ll_cols
    extreme_scores_df = extreme_scores_df[column_order]

    # Rename columns for clarity
    extreme_scores_df = extreme_scores_df.rename(columns={
        'node': 'Sequence_Name',
        'dms_score': 'DMS_Score', 
        'is_epitope_mutation': 'Is_Epitope_Mutation',
        'is_wildtype': 'Is_Wildtype'
    })

    mo.md(f"""
    ## Top and Bottom 15 Sequences Analysis

    **Dataset Summary:**
    - Total rankings created: {len(extreme_scores_df)} entries
    - Metrics analyzed: DMS Score + {len(stats_ll_cols)} Log Likelihood measures  
    - Each metric shows top 15 and bottom 15 sequences

    **Ranking Categories:**
    - **DMS_Score_Top_15**: Highest fitness sequences
    - **DMS_Score_Bottom_15**: Lowest fitness sequences  
    - **[Model]_Top_15**: Highest log likelihood for each ESM model
    - **[Model]_Bottom_15**: Lowest log likelihood for each ESM model

    **Key Insights:**
    """)

    # Calculate some quick stats
    top_dms_epitope_count = sum(extreme_scores_df[extreme_scores_df['Rank_Type'] == 'DMS_Score_Top_15']['Is_Epitope_Mutation'])
    bottom_dms_epitope_count = sum(extreme_scores_df[extreme_scores_df['Rank_Type'] == 'DMS_Score_Bottom_15']['Is_Epitope_Mutation'])

    mo.md(f"""
    - Among top 15 DMS scores: {top_dms_epitope_count}/15 are epitope mutations
    - Among bottom 15 DMS scores: {bottom_dms_epitope_count}/15 are epitope mutations
    - Wildtype sequences in dataset: {sum(extreme_scores_df['Is_Wildtype'])} total
    """)
    return (extreme_scores_df,)


@app.cell
def _(extreme_scores_df, mo):
    # Display the complete extreme scores dataframe
    mo.md("### Complete Top/Bottom 15 Rankings:")
    extreme_scores_df
    return


@app.cell
def _(extreme_scores_df, mo):
    # Create summary analysis of epitope representation in extremes

    # Group by rank type and calculate epitope percentages
    epitope_summary = extreme_scores_df.groupby('Rank_Type').agg({
        'Is_Epitope_Mutation': ['count', 'sum', 'mean'],
        'Is_Wildtype': 'sum'
    }).round(3)

    # Flatten column names
    epitope_summary.columns = ['Total_Count', 'Epitope_Count', 'Epitope_Percentage', 'Wildtype_Count']
    epitope_summary['Non_Epitope_Count'] = epitope_summary['Total_Count'] - epitope_summary['Epitope_Count'] - epitope_summary['Wildtype_Count']

    # Reset index to make Rank_Type a column
    epitope_summary = epitope_summary.reset_index()

    mo.md("""
    ### Epitope Mutation Representation in Top/Bottom Rankings

    This table shows how many epitope vs non-epitope mutations appear in each ranking category:
    """)

    epitope_summary
    return


if __name__ == "__main__":
    app.run()
