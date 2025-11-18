# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo",
#     "pandas==2.3.3",
#     "numpy==2.3.4",
#     "matplotlib==3.10.7",
#     "seaborn==0.13.2",
#     "scipy==1.16.3",
#     "tabulate==0.9.0",
#     "logomaker==0.8.7",
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
def _(pd):
    # Load the data files
    dms_library = pd.read_csv("results/dms_library.csv")
    summary_avgprefs = pd.read_csv("summary_avgprefs.csv")
    wolf_epitopes = pd.read_csv("wolf_epitope_sites.csv")
    return dms_library, summary_avgprefs, wolf_epitopes


@app.cell
def _(pd, summary_avgprefs, wolf_epitopes):
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
    return combined_ll, dms_scores, epitope_sites


@app.cell
def _(
    combined_ll,
    dms_library,
    dms_scores,
    epitope_sites,
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
    return (comparison_df,)


@app.cell
def _(comparison_df):
    # Show sample of combined dataframe
    comparison_df.head(10)
    return


@app.cell
def _(comparison_df):
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
def _(final_comparison, np, plt, sns, wolf_epitopes):
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
        mutation_pattern = r'Estimated_sequence_from_DMS_[A-Z](\d+)[A-Z]'
        mutation_match = re.search(mutation_pattern, node_name)
        if mutation_match:
            mutation_position = int(mutation_match.group(1))
            # Convert to site_fix position (add 16 since site starts at -16 with site_fix=1)
            site_fix_position = mutation_position + 16
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
                              alpha=0.8, s=30, color='gray', label='Perth 2009', marker='D')

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
    return mutation_comparison, re


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
    return stats_ll_cols, summary_stats


@app.cell
def _(mo, summary_stats):
    # Display the complete summary table
    mo.md("### Complete Summary Table:")
    summary_stats
    return


@app.cell
def _(summary_stats):
    stats_markdown = summary_stats.to_markdown(index=False)
    print(stats_markdown)
    return


@app.cell
def _(mutation_comparison, pd, stats_ll_cols):
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

    # Calculate some quick stats
    top_dms_epitope_count = sum(extreme_scores_df[extreme_scores_df['Rank_Type'] == 'DMS_Score_Top_15']['Is_Epitope_Mutation'])
    bottom_dms_epitope_count = sum(extreme_scores_df[extreme_scores_df['Rank_Type'] == 'DMS_Score_Bottom_15']['Is_Epitope_Mutation'])
    return (extreme_scores_df,)


@app.cell
def _(extreme_scores_df, mo):
    # Display the complete extreme scores dataframe
    mo.md("### Complete Top/Bottom 15 Rankings:")
    extreme_scores_df

    extreme_scores_df.iloc[:, 2] = extreme_scores_df.iloc[:, 2].str.replace("Estimated_sequence_from_", "", regex=False)
    return


@app.cell
def _(extreme_scores_df):
    extreme_scores_df
    return


@app.cell
def _(extreme_scores_df):
    print(extreme_scores_df[extreme_scores_df.iloc[:,0]=="DMS_Score_Top_15"].iloc[:,:3].to_markdown(index=False))
    return


@app.cell
def _(extreme_scores_df):
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

    epitope_summary
    return


@app.cell
def _(extreme_scores_df, mo, pd, re):
    import logomaker

    # Extract top 15 DMS scoring mutations for logomaker plot
    top_15_dms = extreme_scores_df[extreme_scores_df['Rank_Type'] == 'DMS_Score_Top_15'].copy()

    # Function to parse mutation information from sequence name
    def parse_top_mutation_details(sequence_name):
        # Extract mutation info from pattern like "DMS_M1A"
        top_pattern = r'DMS_([A-Z])(\d+)([A-Z])'
        top_match = re.search(top_pattern, sequence_name)
        if top_match:
            top_original_aa = top_match.group(1)
            top_mutation_position = int(top_match.group(2))
            top_mutant_aa = top_match.group(3)
            return top_mutation_position, top_original_aa, top_mutant_aa
        return None, None, None

    # Extract mutation details
    logo_mutations_list = []
    for _, top_mutation_row in top_15_dms.iterrows():
        top_mutation_pos, top_orig_aa, top_new_aa = parse_top_mutation_details(top_mutation_row['Sequence_Name'])
        if top_mutation_pos is not None:
            logo_mutations_list.append({
                'position': top_mutation_pos,
                'original_aa': top_orig_aa,
                'mutant_aa': top_new_aa,
                'dms_score': top_mutation_row['DMS_Score'],
                'rank': top_mutation_row['Rank']
            })

    logo_mutations_df = pd.DataFrame(logo_mutations_list)

    mo.md(f"""
    ## Top 15 DMS Mutations for Logo Plot

    Found {len(logo_mutations_df)} mutations:
    - Position range: {logo_mutations_df['position'].min()} to {logo_mutations_df['position'].max()}
    - Score range: {logo_mutations_df['dms_score'].min():.3f} to {logo_mutations_df['dms_score'].max():.3f}
    """)

    logo_mutations_df
    return logo_mutations_df, logomaker


@app.cell
def _(logo_mutations_df, logomaker, pd, plt):
    # Create position weight matrix for logomaker

    # Get all amino acids for logo
    logo_amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

    # Get unique positions and sort them
    logo_positions = sorted(logo_mutations_df['position'].unique())

    # Create the position weight matrix
    # Initialize with small pseudo-counts to avoid log(0) issues
    pwm_data_list = []

    for logo_pos in logo_positions:
        # Get mutations at this position
        pos_mutations = logo_mutations_df[logo_mutations_df['position'] == logo_pos]

        # Initialize row with zeros (no pseudocounts to make letters more visible)
        logo_row = {logo_aa: 0.0 for logo_aa in logo_amino_acids}

        # Add weights for observed mutations
        for _, logo_mut in pos_mutations.iterrows():
            # Use DMS score directly as weight, scaled by inverse rank
            top_mutation_weight = logo_mut['dms_score'] * (16 - logo_mut['rank']) 
            logo_row[logo_mut['mutant_aa']] += top_mutation_weight

        # Only normalize if we have non-zero values
        logo_total = sum(logo_row.values())
        if logo_total > 0:
            # Add small pseudocount only to observed amino acids to avoid division by zero
            for logo_aa in logo_amino_acids:
                if logo_row[logo_aa] > 0:
                    logo_row[logo_aa] = logo_row[logo_aa] / logo_total
                else:
                    logo_row[logo_aa] = 0.0

        logo_row['position'] = logo_pos
        pwm_data_list.append(logo_row)

    # Convert to DataFrame
    logo_pwm_df = pd.DataFrame(pwm_data_list)
    logo_pwm_df = logo_pwm_df.set_index('position')

    # Create the logo plot
    logo_fig, logo_ax = plt.subplots(1, 1, figsize=(12, 4))

    # Generate logo
    sequence_logo = logomaker.Logo(logo_pwm_df, ax=logo_ax, color_scheme='chemistry')

    # Customize the plot
    logo_ax.set_xlabel('Position', fontsize=12)
    logo_ax.set_ylabel('Amino Acid Counts', fontsize=12) 
    logo_ax.set_title('Sequence Logo: Top 15 DMS Scoring Mutations', fontsize=14, weight='bold')

    # Set x-axis limits to show full range from 112 to 277
    logo_ax.set_xlim(111.5, 277.5)

    # Add position labels only for positions with mutations
    logo_ax.set_xticks(logo_positions)
    logo_ax.set_xticklabels([f'{pos}' for pos in logo_positions])

    plt.tight_layout()
    plt.show()
    return (logo_pwm_df,)


@app.cell
def _(logo_mutations_df, logo_pwm_df):
    # Display the detailed mutation information and PWM
    print("Top 15 Mutations Details:")
    print(logo_mutations_df.to_string(index=False))
    print("\nPosition Weight Matrix:")
    logo_pwm_df.round(3)
    return


@app.cell
def _(extreme_scores_df, mo, pd, re):
    # Extract bottom 15 DMS scoring mutations for logomaker plot
    bottom_15_dms = extreme_scores_df[extreme_scores_df['Rank_Type'] == 'DMS_Score_Bottom_15'].copy()

    # Function to parse mutation information from sequence name
    def parse_bottom_mutation_details(sequence_name):
        # Extract mutation info from pattern like "DMS_M1A"
        bottom_pattern = r'DMS_([A-Z])(\d+)([A-Z])'
        bottom_match = re.search(bottom_pattern, sequence_name)
        if bottom_match:
            bottom_original_aa = bottom_match.group(1)
            bottom_mutation_position = int(bottom_match.group(2))
            bottom_mutant_aa = bottom_match.group(3)
            return bottom_mutation_position, bottom_original_aa, bottom_mutant_aa
        return None, None, None

    # Extract mutation details
    bottom_logo_mutations_list = []
    for _, bottom_mutation_row in bottom_15_dms.iterrows():
        bottom_mutation_pos, bottom_orig_aa, bottom_new_aa = parse_bottom_mutation_details(bottom_mutation_row['Sequence_Name'])
        if bottom_mutation_pos is not None:
            bottom_logo_mutations_list.append({
                'position': bottom_mutation_pos,
                'original_aa': bottom_orig_aa,
                'mutant_aa': bottom_new_aa,
                'dms_score': bottom_mutation_row['DMS_Score'],
                'rank': bottom_mutation_row['Rank']
            })

    bottom_logo_mutations_df = pd.DataFrame(bottom_logo_mutations_list)

    mo.md(f"""
    ## Bottom 15 DMS Mutations for Logo Plot

    Found {len(bottom_logo_mutations_df)} mutations:
    - Position range: {bottom_logo_mutations_df['position'].min()} to {bottom_logo_mutations_df['position'].max()}
    - Score range: {bottom_logo_mutations_df['dms_score'].min():.3f} to {bottom_logo_mutations_df['dms_score'].max():.3f}
    """)

    bottom_logo_mutations_df
    return (bottom_logo_mutations_df,)


@app.cell
def _(bottom_logo_mutations_df, logomaker, pd, plt):
    # Create position weight matrix for bottom 15 mutations logomaker

    # Get all amino acids for logo
    bottom_logo_amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

    # Get unique positions and sort them
    bottom_logo_positions = sorted(bottom_logo_mutations_df['position'].unique())

    # Create the position weight matrix
    bottom_pwm_data_list = []

    for bottom_logo_pos in bottom_logo_positions:
        # Get mutations at this position
        bottom_pos_mutations = bottom_logo_mutations_df[bottom_logo_mutations_df['position'] == bottom_logo_pos]

        # Initialize row with zeros
        bottom_logo_row = {bottom_logo_aa: 0.0 for bottom_logo_aa in bottom_logo_amino_acids}

        # Add weights for observed mutations
        for _, bottom_logo_mut in bottom_pos_mutations.iterrows():
            # Use absolute DMS score (since they're negative) scaled by inverse rank
            bottom_mutation_weight = abs(bottom_logo_mut['dms_score']) * (16 - bottom_logo_mut['rank']) 
            bottom_logo_row[bottom_logo_mut['mutant_aa']] += bottom_mutation_weight

        # Only normalize if we have non-zero values
        bottom_logo_total = sum(bottom_logo_row.values())
        if bottom_logo_total > 0:
            # Add small pseudocount only to observed amino acids to avoid division by zero
            for bottom_logo_aa in bottom_logo_amino_acids:
                if bottom_logo_row[bottom_logo_aa] > 0:
                    bottom_logo_row[bottom_logo_aa] = bottom_logo_row[bottom_logo_aa] / bottom_logo_total
                else:
                    bottom_logo_row[bottom_logo_aa] = 0.0

        bottom_logo_row['position'] = bottom_logo_pos
        bottom_pwm_data_list.append(bottom_logo_row)

    # Convert to DataFrame
    bottom_logo_pwm_df = pd.DataFrame(bottom_pwm_data_list)
    bottom_logo_pwm_df = bottom_logo_pwm_df.set_index('position')

    # Create the logo plot
    bottom_logo_fig, bottom_logo_ax = plt.subplots(1, 1, figsize=(12, 4))

    # Generate logo
    bottom_sequence_logo = logomaker.Logo(bottom_logo_pwm_df, ax=bottom_logo_ax, color_scheme='chemistry')

    # Customize the plot
    bottom_logo_ax.set_xlabel('Position', fontsize=12)
    bottom_logo_ax.set_ylabel('Amino Acid Counts', fontsize=12) 
    bottom_logo_ax.set_title('Sequence Logo: Bottom 15 DMS Scoring Mutations', fontsize=14, weight='bold')

    # Set x-axis limits to show full range from 533 to 277
    bottom_logo_ax.set_xlim(532, 534)

    # Add position labels only for positions with mutations
    bottom_logo_ax.set_xticks(bottom_logo_positions)
    bottom_logo_ax.set_xticklabels([f'{pos}' for pos in bottom_logo_positions])

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(extreme_scores_df):
    # Check the column names to see the exact format
    print("Column names in extreme_scores_df:")
    print(list(extreme_scores_df.columns))
    print("\nUnique Rank_Type values:")
    print(extreme_scores_df['Rank_Type'].unique())
    return


@app.cell
def _(extreme_scores_df, pd, re):
    # Extract top 15 ESM2-650M base model scoring mutations for logomaker plot
    esm_650m_top_15 = extreme_scores_df[extreme_scores_df['Rank_Type'] == 'model~esm2_t33_650M_UR50D_base_Top_15'].copy()

    # Function to parse mutation information from sequence name
    def parse_esm_mutation_details(sequence_name):
        # Extract mutation info from pattern like "DMS_M1A"
        esm_pattern = r'DMS_([A-Z])(\d+)([A-Z])'
        esm_match = re.search(esm_pattern, sequence_name)
        if esm_match:
            esm_original_aa = esm_match.group(1)
            esm_mutation_position = int(esm_match.group(2))
            esm_mutant_aa = esm_match.group(3)
            return esm_mutation_position, esm_original_aa, esm_mutant_aa
        return None, None, None

    # Extract mutation details for top 15
    esm_top_mutations_list = []
    for _, esm_mutation_row in esm_650m_top_15.iterrows():
        esm_mutation_pos, esm_orig_aa, esm_new_aa = parse_esm_mutation_details(esm_mutation_row['Sequence_Name'])
        if esm_mutation_pos is not None:
            esm_top_mutations_list.append({
                'position': esm_mutation_pos,
                'original_aa': esm_orig_aa,
                'mutant_aa': esm_new_aa,
                'esm_score': esm_mutation_row['model~esm2_t33_650M_UR50D_base'],
                'rank': esm_mutation_row['Rank']
            })

    esm_top_mutations_df = pd.DataFrame(esm_top_mutations_list)

    # Extract bottom 15 ESM2-650M base model scoring mutations
    esm_650m_bottom_15 = extreme_scores_df[extreme_scores_df['Rank_Type'] == 'model~esm2_t33_650M_UR50D_base_Bottom_15'].copy()

    # Extract mutation details for bottom 15
    esm_bottom_mutations_list = []
    for _, esm_bottom_mutation_row in esm_650m_bottom_15.iterrows():
        esm_bottom_mutation_pos, esm_bottom_orig_aa, esm_bottom_new_aa = parse_esm_mutation_details(esm_bottom_mutation_row['Sequence_Name'])
        if esm_bottom_mutation_pos is not None:
            esm_bottom_mutations_list.append({
                'position': esm_bottom_mutation_pos,
                'original_aa': esm_bottom_orig_aa,
                'mutant_aa': esm_bottom_new_aa,
                'esm_score': esm_bottom_mutation_row['model~esm2_t33_650M_UR50D_base'],
                'rank': esm_bottom_mutation_row['Rank']
            })

    esm_bottom_mutations_df = pd.DataFrame(esm_bottom_mutations_list)

    print(f"ESM2-650M Top 15 mutations: {len(esm_top_mutations_df)}")
    print(f"ESM2-650M Bottom 15 mutations: {len(esm_bottom_mutations_df)}")
    return (
        esm_650m_bottom_15,
        esm_650m_top_15,
        esm_bottom_mutations_df,
        esm_top_mutations_df,
    )


@app.cell
def _(esm_top_mutations_df, logomaker, pd, plt):
    # Create position weight matrix for ESM2-650M top 15 mutations logomaker

    # Get all amino acids for logo
    esm_top_amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

    # Get unique positions and sort them
    esm_top_positions = sorted(esm_top_mutations_df['position'].unique())

    # Create the position weight matrix
    esm_top_pwm_data_list = []

    for esm_top_pos in esm_top_positions:
        # Get mutations at this position
        esm_top_pos_mutations = esm_top_mutations_df[esm_top_mutations_df['position'] == esm_top_pos]

        # Initialize row with zeros
        esm_top_row = {esm_top_aa: 0.0 for esm_top_aa in esm_top_amino_acids}

        # Add weights for observed mutations
        for _, esm_top_mut in esm_top_pos_mutations.iterrows():
            # Use inverse rank as weight (rank 1 gets weight 15, rank 15 gets weight 1)
            # This gives higher weights to better-ranked mutations
            esm_top_weight = 16 - esm_top_mut['rank']
            esm_top_row[esm_top_mut['mutant_aa']] += esm_top_weight

        # Only normalize if we have non-zero values
        esm_top_total = sum(esm_top_row.values())
        if esm_top_total > 0:
            for esm_top_aa in esm_top_amino_acids:
                if esm_top_row[esm_top_aa] > 0:
                    esm_top_row[esm_top_aa] = esm_top_row[esm_top_aa] / esm_top_total
                else:
                    esm_top_row[esm_top_aa] = 0.0

        esm_top_row['position'] = esm_top_pos
        esm_top_pwm_data_list.append(esm_top_row)

    # Convert to DataFrame
    esm_top_pwm_df = pd.DataFrame(esm_top_pwm_data_list)
    esm_top_pwm_df = esm_top_pwm_df.set_index('position')

    # Create the logo plot
    esm_top_fig, esm_top_ax = plt.subplots(1, 1, figsize=(12, 4))

    # Generate logo
    esm_top_sequence_logo = logomaker.Logo(esm_top_pwm_df, ax=esm_top_ax, color_scheme='chemistry')

    # Customize the plot
    esm_top_ax.set_xlabel('Position', fontsize=12)
    esm_top_ax.set_ylabel('Amino Acid Counts', fontsize=12) 
    esm_top_ax.set_title('Sequence Logo: Top 15 ESM2-650M Base Model Scoring Mutations', fontsize=14, weight='bold')

    # Set x-axis limits based on data range
    if len(esm_top_positions) > 0:
        esm_top_ax.set_xlim(min(esm_top_positions) - 0.5, max(esm_top_positions) + 0.5)

    # Add position labels only for positions with mutations
    esm_top_ax.set_xticks(esm_top_positions)
    esm_top_ax.set_xticklabels([f'{pos}' for pos in esm_top_positions])

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(esm_650m_top_15):
    esm_650m_top_15
    return


@app.cell
def _(esm_bottom_mutations_df, logomaker, pd, plt):
    # Create position weight matrix for ESM2-650M bottom 15 mutations logomaker

    # Get all amino acids for logo
    esm_bottom_amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

    # Get unique positions and sort them
    esm_bottom_positions = sorted(esm_bottom_mutations_df['position'].unique())

    # Create the position weight matrix
    esm_bottom_pwm_data_list = []

    for esm_bottom_pos in esm_bottom_positions:
        # Get mutations at this position
        esm_bottom_pos_mutations = esm_bottom_mutations_df[esm_bottom_mutations_df['position'] == esm_bottom_pos]

        # Initialize row with zeros
        esm_bottom_row = {esm_bottom_aa: 0.0 for esm_bottom_aa in esm_bottom_amino_acids}

        # Add weights for observed mutations
        for _, esm_bottom_mut in esm_bottom_pos_mutations.iterrows():
            # Use inverse rank as weight (rank 1 gets weight 15, rank 15 gets weight 1)
            # This gives higher weights to mutations ranked worse (more negative scores)
            esm_bottom_weight = 16 - esm_bottom_mut['rank']
            esm_bottom_row[esm_bottom_mut['mutant_aa']] += esm_bottom_weight

        # Only normalize if we have non-zero values
        esm_bottom_total = sum(esm_bottom_row.values())
        if esm_bottom_total > 0:
            for esm_bottom_aa in esm_bottom_amino_acids:
                if esm_bottom_row[esm_bottom_aa] > 0:
                    esm_bottom_row[esm_bottom_aa] = esm_bottom_row[esm_bottom_aa] / esm_bottom_total
                else:
                    esm_bottom_row[esm_bottom_aa] = 0.0

        esm_bottom_row['position'] = esm_bottom_pos
        esm_bottom_pwm_data_list.append(esm_bottom_row)

    # Convert to DataFrame
    esm_bottom_pwm_df = pd.DataFrame(esm_bottom_pwm_data_list)
    esm_bottom_pwm_df = esm_bottom_pwm_df.set_index('position')

    # Create the logo plot
    esm_bottom_fig, esm_bottom_ax = plt.subplots(1, 1, figsize=(12, 4))

    # Generate logo
    esm_bottom_sequence_logo = logomaker.Logo(esm_bottom_pwm_df, ax=esm_bottom_ax, color_scheme='chemistry')

    # Customize the plot
    esm_bottom_ax.set_xlabel('Position', fontsize=12)
    esm_bottom_ax.set_ylabel('Amino Acid Counts', fontsize=12) 
    esm_bottom_ax.set_title('Sequence Logo: Bottom 15 ESM2-650M Base Model Scoring Mutations', fontsize=14, weight='bold')

    # Set x-axis limits based on data range
    if len(esm_bottom_positions) > 0:
        esm_bottom_ax.set_xlim(min(esm_bottom_positions) - 0.5, max(esm_bottom_positions) + 0.5)

    # Add position labels only for positions with mutations
    esm_bottom_ax.set_xticks(esm_bottom_positions)
    esm_bottom_ax.set_xticklabels([f'{pos}' for pos in esm_bottom_positions])

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(esm_650m_bottom_15):
    esm_650m_bottom_15
    return


if __name__ == "__main__":
    app.run()
