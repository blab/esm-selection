# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo",
#     "pandas==2.3.3",
#     "numpy==2.3.4",
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
    return mo, pd


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
def _(mo, summary_avgprefs):
    # Create a lookup dictionary for DMS scores
    # Key: (position, amino_acid), Value: score
    dms_scores = {}

    # Get amino acid columns (skip 'site' and 'site_fix' columns)
    amino_acids = [col for col in summary_avgprefs.columns if col not in ['site', 'site_fix']]

    for _, row in summary_avgprefs.iterrows():
        position = row['site_fix']
        for aa in amino_acids:
            dms_scores[(position, aa)] = row[aa]

    mo.md(f"Created DMS score lookup with {len(dms_scores)} entries for amino acids: {', '.join(amino_acids)}")
    return (dms_scores,)


@app.cell
def _(dms_library, dms_scores, mo, summary_avgprefs):
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

    mo.md("## DMS Scores Calculated")
    return (scored_library,)


@app.cell
def _(mo, scored_library):
    # Display results
    results_summary = scored_library[['node', 'dms_score']].sort_values('dms_score', ascending=False)

    mo.md(f"""
    ## Results Summary

    Total sequences scored: {len(results_summary)}

    **Top 10 sequences by DMS score:**
    """)
    return (results_summary,)


@app.cell
def _(results_summary):
    # Show top 10 results
    top_results = results_summary.head(10)
    return


@app.cell
def _(mo, results_summary):
    # Basic statistics
    stats_display = mo.md(f"""
    ## Statistics

    - **Highest DMS Score:** {results_summary['dms_score'].max():.4f}
    - **Lowest DMS Score:** {results_summary['dms_score'].min():.4f}
    - **Mean DMS Score:** {results_summary['dms_score'].mean():.4f}
    - **Standard Deviation:** {results_summary['dms_score'].std():.4f}
    """)
    return


@app.cell
def _(mo, scored_library):
    # Save results to CSV
    output_path = "results/dms_scored_sequences.csv"
    scored_library[['node', 'dms_score']].to_csv(output_path, index=False)

    export_message = mo.md(f"""
    ## Export

    Results saved to: `{output_path}`
    """)
    return


if __name__ == "__main__":
    app.run()
