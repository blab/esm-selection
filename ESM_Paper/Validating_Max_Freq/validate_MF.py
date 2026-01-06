# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "altair==6.0.0",
#     "basedpyright==1.36.0",
#     "duckdb==1.4.3",
#     "matplotlib==3.10.7",
#     "nbformat==5.10.4",
#     "numpy==2.3.5",
#     "openai==2.9.0",
#     "pandas==2.3.3",
#     "polars[pyarrow]==1.36.1",
#     "pytest==9.0.2",
#     "python-lsp-ruff==2.3.0",
#     "python-lsp-server==1.14.0",
#     "ruff==0.14.8",
#     "scikit-learn==1.7.2",
#     "scipy==1.16.3",
#     "seaborn==0.13.2",
#     "sqlglot==28.1.0",
#     "ty==0.0.1a33",
#     "vegafusion==2.0.3",
#     "vl-convert-python==1.8.0",
#     "websockets==15.0.1",
# ]
# ///

import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Max frequency validation
    """)
    return


@app.cell
def _():
    import pandas as pd
    import os
    import marimo as mo
    import subprocess
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import r2_score
    import altair as alt
    return (
        alt,
        json,
        mo,
        np,
        pd,
        pearsonr,
        plt,
        r2_score,
        sns,
        spearmanr,
        subprocess,
    )


@app.cell
def _(alt):
    # Configure Altair to use light theme (override marimo's theme setting)
    # Force light theme explicitly
    alt.themes.enable('default')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Download metadata for simulated samples with fitness and dates
    """)
    return


@app.cell
def _(subprocess):
    subprocess.run(
        [
            "curl",
            "-OL",
            "https://github.com/blab/flu-forecasting/raw/refs/heads/master/data/simulated/simulated_sample_3/filtered_metadata.tsv",
        ],
        check=True,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Download sequences for simulated samples.
    """)
    return


@app.cell
def _(subprocess):
    subprocess.run(
        [
            "curl",
            "-OL",
            "https://github.com/blab/flu-forecasting/raw/refs/heads/master/data/simulated/simulated_sample_3/filtered_sequences.fasta",
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Filter records by date and subsample to reasonable number.
    """)
    return


@app.cell
def _(subprocess):
    subprocess.run(
        [
            "augur",
            "filter",
            "--metadata",
            "filtered_metadata.tsv",
            "--sequences",
            "filtered_sequences.fasta",
            "--min-date",
            "2010-01-01",
            "--max-date",
            "2015-01-01",
            "--subsample-max-sequences",
            "3000",
            "--group-by",
            "year",
            "month",
            "--output-metadata",
            "subsampled_metadata.tsv",
            "--output-sequences",
            "subsampled_sequences.fasta",
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Build a tree. The sequences are already "aligned" from the simulations.
    """)
    return


@app.cell
def _(subprocess):
    subprocess.run(
        [
            "augur",
            "tree",
            "--alignment",
            "subsampled_sequences.fasta",
            "--output",
            "tree_raw.nwk",
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Add internal node names with TreeTime.
    """)
    return


@app.cell
def _(subprocess):
    subprocess.run(
        [
            "augur",
            "refine",
            "--tree",
            "tree_raw.nwk",
            "--alignment",
            "subsampled_sequences.fasta",
            "--metadata",
            "subsampled_metadata.tsv",
            "--timetree",
            "--output-tree",
            "tree.nwk",
            "--output-node-data",
            "branch_lengths.json",
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Infer ancestral sequences and mutations
    """)
    return


@app.cell
def _(subprocess):
    subprocess.run(
        [
            "augur",
            "ancestral",
            "--tree",
            "tree.nwk",
            "--alignment",
            "subsampled_sequences.fasta",
            "--output-node-data",
            "nt_muts.json",
            "--output-sequences",
            "ancestral_sequences.fasta",
            "--inference",
            "joint",
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Estimate frequencies from the tree with the diffusion method.
    ### Use 6-month intervals for pivots.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Frequencies diffusion
    """)
    return


@app.cell
def _(subprocess):
    subprocess.run(
        [
            "augur",
            "frequencies",
            "--method",
            "diffusion",
            "--metadata",
            "subsampled_metadata.tsv",
            "--tree",
            "tree.nwk",
            "--include-internal-nodes",
            "--stiffness",
            "20",
            "--inertia",
            "0.2",
            "--pivot-interval",
            "6",
            "--min-date",
            "2010-01-01",
            "--max-date",
            "2015-01-01",
            "--output",
            "frequencies_diffusion.json",
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Frequencies KDE
    """)
    return


@app.cell
def _(subprocess):
    subprocess.run(
        [
            "augur",
            "frequencies",
            "--method",
            "kde",
            "--metadata",
            "subsampled_metadata.tsv",
            "--tree",
            "tree.nwk",
            "--include-internal-nodes",
            "--stiffness",
            "20",
            "--inertia",
            "0.2",
            "--pivot-interval",
            "6",
            "--min-date",
            "2010-01-01",
            "--max-date",
            "2015-01-01",
            "--output",
            "frequencies_kde.json",
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Export Nextstrain visualization with frequencies and mutations
    ### Creates auspice.json for visualization and analysis
    """)
    return


@app.cell
def _(json, subprocess):
    # Export full Nextstrain visualization with diffusion frequencies
    subprocess.run(
        [
            "augur",
            "export",
            "v2",
            "--tree",
            "tree.nwk",
            "--metadata",
            "subsampled_metadata.tsv",
            "--node-data",
            "branch_lengths.json",
            "nt_muts.json",
            "frequencies_diffusion.json",
            "--color-by-metadata",
            "fitness",
            "generation",
            "--metadata-columns",
            "date",
            "num_date",
            "fitness",
            "generation",
            "--panels",
            "tree",
            "frequencies",
            "entropy",
            "--title",
            "Simulated Influenza - Max Frequency Validation (Diffusion)",
            "--output",
            "simulated_flu_diffusion.json",
        ]
    )

    # Export full Nextstrain visualization with KDE frequencies
    subprocess.run(
        [
            "augur",
            "export",
            "v2",
            "--tree",
            "tree.nwk",
            "--metadata",
            "subsampled_metadata.tsv",
            "--node-data",
            "branch_lengths.json",
            "nt_muts.json",
            "frequencies_kde.json",
            "--color-by-metadata",
            "fitness",
            "generation",
            "--metadata-columns",
            "date",
            "num_date",
            "fitness",
            "generation",
            "--panels",
            "tree",
            "frequencies",
            "entropy",
            "--title",
            "Simulated Influenza - Max Frequency Validation (KDE)",
            "--output",
            "simulated_flu_kde.json",
        ]
    )

    # Convert frequencies to Auspice v2 tip-frequencies format for both methods
    for method in ["diffusion", "kde"]:
        freq_file = f"frequencies_{method}.json"
        # Diffusion uses "global" key, KDE uses "frequencies" key
        freq_key = "global" if method == "diffusion" else "frequencies"

        with open(freq_file, "r") as f:
            freq_data = json.load(f)

        # Extract pivots and create Auspice format
        auspice_frequencies = {
            "pivots": freq_data["pivots"],
            "projection_pivot": freq_data["pivots"][-1] if "pivots" in freq_data else None,
        }

        # Add frequency data for each node
        for node_name, node_data in freq_data.items():
            if node_name not in ["pivots", "generated_by"]:
                auspice_frequencies[node_name] = {
                    "frequencies": node_data.get(freq_key, [])
                }

        # Write the tip-frequencies sidecar file
        with open(f"simulated_flu_{method}_tip-frequencies.json", "w") as f:
            json.dump(auspice_frequencies, f, indent=2)

    # Create auspice.json symlink for convenience (use diffusion as default)
    subprocess.run(["ln", "-sf", "simulated_flu_diffusion.json", "auspice.json"])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Assign Fitness to Internal Nodes
    ### Based on nearest descendant leaf by divergence (temporal tie-breaking)
    """)
    return


@app.function
def assign_fitness_to_internal_nodes(tree_data, method_name="diffusion"):
    """
    Assign fitness scores to internal nodes based on nearest leaf descendant.

    When multiple leaves have the same minimum divergence distance, uses temporal
    proximity (num_date) as a tie-breaker.

    Args:
        tree_data: The loaded JSON tree (Auspice v2 format)
        method_name: "diffusion" or "kde" for tracking

    Returns:
        tuple: (modified_tree_data, statistics_dict)
    """
    import copy

    # Statistics tracking
    stats = {
        'method': method_name,
        'internal_nodes_total': 0,
        'internal_nodes_assigned': 0,
        'divergence_ties': 0,
        'temporal_ties': 0
    }

    def find_nearest_leaf_with_fitness(node, parent_div, parent_time):
        """
        Recursively find the nearest leaf with fitness information.

        Returns:
            tuple: (fitness_value, div_distance, time_distance, leaf_name, leaf_time)
                   or (None, float('inf'), float('inf'), None, None) if no fitness found
        """
        current_div = node.get('node_attrs', {}).get('div', 0)
        current_time = node.get('node_attrs', {}).get('num_date', {}).get('value', 0)
        children = node.get('children', [])

        # Base case: leaf node
        if not children:
            fitness_info = node.get('node_attrs', {}).get('fitness', {})
            if 'value' in fitness_info:
                div_distance = current_div - parent_div
                time_distance = abs(current_time - parent_time)
                return (
                    fitness_info['value'],
                    div_distance,
                    time_distance,
                    node.get('name'),
                    current_time
                )
            else:
                return (None, float('inf'), float('inf'), None, None)

        # Recursive case: internal node - collect candidates from all children
        candidates = []
        for child in children:
            result = find_nearest_leaf_with_fitness(child, parent_div, parent_time)
            if result[0] is not None:  # fitness value exists
                candidates.append(result)

        if not candidates:
            return (None, float('inf'), float('inf'), None, None)

        # Step 1: Find minimum divergence distance
        min_div_distance = min(c[1] for c in candidates)

        # Step 2: Filter to candidates with minimum divergence
        tied_candidates = [c for c in candidates if c[1] == min_div_distance]

        # Track divergence ties
        if len(tied_candidates) > 1:
            # Step 3: Among divergence ties, find minimum temporal distance
            min_time_distance = min(c[2] for c in tied_candidates)
            final_candidates = [c for c in tied_candidates if c[2] == min_time_distance]

            # Select first candidate (deterministic based on traversal order)
            selected = final_candidates[0]
            return selected
        else:
            return tied_candidates[0]

    def assign_fitness_recursive(node):
        """Recursively assign fitness to internal nodes."""
        current_name = node.get('name', '')
        current_div = node.get('node_attrs', {}).get('div', 0)
        current_time = node.get('node_attrs', {}).get('num_date', {}).get('value', 0)

        # Process internal nodes only
        if current_name.startswith('NODE_'):
            stats['internal_nodes_total'] += 1

            # Find nearest leaf descendant
            fitness_value, div_distance, time_distance, leaf_name, leaf_time = \
                find_nearest_leaf_with_fitness(node, current_div, current_time)

            if fitness_value is not None:
                # Assign fitness with metadata
                if 'node_attrs' not in node:
                    node['node_attrs'] = {}

                node['node_attrs']['fitness'] = {
                    'value': fitness_value,
                    'inferred': True,
                    'source_leaf': leaf_name,
                    'divergence_distance': div_distance
                }

                stats['internal_nodes_assigned'] += 1

                # Check if this was a divergence tie (distance matches another candidate)
                # This is implicitly tracked in find_nearest_leaf_with_fitness

        # Recurse to all children
        for child in node.get('children', []):
            assign_fitness_recursive(child)

    # Process the tree starting from root
    tree_root = tree_data.get('tree', tree_data)
    assign_fitness_recursive(tree_root)

    return tree_data, stats


@app.cell
def _(json):
    import copy

    # Process both methods using a loop
    _stats = {}
    for _method in ["diffusion", "kde"]:
        # Load tree data
        with open(f"simulated_flu_{_method}.json", "r") as _f_in:
            _tree_data = json.load(_f_in)

        # Assign fitness to internal nodes
        _enriched_data, _stats[_method] = assign_fitness_to_internal_nodes(
            copy.deepcopy(_tree_data),
            _method
        )

        # Save enriched tree
        with open(f"simulated_flu_{_method}_enriched.json", "w") as _f_out:
            json.dump(_enriched_data, _f_out, indent=2)

    diffusion_stats = _stats["diffusion"]
    kde_stats = _stats["kde"]
    return diffusion_stats, kde_stats


@app.cell
def _(diffusion_stats, kde_stats, mo, pd):
    stats_df = pd.DataFrame({
        'Method': ['Diffusion', 'KDE'],
        'Internal Nodes': [diffusion_stats['internal_nodes_total'],
                           kde_stats['internal_nodes_total']],
        'Assigned Fitness': [diffusion_stats['internal_nodes_assigned'],
                             kde_stats['internal_nodes_assigned']],
        'Success Rate': [
            f"{100 * diffusion_stats['internal_nodes_assigned'] / max(diffusion_stats['internal_nodes_total'], 1):.1f}%",
            f"{100 * kde_stats['internal_nodes_assigned'] / max(kde_stats['internal_nodes_total'], 1):.1f}%"
        ]
    })

    mo.ui.table(stats_df)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Extract maximum frequency from JSON
    """)
    return


@app.cell
def _(json, pd):
    def collect_terminal_nodes(node):
        """
        Recursively collect terminal nodes (nodes without children) for each node in the tree.
        """
        name = node.get("name", "(unnamed)")
        children = node.get("children", [])

        # If the node has no children, it's a terminal node
        if not children:
            return [name]

        # Otherwise, collect terminal nodes from all children
        terminal_nodes = []
        for child in children:
            terminal_nodes.extend(collect_terminal_nodes(child))

        return terminal_nodes


    def map_terminal_nodes(node, node_terminal_map):
        """
        Create a mapping of each node to its terminal nodes.
        """
        name = node.get("name", "(unnamed)")
        children = node.get("children", [])

        # Collect terminal nodes for this node
        terminal_nodes = collect_terminal_nodes(node)
        node_terminal_map[name] = terminal_nodes

        # Recurse into each child
        for child in children:
            map_terminal_nodes(child, node_terminal_map)


    def extract_node_metadata(node, metadata_map):
        """
        Recursively extract fitness, generation, and date from all nodes in tree.

        Args:
            node: Current node in tree traversal
            metadata_map: Dictionary to populate with node_name -> {fitness, generation, date}
        """
        name = node.get("name")
        node_attrs = node.get("node_attrs", {})

        # Extract metadata fields (safely handling missing values)
        fitness_obj = node_attrs.get("fitness", {})
        generation_obj = node_attrs.get("generation", {})
        date_obj = node_attrs.get("date", {})

        metadata_map[name] = {
            'fitness': fitness_obj.get('value') if isinstance(fitness_obj, dict) else None,
            'generation': generation_obj.get('value') if isinstance(generation_obj, dict) else None,
            'date': date_obj.get('value') if isinstance(date_obj, dict) else None
        }

        # Recurse to children
        for child in node.get("children", []):
            extract_node_metadata(child, metadata_map)


    def get_freq_sum(tree_file, tip_freq_file, freq_key="global"):
        # Load tree JSON
        with open(tree_file, "r") as f:
            data = json.load(f)

        # Entry point (usually under 'tree' or 'nodes')
        tree_root = data.get("tree", data)

        # Dictionary to store the mapping of nodes to their terminal nodes
        node_terminal_map = {}

        # Map terminal nodes for each node
        map_terminal_nodes(tree_root, node_terminal_map)

        # Extract metadata (fitness, generation, date) from tree
        node_metadata_map = {}
        extract_node_metadata(tree_root, node_metadata_map)

        # Import and clean up the frequency JSON
        with open(tip_freq_file, "r") as json_fh_frequency:
            json_dict_frequency = json.load(json_fh_frequency)

        # Remove metadata fields
        json_dict_frequency.pop("pivots", None)
        json_dict_frequency.pop("generated_by", None)

        # Sum frequencies
        summed_frequencies = {}
        for node, terminals in node_terminal_map.items():
            summed = None
            for terminal in terminals:
                if terminal in json_dict_frequency:
                    freqs = json_dict_frequency[terminal][freq_key]
                    if summed is None:
                        summed = freqs.copy()
                    else:
                        summed = [x + y for x, y in zip(summed, freqs)]
            if summed is not None:
                summed_frequencies[node] = summed

        # Get max frequency for each node
        max_values = {key: max(values) for key, values in summed_frequencies.items()}

        # Create DataFrame WITH metadata from tree
        node_data = {}
        for node, score in max_values.items():
            # Determine if it's an internal or terminal node
            node_type = "internal" if node.startswith("NODE_") else "terminal"

            # Get metadata from tree
            metadata = node_metadata_map.get(node, {})

            node_data[node] = {
                "max_frequency": score,
                "node_type": node_type,
                "fitness": metadata.get('fitness'),
                "generation": metadata.get('generation'),
                "date": metadata.get('date')
            }

        df = pd.DataFrame.from_dict(node_data, orient="index")
        df = df.reset_index()
        df = df.rename(columns={df.columns[0]: "node_id"})

        return df

    # Extract maximum frequencies for both methods (using enriched files with internal node fitness)
    _freq_params = {
        "diffusion": {"tree": "simulated_flu_diffusion_enriched.json", "freq": "frequencies_diffusion.json", "key": "global"},
        "kde": {"tree": "simulated_flu_kde_enriched.json", "freq": "frequencies_kde.json", "key": "frequencies"}
    }

    freq_df_diffusion = get_freq_sum(_freq_params["diffusion"]["tree"], _freq_params["diffusion"]["freq"], freq_key=_freq_params["diffusion"]["key"])
    freq_df_kde = get_freq_sum(_freq_params["kde"]["tree"], _freq_params["kde"]["freq"], freq_key=_freq_params["kde"]["key"])
    return freq_df_diffusion, freq_df_kde


@app.cell
def _(mo):
    mo.md("""
    ### Diffusion Frequencies
    """)
    return


@app.cell
def _(freq_df_diffusion):
    freq_df_diffusion
    return


@app.cell
def _(mo):
    mo.md("""
    ### KDE Frequencies
    """)
    return


@app.cell
def _(freq_df_kde):
    freq_df_kde
    return


@app.function
def normalize_and_prepare_data(df):
    """
    Filter, normalize fitness, reorder columns, and sort frequency data.

    Args:
        df: DataFrame with max_frequency and fitness columns

    Returns:
        Processed DataFrame with normalized_fitness column
    """
    # Filter to nonzero maximum frequencies
    df = df[df['max_frequency'] > 0].copy()

    # Normalize fitness by maximum fitness
    df["normalized_fitness"] = df["fitness"] / df["fitness"].max()

    # Reorder columns to include normalized_fitness
    df = df[
        ["node_id", "node_type", "max_frequency", "generation", "fitness", "normalized_fitness", "date"]
    ]

    # Sort by max_frequency descending
    return df.sort_values("max_frequency", ascending=False).reset_index(drop=True)


@app.cell
def _(freq_df_diffusion, freq_df_kde):
    # Process both methods using the same helper function
    enriched_df_diffusion = normalize_and_prepare_data(freq_df_diffusion)
    enriched_df_kde = normalize_and_prepare_data(freq_df_kde)
    return enriched_df_diffusion, enriched_df_kde


@app.function
def calculate_correlation_statistics(data, np, pearsonr, spearmanr, r2_score):
    """
    Calculate all correlation statistics and regression parameters.

    Centralizes statistics calculation to follow DRY principle - used by both
    matplotlib and Altair plotting functions.

    Args:
        data: DataFrame with normalized_fitness and max_frequency columns
        np, pearsonr, spearmanr, r2_score: Required libraries

    Returns:
        dict: Statistics including r2, pearson_r, spearman_rho, slope, intercept, n_points
    """
    # Filter NaN values
    plot_data = data[data['normalized_fitness'].notna()].copy()

    x_data = plot_data['normalized_fitness']
    y_data = plot_data['max_frequency']

    # Calculate correlations
    pearson_r, _ = pearsonr(x_data, y_data)
    spearman_rho, _ = spearmanr(x_data, y_data)

    # Calculate regression
    coeffs = np.polyfit(x_data, y_data, 1)
    slope, intercept = coeffs[0], coeffs[1]

    # Calculate R²
    y_pred = slope * x_data + intercept
    r2 = r2_score(y_data, y_pred)

    return {
        'r2': r2,
        'pearson_r': pearson_r,
        'spearman_rho': spearman_rho,
        'slope': slope,
        'intercept': intercept,
        'n_points': len(plot_data)
    }


@app.function
def create_altair_correlation_plot(data, method_name, scatter_color, edge_color, stats, include_filter=True, alt=None, pd=None):
    """
    Create interactive Altair scatter plot with regression line and statistics.

    Args:
        data: DataFrame with normalized_fitness, max_frequency, and metadata columns
        method_name: Name of method for plot title (e.g., "Diffusion", "KDE")
        scatter_color: Color for scatter points
        edge_color: Color for scatter point edges
        stats: Dictionary of pre-calculated statistics from calculate_correlation_statistics()
        include_filter: Whether to include node_type dropdown filter (default True)
        alt: altair module
        pd: pandas module

    Returns:
        Altair Chart object with interactive features
    """
    # Filter NaN values
    plot_data = data[data['normalized_fitness'].notna()].copy()

    # Determine available tooltip fields
    tooltip_fields = [
        alt.Tooltip('node_id:N', title='Node ID'),
        alt.Tooltip('node_type:N', title='Type'),
        alt.Tooltip('normalized_fitness:Q', title='Normalized Fitness', format='.4f'),
        alt.Tooltip('max_frequency:Q', title='Max Frequency', format='.4f'),
        alt.Tooltip('fitness:Q', title='Raw Fitness', format='.2f'),
        alt.Tooltip('generation:Q', title='Generation'),
        alt.Tooltip('date:T', title='Date')
    ]

    # Add descendant_count to tooltip if available (for internal-node plots)
    if 'descendant_count' in plot_data.columns:
        tooltip_fields.append(alt.Tooltip('descendant_count:Q', title='Descendants', format='d'))

    # Create base scatter plot
    scatter = alt.Chart(plot_data).mark_circle(
        size=60,
        opacity=0.6,
        color=scatter_color,
        stroke=edge_color,
        strokeWidth=0.5
    ).encode(
        x=alt.X('normalized_fitness:Q', title='Normalized Fitness'),
        y=alt.Y('max_frequency:Q', title='Maximum Frequency'),
        tooltip=tooltip_fields
    )

    # Add interactive node_type filter if requested
    if include_filter and 'node_type' in plot_data.columns:
        node_type_dropdown = alt.binding_select(
            options=[None, 'internal', 'terminal'],
            name='Node Type: '
        )
        node_type_selection = alt.selection_point(
            fields=['node_type'],
            bind=node_type_dropdown
        )
        scatter = scatter.add_params(node_type_selection).transform_filter(
            node_type_selection
        )

    # Create regression line layer
    x_min, x_max = plot_data['normalized_fitness'].min(), plot_data['normalized_fitness'].max()
    regression_df = pd.DataFrame({
        'x': [x_min, x_max],
        'y': [stats['slope'] * x_min + stats['intercept'],
              stats['slope'] * x_max + stats['intercept']]
    })

    regression = alt.Chart(regression_df).mark_line(
        strokeDash=[5, 5],
        color='red',
        opacity=0.8,
        size=2
    ).encode(
        x=alt.X('x:Q', title='Normalized Fitness'),
        y=alt.Y('y:Q', title='Maximum Frequency')
    )

    # Create statistics text annotation (3 separate text marks for multi-line display)
    x_pos = plot_data['normalized_fitness'].min()
    y_pos = plot_data['max_frequency'].max()
    y_range = plot_data['max_frequency'].max() - plot_data['max_frequency'].min()
    y_spacing = y_range * 0.05

    stats_df = pd.DataFrame({
        'x': [x_pos] * 3,
        'y': [y_pos, y_pos - y_spacing, y_pos - 2*y_spacing],
        'text': [
            f"R² = {stats['r2']:.3f}",
            f"Pearson r = {stats['pearson_r']:.3f}",
            f"Spearman ρ = {stats['spearman_rho']:.3f}"
        ]
    })

    text = alt.Chart(stats_df).mark_text(
        align='left',
        baseline='top',
        dx=10,
        dy=10,
        fontSize=11,
        color='black'
    ).encode(
        x=alt.X('x:Q'),
        y=alt.Y('y:Q'),
        text='text:N'
    )

    # Combine layers and configure chart
    # The light theme is set globally, so we only need minimal configuration here
    chart = (scatter + regression + text).properties(
        width=600,
        height=450,
        title={
            'text': f"Normalized Fitness vs Maximum Frequency ({method_name})",
            'fontSize': 14,
            'fontWeight': 'bold'
        }
    ).interactive()

    return chart


@app.cell
def _(mo):
    mo.md("""
    ### Interactive Tables - Diffusion & KDE
    """)
    return


@app.cell
def _(mo):
    # Display both interactive tables side by side
    mo.md("""
    **Diffusion Method:**
    """)
    return


@app.cell
def _(enriched_df_diffusion, mo):
    mo.ui.dataframe(enriched_df_diffusion)
    return


@app.cell
def _(mo):
    mo.md("""
    **KDE Method:**
    """)
    return


@app.cell
def _(enriched_df_kde, mo):
    mo.ui.dataframe(enriched_df_kde)
    return


@app.function
def create_correlation_plot(data, method_name, scatter_color, edge_color, np, pearsonr, plt, r2_score, sns, spearmanr):
    """
    Create scatter plot with regression line and correlation statistics using seaborn.

    Args:
        data: DataFrame with normalized_fitness and max_frequency columns
        method_name: Name of method for plot title (e.g., "Diffusion", "KDE")
        scatter_color: Color for scatter points
        edge_color: Color for scatter point edges (note: seaborn regplot doesn't support edge colors directly)
        np, pearsonr, plt, r2_score, sns, spearmanr: Required libraries

    Returns:
        tuple: (pearson_r, r2, spearman_rho, figure)
    """
    # Filter to remove NaN values
    plot_data = data[data['normalized_fitness'].notna()].copy()

    # Set up seaborn style
    sns.set_style("whitegrid")
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params)

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Extract x and y data
    x_data = plot_data['normalized_fitness']
    y_data = plot_data['max_frequency']

    # Calculate correlations and R²
    pearson_r, _ = pearsonr(x_data, y_data)
    spearman_rho, _ = spearmanr(x_data, y_data)

    # Calculate R² using polyfit
    z = np.polyfit(x_data, y_data, 1)
    p = np.poly1d(z)
    y_pred = p(x_data)
    r2 = r2_score(y_data, y_pred)

    # Create seaborn regression plot
    sns.regplot(
        data=plot_data,
        x='normalized_fitness',
        y='max_frequency',
        ax=ax,
        scatter_kws={
            'alpha': 0.6,
            's': 40,
            'color': scatter_color,
            'edgecolors': edge_color,
            'linewidths': 0.5  # Note: plural form for matplotlib.scatter
        },
        line_kws={
            'color': 'red',
            'linestyle': '--',
            'alpha': 0.8,
            'linewidth': 2
        },
        ci=None  # Don't show confidence interval
    )

    # Add statistics text box
    stats_text = f"R² = {r2:.3f}\nPearson r = {pearson_r:.3f}\nSpearman ρ = {spearman_rho:.3f}"
    ax.text(0.05, 0.95, stats_text,
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

    # Format plot
    ax.set_xlabel("Normalized Fitness", fontsize=11)
    ax.set_ylabel("Maximum Frequency", fontsize=11)
    ax.set_title(f"Normalized Fitness vs Maximum Frequency ({method_name})", fontsize=12, fontweight='bold')

    plt.tight_layout()
    return pearson_r, r2, spearman_rho, ax.figure


@app.cell
def _(enriched_df_diffusion, np, pearsonr, r2_score, spearmanr):
    # Calculate statistics for Diffusion all-nodes (shared by both plot types)
    stats_diffusion = calculate_correlation_statistics(
        enriched_df_diffusion, np, pearsonr, spearmanr, r2_score
    )
    return (stats_diffusion,)


@app.cell
def _(enriched_df_diffusion, np, pearsonr, plt, r2_score, sns, spearmanr):
    # Create diffusion matplotlib plot
    pearson_r_diffusion, r2_diffusion, spearman_rho_diffusion, _fig_diffusion = create_correlation_plot(
        enriched_df_diffusion, "Diffusion", "steelblue", "navy", np, pearsonr, plt, r2_score, sns, spearmanr
    )
    _fig_diffusion
    return pearson_r_diffusion, r2_diffusion, spearman_rho_diffusion


@app.cell
def _(alt):
    def light_theme():
        return {
            "config": {
                "background": "white",
                "view": {"fill": "white"},
                "axis": {
                    "labelColor": "black",
                    "titleColor": "black",
                    "gridColor": "#e0e0e0",
                    "domainColor": "black",
                    "tickColor": "black",
                },
                "legend": {
                    "labelColor": "black",
                    "titleColor": "black",
                },
                "title": {
                    "color": "black",
                },
            }
        }

    alt.themes.register("light", light_theme)
    alt.themes.enable("light")
    return


@app.cell
def _(alt, enriched_df_diffusion, pd, stats_diffusion):
    # Create diffusion Altair interactive plot
    _chart_diffusion = create_altair_correlation_plot(
        enriched_df_diffusion, "Diffusion", "steelblue", "navy", stats_diffusion,
        include_filter=True, alt=alt, pd=pd
    )
    _chart_diffusion
    return


@app.cell
def _(enriched_df_kde, np, pearsonr, r2_score, spearmanr):
    # Calculate statistics for KDE all-nodes (shared by both plot types)
    stats_kde = calculate_correlation_statistics(
        enriched_df_kde, np, pearsonr, spearmanr, r2_score
    )
    return (stats_kde,)


@app.cell
def _(enriched_df_kde, np, pearsonr, plt, r2_score, sns, spearmanr):
    # Create KDE matplotlib plot
    pearson_r_kde, r2_kde, spearman_rho_kde, _fig_kde = create_correlation_plot(
        enriched_df_kde, "KDE", "darkorange", "darkred", np, pearsonr, plt, r2_score, sns, spearmanr
    )
    _fig_kde
    return pearson_r_kde, r2_kde, spearman_rho_kde


@app.cell
def _(alt, enriched_df_kde, pd, stats_kde):
    # Create KDE Altair interactive plot
    _chart_kde = create_altair_correlation_plot(
        enriched_df_kde, "KDE", "darkorange", "darkred", stats_kde,
        include_filter=True, alt=alt, pd=pd
    )
    _chart_kde
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Internal Nodes Only Analysis
    ### Examining correlation for inferred fitness values
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
def _(enriched_df_diffusion, json):
    # Get descendant counts for internal nodes
    _descendant_counts_diffusion = count_descendant_terminals("simulated_flu_diffusion_enriched.json", json)

    # Filter to internal nodes with at least 10 descendants
    _internal_df_diffusion = enriched_df_diffusion[enriched_df_diffusion['node_type'] == 'internal'].copy()
    _internal_df_diffusion['descendant_count'] = _internal_df_diffusion['node_id'].map(_descendant_counts_diffusion)
    _internal_df_diffusion = _internal_df_diffusion[_internal_df_diffusion['descendant_count'] >= 10].copy()

    internal_df_diffusion = _internal_df_diffusion
    return (internal_df_diffusion,)


@app.cell
def _(internal_df_kde):
    internal_df_kde
    return


@app.cell
def _(internal_df_diffusion, np, pearsonr, r2_score, spearmanr):
    # Calculate statistics for Diffusion internal-nodes (shared by both plot types)
    stats_diffusion_internal = calculate_correlation_statistics(
        internal_df_diffusion, np, pearsonr, spearmanr, r2_score
    )
    return (stats_diffusion_internal,)


@app.cell
def _(internal_df_diffusion, np, pearsonr, plt, r2_score, sns, spearmanr):
    # Create diffusion matplotlib plot for internal nodes with >=10 descendants
    pearson_r_diffusion_internal, r2_diffusion_internal, spearman_rho_diffusion_internal, _fig_diffusion_internal = create_correlation_plot(
        internal_df_diffusion, "Diffusion - Internal Nodes (≥10 descendants)", "steelblue", "navy", np, pearsonr, plt, r2_score, sns, spearmanr
    )
    _fig_diffusion_internal
    return


@app.cell
def _(alt, internal_df_diffusion, pd, stats_diffusion_internal):
    # Create diffusion Altair interactive plot for internal nodes
    _chart_diffusion_internal = create_altair_correlation_plot(
        internal_df_diffusion, "Diffusion - Internal Nodes (≥10 descendants)", "steelblue", "navy", stats_diffusion_internal,
        include_filter=False, alt=alt, pd=pd
    )
    _chart_diffusion_internal
    return


@app.cell
def _(enriched_df_kde, json):
    # Get descendant counts for internal nodes
    _descendant_counts_kde = count_descendant_terminals("simulated_flu_kde_enriched.json", json)

    # Filter to internal nodes with at least 10 descendants
    _internal_df_kde = enriched_df_kde[enriched_df_kde['node_type'] == 'internal'].copy()
    _internal_df_kde['descendant_count'] = _internal_df_kde['node_id'].map(_descendant_counts_kde)
    _internal_df_kde = _internal_df_kde[_internal_df_kde['descendant_count'] >= 10].copy()

    internal_df_kde = _internal_df_kde
    return (internal_df_kde,)


@app.cell
def _(internal_df_kde, np, pearsonr, r2_score, spearmanr):
    # Calculate statistics for KDE internal-nodes (shared by both plot types)
    stats_kde_internal = calculate_correlation_statistics(
        internal_df_kde, np, pearsonr, spearmanr, r2_score
    )
    return (stats_kde_internal,)


@app.cell
def _(internal_df_kde, np, pearsonr, plt, r2_score, sns, spearmanr):
    # Create KDE matplotlib plot for internal nodes with >=10 descendants
    pearson_r_kde_internal, r2_kde_internal, spearman_rho_kde_internal, _fig_kde_internal = create_correlation_plot(
        internal_df_kde, "KDE - Internal Nodes (≥10 descendants)", "darkorange", "darkred", np, pearsonr, plt, r2_score, sns, spearmanr
    )
    _fig_kde_internal
    return


@app.cell
def _(alt, internal_df_kde, pd, stats_kde_internal):
    # Create KDE Altair interactive plot for internal nodes
    _chart_kde_internal = create_altair_correlation_plot(
        internal_df_kde, "KDE - Internal Nodes (≥10 descendants)", "darkorange", "darkred", stats_kde_internal,
        include_filter=False, alt=alt, pd=pd
    )
    _chart_kde_internal
    return


@app.cell
def _(internal_df_kde):
    internal_df_kde
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Comparison of Diffusion vs KDE Methods
    """)
    return


@app.cell
def _(
    mo,
    pd,
    pearson_r_diffusion,
    pearson_r_kde,
    r2_diffusion,
    r2_kde,
    spearman_rho_diffusion,
    spearman_rho_kde,
):
    # Create comparison table
    comparison_data = {
        'Method': ['Diffusion', 'KDE'],
        'R²': [r2_diffusion, r2_kde],
        'Pearson r': [pearson_r_diffusion, pearson_r_kde],
        'Spearman ρ': [spearman_rho_diffusion, spearman_rho_kde]
    }
    comparison_df = pd.DataFrame(comparison_data)

    # Display table
    mo.md(f"""
    ### Statistical Comparison

    | Method | R² | Pearson r | Spearman ρ |
    |--------|-----|-----------|------------|
    | Diffusion | {r2_diffusion:.4f} | {pearson_r_diffusion:.4f} | {spearman_rho_diffusion:.4f} |
    | KDE | {r2_kde:.4f} | {pearson_r_kde:.4f} | {spearman_rho_kde:.4f} |

    **Best Method:** {'Diffusion' if r2_diffusion > r2_kde else 'KDE'} (based on R²)
    """)
    return


if __name__ == "__main__":
    app.run()
