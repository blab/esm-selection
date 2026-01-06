# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "altair==6.0.0",
#     "duckdb==1.4.2",
#     "marimo",
#     "matplotlib==3.10.7",
#     "nbformat==5.10.4",
#     "numpy==2.3.4",
#     "openai==2.8.0",
#     "pandas==2.3.3",
#     "polars[pyarrow]==1.35.2",
#     "pytest==9.0.1",
#     "scikit-learn==1.7.2",
#     "scipy==1.16.3",
#     "seaborn==0.13.2",
#     "sqlglot==27.29.0",
#     "vegafusion==2.0.3",
#     "vl-convert-python==1.8.0",
# ]
# ///

import marimo

__generated_with = "0.17.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.linear_model import LinearRegression
    from sklearn.decomposition import PCA
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    import matplotlib.pyplot as plt
    import seaborn as sns
    import marimo as mo
    import json
    from scipy.stats import pearsonr, spearmanr
    return (
        LinearRegression,
        OneHotEncoder,
        PCA,
        json,
        np,
        pd,
        pearsonr,
        plt,
        r2_score,
        spearmanr,
    )


@app.cell
def _(OneHotEncoder, np):
    # Define amino-acid alphabet and desired order
    AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")  # 20 standard AAs
    VOCAB_SIZE = len(AA_ORDER)

    # Create sklearn OneHotEncoder with fixed categories
    encoder = OneHotEncoder(
        categories=[AA_ORDER],   # enforce custom order
        sparse_output=False,     # return dense arrays (numpy)
        handle_unknown="ignore"  # unknowns become zero rows
    )

    # Fit encoder once on the amino-acid alphabet
    encoder.fit(np.array(AA_ORDER).reshape(-1, 1))


    def one_hot_encode_sklearn(seq):
        """
        Return (len(seq), VOCAB_SIZE) one-hot matrix using sklearn.
        Unknown residues -> zero row (via handle_unknown='ignore').
        """
        # Convert characters to a column vector of shape (L,1)
        chars = np.array(list(seq)).reshape(-1, 1)

        # Each row = encoding for that residue
        arr = encoder.transform(chars)

        return arr.astype(np.uint8)
    return (one_hot_encode_sklearn,)


@app.cell
def _(pd):
    # Load the CSV file with FASTA sequences
    csv_path = "next_tree~h3n2/epochs~1/learning_rate~5e-05/model~esm2_t33_650M_UR50D/time~2000/base/Max_Freq_Fasta_LL_ha.csv"
    df = pd.read_csv(csv_path)

    # Check for sequences with deletions
    sequences_with_deletions = df["sequence"].str.contains("-", na=False).sum()

    df.head()
    return (df,)


@app.cell
def _(df):
    # Filter out sequences containing deletions
    df_filtered = df[~df["sequence"].str.contains("-", na=False)].copy()

    # Check sequence lengths
    seq_lengths = df_filtered["sequence"].str.len()

    df_filtered.head()
    return (df_filtered,)


@app.cell
def _(df_filtered, one_hot_encode_sklearn):
    # Apply the sklearn-based one-hot encoding to filtered sequences
    encoded_sequences = df_filtered["sequence"].apply(one_hot_encode_sklearn).tolist()

    # Create a copy with encoded sequences added
    df_with_encoding = df_filtered.copy()
    df_with_encoding["one_hot"] = encoded_sequences
    return df_with_encoding, encoded_sequences


@app.cell
def _(df_with_encoding):
    df_with_encoding
    return


@app.cell
def _(encoded_sequences, np):
    # Create padded tensor for ML models
    max_length = max(seq.shape[0] for seq in encoded_sequences)
    vocab_size = encoded_sequences[0].shape[1]

    # Create padded tensor
    padded_tensor = np.zeros(
        (len(encoded_sequences), max_length, vocab_size), dtype=np.uint8
    )

    for i, seq in enumerate(encoded_sequences):
        seq_len = seq.shape[0]
        padded_tensor[i, :seq_len, :] = seq
    return (padded_tensor,)


@app.cell
def _(df_with_encoding):
    # Create summary statistics
    unique_sequences = df_with_encoding["sequence"].nunique()

    # Display first few sequences info
    df_summary = df_with_encoding[
        ["node", "max_frequency", "sequence", "log_likelihood"]
    ].head()
    df_summary
    return


@app.cell
def _(df_with_encoding):
    df_with_encoding
    return


@app.cell
def _(df_with_encoding, padded_tensor):
    # Prepare data for regression

    # Flatten the 3D padded tensor to 2D for sklearn
    # Shape: (n_samples, max_length × 20)
    X_flattened = padded_tensor.reshape(padded_tensor.shape[0], -1)

    # Extract target variable (max_frequency)
    y = df_with_encoding["max_frequency"].values
    return X_flattened, y


@app.cell
def _(X_flattened, np):
    np.set_printoptions(threshold=np.inf)
    print(X_flattened[0])
    return


@app.cell
def _(df_with_encoding, json, pd):
    # Extract time information from JSON file

    # Load the JSON file containing phylogenetic tree with time information
    json_path = "ha.json"
    with open(json_path, "r") as f:
        tree_data = json.load(f)


    # Function to recursively extract node information
    def extract_node_info(node, node_list):
        node_name = node.get("name", "")

        # Extract time information if available
        time_info = None
        if "node_attrs" in node and "num_date" in node["node_attrs"]:
            time_info = node["node_attrs"]["num_date"]["value"]

        node_list.append({"node": node_name, "time": time_info})

        # Recursively process children
        if "children" in node:
            for child in node["children"]:
                extract_node_info(child, node_list)


    # Extract all nodes and their time information
    all_nodes = []
    if "tree" in tree_data:
        extract_node_info(tree_data["tree"], all_nodes)

    # Convert to DataFrame
    node_time_df = pd.DataFrame(all_nodes)

    # Remove rows with missing time information
    node_time_df = node_time_df.dropna(subset=["time"])

    # Merge with our sequence data
    df_with_time = df_with_encoding.merge(node_time_df, on="node", how="left")
    return (df_with_time,)


@app.cell
def _(X_flattened, df_with_time, y):
    # Time-based train/test split (before 2000 vs after 2000)

    # Filter out sequences without time information
    valid_time_mask = df_with_time["time"].notna()
    df_time_filtered = df_with_time[valid_time_mask].copy()
    X_time_filtered = X_flattened[valid_time_mask]
    y_time_filtered = y[valid_time_mask]

    # Create time-based split: before 2000 for training, 2000+ for testing
    time_threshold = 2000.0
    train_mask = df_time_filtered["time"] < time_threshold
    test_mask = df_time_filtered["time"] >= time_threshold

    X_train = X_time_filtered[train_mask]
    X_test = X_time_filtered[test_mask]
    y_train = y_time_filtered[train_mask]
    y_test = y_time_filtered[test_mask]

    # Get time statistics for each split
    train_times = df_time_filtered[train_mask]["time"]
    test_times = df_time_filtered[test_mask]["time"]
    return X_test, X_train, y_test, y_train


@app.cell
def _(LinearRegression, X_train, y_train):
    # Fit Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    return (model,)


@app.cell
def _(
    X_test,
    X_train,
    model,
    pearsonr,
    plt,
    r2_score,
    spearmanr,
    y_test,
    y_train,
):
    # Visualize prediction results with correlation metrics

    # Make predictions and evaluate model performance
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate R² scores
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    train_pearson, _ = pearsonr(y_train, y_train_pred)
    test_pearson, _ = pearsonr(y_test, y_test_pred)

    train_spearman, _ = spearmanr(y_train, y_train_pred)
    test_spearman, _ = spearmanr(y_test, y_test_pred)

    # Set light theme styling
    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor='white')

    # Training set predictions
    ax1.scatter(y_train, y_train_pred, alpha=0.7, s=40, color='steelblue', edgecolors='navy', linewidth=0.5)
    ax1.plot(
        [y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 
        color='darkred', linestyle='--', linewidth=2
    )
    ax1.set_xlabel("Actual Max Frequency", fontsize=11)
    ax1.set_ylabel("Predicted Max Frequency", fontsize=11)
    ax1.set_title("Training Set (Pre 2000): Predicted vs Actual", fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.4, color='lightgray')
    ax1.set_facecolor('white')

    # Add metrics text box for training set
    train_text = f"R² = {train_r2:.3f}\nPearson = {train_pearson:.3f}\nSpearman = {train_spearman:.3f}"
    ax1.text(0.05, 0.95, train_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

    # Test set predictions
    ax2.scatter(y_test, y_test_pred, alpha=0.7, s=40, color='darkorange', edgecolors='darkred', linewidth=0.5)
    ax2.plot(
        [y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
        color='darkred', linestyle='--', linewidth=2
    )
    ax2.set_xlabel("Actual Max Frequency", fontsize=11)
    ax2.set_ylabel("Predicted Max Frequency", fontsize=11)
    ax2.set_title("Test Set (Post 2000): Predicted vs Actual", fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.4, color='lightgray')
    ax2.set_facecolor('white')

    # Add metrics text box for test set
    test_text = f"R² = {test_r2:.3f}\nPearson = {test_pearson:.3f}\nSpearman = {test_spearman:.3f}"
    ax2.text(0.05, 0.95, test_text, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

    plt.tight_layout()
    plt.show()
    return (y_test_pred,)


@app.cell
def _(y_test):
    y_test
    return


@app.cell
def _(y_test_pred):
    y_test_pred
    return


@app.cell
def _(PCA, X_test, X_train, np):
    # Apply PCA for dimensional reduction

    # Determine number of components to retain 95% variance
    pca_full = PCA()
    pca_full.fit(X_train)

    # Calculate cumulative explained variance
    cumsum_var = np.cumsum(pca_full.explained_variance_ratio_)
    n_components_95 = np.argmax(cumsum_var >= 0.95) + 1
    n_components_99 = np.argmax(cumsum_var >= 0.99) + 1

    # Apply PCA with 95% variance retention
    pca = PCA(n_components=n_components_95, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    return X_test_pca, X_train_pca, n_components_95, n_components_99


@app.cell
def _(n_components_95):
    n_components_95
    return


@app.cell
def _(n_components_99):
    n_components_99
    return


@app.cell
def _(LinearRegression, X_train_pca, y_train):
    # Fit Linear Regression model on PCA-reduced data
    model_pca = LinearRegression()
    model_pca.fit(X_train_pca, y_train)
    return (model_pca,)


@app.cell
def _(
    X_test_pca,
    X_train_pca,
    model_pca,
    pearsonr,
    plt,
    r2_score,
    spearmanr,
    y_test,
    y_train,
):
    # Visualize PCA model prediction results with correlation metrics

    y_train_pred_pca = model_pca.predict(X_train_pca)
    y_test_pred_pca = model_pca.predict(X_test_pca)

    # Calculate R² scores and correlations for PCA model
    train_r2_pca = r2_score(y_train, y_train_pred_pca)
    test_r2_pca = r2_score(y_test, y_test_pred_pca)

    train_pearson_pca, _ = pearsonr(y_train, y_train_pred_pca)
    test_pearson_pca, _ = pearsonr(y_test, y_test_pred_pca)

    train_spearman_pca, _ = spearmanr(y_train, y_train_pred_pca)
    test_spearman_pca, _ = spearmanr(y_test, y_test_pred_pca)

    # Set light theme styling
    plt.style.use('default')
    pca_fig, (pca_ax1, pca_ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor='white')

    # Training set predictions (PCA)
    pca_ax1.scatter(y_train, y_train_pred_pca, alpha=0.7, s=40, color='mediumseagreen', edgecolors='darkgreen', linewidth=0.5)
    pca_ax1.plot(
        [y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 
        color='darkred', linestyle='--', linewidth=2
    )
    pca_ax1.set_xlabel("Actual Max Frequency", fontsize=11)
    pca_ax1.set_ylabel("Predicted Max Frequency", fontsize=11)
    pca_ax1.set_title("PCA Training Set: Predicted vs Actual", fontsize=12, fontweight='bold')
    pca_ax1.grid(True, alpha=0.4, color='lightgray')
    pca_ax1.set_facecolor('white')

    # Add metrics text box for training set (PCA)
    pca_train_text = f"R² = {train_r2_pca:.3f}\nPearson = {train_pearson_pca:.3f}\nSpearman = {train_spearman_pca:.3f}"
    pca_ax1.text(0.05, 0.95, pca_train_text, transform=pca_ax1.transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

    # Test set predictions (PCA)
    pca_ax2.scatter(y_test, y_test_pred_pca, alpha=0.7, s=40, color='mediumpurple', edgecolors='darkblue', linewidth=0.5)
    pca_ax2.plot(
        [y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
        color='darkred', linestyle='--', linewidth=2
    )
    pca_ax2.set_xlabel("Actual Max Frequency", fontsize=11)
    pca_ax2.set_ylabel("Predicted Max Frequency", fontsize=11)
    pca_ax2.set_title("PCA Test Set: Predicted vs Actual", fontsize=12, fontweight='bold')
    pca_ax2.grid(True, alpha=0.4, color='lightgray')
    pca_ax2.set_facecolor('white')

    # Add metrics text box for test set (PCA)
    pca_test_text = f"R² = {test_r2_pca:.3f}\nPearson = {test_pearson_pca:.3f}\nSpearman = {test_spearman_pca:.3f}"
    pca_ax2.text(0.05, 0.95, pca_test_text, transform=pca_ax2.transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

    plt.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
