import os
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats


def main(input_file, segment):
    # Load data
    df = pd.read_csv(input_file)

    # Calculate Pearson correlation and linear regression slope
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        df["max_frequency"], df["log_likelihood_y"]
    )

    # Set seaborn theme
    sns.set_theme(style="whitegrid")
    sns.set_style("ticks")

    # Create scatter plot with regression line
    plt.figure(figsize=(7, 7))
    ax = sns.regplot(
        data=df,
        x="max_frequency",
        y="log_likelihood_y",
        scatter_kws={"s": 50, "alpha": 0.35},
        line_kws={"color": "red"},
    )

    # Annotate correlation coefficient and slope on the plot
    textstr = f"$r = {r_value:.2f}$\nSlope = {slope:.2f}"
    plt.text(
        0.82,
        0.08,
        textstr,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.0),
    )

    # Customize plot
    ax.set_title("LL vs Max Frequency for HA", fontsize=16)
    ax.set_xlabel("Max Frequency", fontsize=12)
    ax.set_ylabel("Log Likelihood", fontsize=12)

    # Save plot
    plt.tight_layout()
    plt.savefig(f"Max_Freq_Vs_LL_{segment}.png", dpi=300)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--Max-Freq-LL",
    )

    parser.add_argument(
        "--segment",
    )

    args = parser.parse_args()

    main(args.Max_Freq_LL, args.segment)
