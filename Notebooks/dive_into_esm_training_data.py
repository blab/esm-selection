# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "altair==5.5.0",
#     "marimo",
#     "pandas==2.3.1",
#     "pyarrow==21.0.0",
#     "ty==0.0.1a15",
# ]
# ///

import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""# Setup""")
    return


@app.cell
def _(mo):
    mo.md(r"""## Import Libraries""")
    return


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import altair as alt
    return alt, mo, pd


@app.cell
def _(mo):
    mo.md(r"""## Import uniprot tsv""")
    return


@app.cell
def _(pd):
    uniprot_df = pd.read_csv("Dataframes/uniref_identity_0_9_AND_taxonomy_nam_2025_07_17.tsv", sep="\t")
    #uniprot_df = pd.read_csv("Dataframes/uniref_identity_0_5_AND_taxonomy_nam_2025_07_21.tsv", sep="\t")
    return (uniprot_df,)


@app.cell
def _(uniprot_df):
    uniprot_df
    return


@app.cell
def _(uniprot_df):
    year_re = r'\b(19[6-9]\d|200\d|201\d|202[0-5])\b'

    # extract first matching year (as string), convert to float/int, else NaN
    uniprot_df['year'] = (
        uniprot_df['Organisms']
          .str.extract(year_re, expand=False)
          .astype(float)
    )

    uniprot_df_no_null_year = uniprot_df.dropna(subset=['year'])
    return (uniprot_df_no_null_year,)


@app.cell
def _(uniprot_df):
    subtype_re  = r'\b(H\d+N\d+)\b'

    # extract first subtype (or NaN if none)
    uniprot_df['subtype'] = uniprot_df['Organisms'].str.extract(subtype_re, expand=False)

    uniprot_df_no_null_subtype = uniprot_df.dropna(subset=['subtype'])
    return (uniprot_df_no_null_subtype,)


@app.cell
def _(uniprot_df_no_null_year):
    uniprot_df_no_null_year
    return


@app.cell
def _(alt, uniprot_df_no_null_year):
    _chart = (
        alt.Chart(uniprot_df_no_null_year)
        .mark_bar()
        .encode(
            x=alt.X(field='year', type='nominal'),
            y=alt.Y(field='Cluster ID', type='nominal', aggregate='count'),
            tooltip=[
                alt.Tooltip(field='year', format=',.2f'),
                alt.Tooltip(field='Cluster ID', aggregate='count')
            ]
        )
        .properties(
            height=290,
            width='container',
            config={
                'axis': {
                    'grid': False
                }
            }
        )
            .properties(title="Year", width="container")
            .configure_view(stroke=None)
            .configure_axis(grid=False)
            .configure(background='white')
                .configure_axis(
                    grid=False,            # you already disabled grid
                    domainColor='black',   # the axis line
                    tickColor='black',     # tick marks
                    labelColor='black',    # tick labels
                    titleColor='black'     # axis titles
                )
            .configure_title(
                    color='black',    # title text color
                    fontSize=16       # optional: adjust size
            )
    )
    _chart
    return


@app.cell
def _(uniprot_df_no_null_subtype):
    uniprot_df_no_null_subtype
    return


@app.cell
def _(alt, uniprot_df_no_null_subtype):
    _chart = (
        alt.Chart(uniprot_df_no_null_subtype)
        .mark_bar()
        .transform_aggregate(count="count()", groupby=["subtype"])
        .transform_window(
            rank="rank()",
            sort=[
                alt.SortField("count", order="descending"),
                alt.SortField("subtype", order="ascending"),
            ],
        )
        .transform_filter(alt.datum.rank <= 10)
        .encode(
            y=alt.Y(
                "subtype:N",
                sort="-x",
                axis=alt.Axis(title=None),
            ),
            x=alt.X("count:Q", title="Number of records"),
            tooltip=[
                alt.Tooltip("subtype:N"),
                alt.Tooltip("count:Q", format=",.0f", title="Number of records"),
            ],
        )
        .properties(title="Top 10 Flu subtypes (uniref90)", width="container")
        .configure_view(stroke=None)
        .configure_axis(grid=False)
        .configure(background='white')
            .configure_axis(
                grid=False,            # you already disabled grid
                domainColor='black',   # the axis line
                tickColor='black',     # tick marks
                labelColor='black',    # tick labels
                titleColor='black'     # axis titles
            )
        .configure_title(
                color='black',    # title text color
                fontSize=16       # optional: adjust size
        )
    )
    _chart
    return


@app.cell
def _(uniprot_df):
    uniprot_df_H3N2 = uniprot_df[uniprot_df["subtype"] == "H3N2"]
    return (uniprot_df_H3N2,)


@app.cell
def _(alt, uniprot_df_H3N2):
    _chart = (
        alt.Chart(uniprot_df_H3N2)
        .mark_bar()
        .encode(
            x=alt.X(field='year', type='nominal', title='Year'),
            y=alt.Y(field='Cluster ID', type='nominal', aggregate='count', title='Sample Counts'),
            tooltip=[
                alt.Tooltip(field='year', format=',.2f'),
                alt.Tooltip(field='Cluster ID', aggregate='count')
            ]
        )
        .properties(
            height=290,
            width='container',
            config={
                'axis': {
                    'grid': False
                }
            },
            title="Counts of H3N2 samples by year (Uniref90)"
        )
        .configure(background='white')
        .configure_axis(
            grid=False,            # you already disabled grid
            domainColor='black',   # the axis line
            tickColor='black',     # tick marks
            labelColor='black',    # tick labels
            titleColor='black'     # axis titles
        )
        .configure_title(
            color='black',    # title text color
            fontSize=16       # optional: adjust size
        )
    )
    _chart
    return


@app.cell
def _(uniprot_df_no_null_subtype):
    uniprot_df_no_null_subtype
    return


if __name__ == "__main__":
    app.run()
