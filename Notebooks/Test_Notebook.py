import marimo

__generated_with = "0.13.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return


@app.cell
def _():
    print("test")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
