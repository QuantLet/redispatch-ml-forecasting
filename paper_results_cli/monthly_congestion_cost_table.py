"""Generate a yearly congestion-cost table from monthly ENTSO-E data.

Example
-------
python -m paper_results_cli.monthly_congestion_cost_table \
    --input_file data/redispatching_costs/entsoe_redispatching_congestion_costs.csv \
    --output_file paper_results/tables/monthly_congestion_cost_table.tex
"""

import argparse
from pathlib import Path

import pandas as pd


TSO_NAME_MAPPING = {
    "price_DE_50HzT": "50Hertz",
    "price_DE_Amprion": "Amprion",
    "price_DE_TenneT": "TenneT DE",
    "price_DE_TransnetBW": "TransnetBW",
}
START_YEAR = 2023
END_MONTH_YEAR = "2026-06"


def get_args():
    parser = argparse.ArgumentParser(
        description="Generate a yearly congestion-cost table from monthly data."
    )
    parser.add_argument(
        "--input_file",
        type=Path,
        required=True,
        help="Path to the input CSV file containing redispatching congestion costs.",
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        required=True,
        help="Path to the output tex file where the monthly congestion cost table will be saved.",
    )
    return parser.parse_args()


def build_table(df):
    """Return TSO-by-year costs in EUR millions."""
    cost_columns = list(TSO_NAME_MAPPING)
    # Remove repeated observations with slightly different interval boundaries.
    df = df.drop_duplicates(subset=cost_columns).copy()
    df["year"] = df["end_date"].dt.year
    df = df.loc[df["year"] >= START_YEAR]
    df = df.loc[df["end_date"] <= pd.to_datetime(END_MONTH_YEAR) + pd.offsets.MonthEnd(0)]

    table = df.groupby("year", sort=True)[cost_columns].sum().T / 1e6
    table = table.rename(index=TSO_NAME_MAPPING)
    table = table.reindex(TSO_NAME_MAPPING.values())
    table.index.name = "TSO"
    table.columns = pd.MultiIndex.from_product(
        [["Costs (EUR million)"], table.columns.astype(str)],
        names=[None, None],
    )
    return table


def covered_period(df):
    """Describe the months represented by records from START_YEAR onward."""
    included = df.loc[df["end_date"].dt.year >= START_YEAR]
    first_month = included["end_date"].min().strftime("%B %Y")
    last_month = (included["end_date"].max() - pd.Timedelta(days=1)).strftime("%B %Y")
    return f"{first_month}--{last_month}"


def main():
    args = get_args()
    df = pd.read_csv(args.input_file, parse_dates=["begin_date", "end_date"])

    table = build_table(df)
    caption = (
        f"Redispatching costs by TSO, {covered_period(df)}. "
        "Source: ENTSO-E, Article 13.1.C."
    )
    latex = table.to_latex(
        caption=caption,
        column_format="l" + "r" * len(table.columns),
        float_format="%.0f",
        multicolumn_format="c",
        position="ht",
    )
    years = [str(year) for year in table.columns.get_level_values(-1)]
    latex = latex.replace(
        " & " + " & ".join(years) + r" \\",
        "TSO & " + " & ".join(years) + r" \\",
        1,
    )
    latex = "\n".join(
        line for line in latex.splitlines() if not line.startswith("TSO &  &")
    ) + "\n"
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(latex, encoding="utf-8")


if __name__ == "__main__":
    main()
