import pandas as pd
import argparse


def main():
    parser = argparse.ArgumentParser(description="Process results in CSV file.")
    parser.add_argument("csv_file", type=str, help="Path to the CSV file")
    args = parser.parse_args()

    # Load the CSV file
    df = pd.read_csv(args.csv_file)

    # Create a new column '1-alpha'
    df["1-alpha"] = 1 - df["alpha"]

    # Group by 'covratio' and '1-alpha' columns
    grouped = df.groupby(["1-alpha", "covratio"])

    # Compute the average of 'coverage', 'stretch', and 'cp_quantile' columns
    result_avg = grouped[["coverage", "stretch", "cp_quantile"]].mean()
    result_std = grouped[["coverage", "stretch", "cp_quantile"]].std()

    # Format the results as a LaTeX table
    latex_table = ""
    for (cov_nominal, covratio), row in result_avg.iterrows():
        dataset = df["dataset"].unique()[0]
        if dataset.lower() in ["wbc", "oasis"]:
            dataset = dataset.upper()
        model = df["model"].unique()[0]
        if model.lower() == "universeg":
            model = "UniverSeg"

        if model.lower() == "pranet":
            model = "PraNet"

        std_row = result_std.loc[(cov_nominal, covratio)]
        emp_cov = row["coverage"]
        emp_stretch = row["stretch"]
        emp_cp_quantile = row["cp_quantile"]
        std_cov = std_row["coverage"]
        std_stretch = std_row["stretch"]
        std_cp_quantile = std_row["cp_quantile"]
        if latex_table:
            row = f"& & & ${cov_nominal}$ & ${covratio}$  & & {emp_cov:.3f} & {{\\scriptsize ({std_cov:.3f}) }} & {emp_stretch:.3f} & {{\\scriptsize ({std_stretch:.3f})}} & {emp_cp_quantile:.3f} & {{\\scriptsize ({std_cp_quantile:.3f})}} \\\\"
        else:
            row = f"{model} & {dataset} & & ${cov_nominal}$ & ${covratio}$  & & {emp_cov:.3f} & {{\\scriptsize ({std_cov:.3f}) }} & {emp_stretch:.3f} & {{\\scriptsize ({std_stretch:.3f})}} & {emp_cp_quantile:.3f} & {{\\scriptsize ({std_cp_quantile:.3f})}} \\\\"
        latex_table += row + "\n"

    print(latex_table)


if __name__ == "__main__":
    main()
