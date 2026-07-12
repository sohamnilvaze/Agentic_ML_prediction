import json
import pandas as pd


def save_csv_report(
    rows,
    output_path
):

    df = pd.DataFrame(rows)

    df.to_csv(
        output_path,
        index=False
    )


def save_json_report(
    report,
    output_path
):

    with open(
        output_path,
        "w"
    ) as f:

        json.dump(
            report,
            f,
            indent=4
        )