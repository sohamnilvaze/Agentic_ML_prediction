import json
from pathlib import Path
import pandas as pd

from core.utils import ensure_directory_exists


def save_csv_report(
    rows,
    output_path
):

    output_path = Path(output_path)
    ensure_directory_exists(output_path.parent)

    df = pd.DataFrame(rows)

    df.to_csv(
        output_path,
        index=False
    )


def save_json_report(
    report,
    output_path
):

    output_path = Path(output_path)
    ensure_directory_exists(output_path.parent)

    with open(
        output_path,
        "w"
    ) as f:

        json.dump(
            report,
            f,
            indent=4
        )
