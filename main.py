from __future__ import annotations

import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict


def load_daily_usage(csv_path: str | Path) -> List[Dict[str, str | float]]:
    """
    Read an electric interval CSV and return a list of daily usage totals.

    Output format:
        [{"date": "YYYY-MM-DD", "usage": <float_kwh>}, ...]
    """
    csv_path = Path(csv_path)

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        # Skip metadata lines until we reach the data header
        for line in f:
            if line.strip().startswith("Start Interval Date/Time"):
                header = line
                break
        else:
            raise ValueError("Data header not found in CSV.")

        reader = csv.DictReader(f, fieldnames=next(csv.reader([header])))

        totals: defaultdict[str, float] = defaultdict(float)
        for row in reader:
            start_ts = row.get("Start Interval Date/Time")
            usage = row.get("Usage")
            if not start_ts or not usage:
                continue

            # Convert to date string (YYYY-MM-DD)
            date_obj = datetime.strptime(start_ts.strip(), "%m/%d/%Y %H:%M").date()
            date_key = date_obj.isoformat()

            totals[date_key] += float(usage)

    return [{"date": d, "usage": totals[d]} for d in sorted(totals.keys())]


if __name__ == "__main__":
    data = load_daily_usage("local/electric.csv")
    print(data[:5])
