"""Download Metrica's public sample games, the validation set for everything in possession/.

The competition match has only 5 public minutes of ground truth, which cannot choose between two
versions of a method (EVALUATION.md proves it on the baselines). RULES.md allows data from other
matches in any amount, so hyperparameters are chosen on these games instead.

    python validation/download_open_data.py

Source: https://github.com/metrica-sports/sample-data -- anonymised, tracking and events in sync.
Its README asks only that public use acknowledges the source. Files land in external_data/, which
is gitignored, and existing files are not fetched again.
"""

import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# Params
BASE_URL = "https://raw.githubusercontent.com/metrica-sports/sample-data/master/data"
OUT_DIR = os.path.join(ROOT, "external_data", "metrica")
FILES = (
    "Sample_Game_1/Sample_Game_1_RawEventsData.csv",
    "Sample_Game_1/Sample_Game_1_RawTrackingData_Home_Team.csv",
    "Sample_Game_1/Sample_Game_1_RawTrackingData_Away_Team.csv",
    "Sample_Game_2/Sample_Game_2_RawEventsData.csv",
    "Sample_Game_2/Sample_Game_2_RawTrackingData_Home_Team.csv",
    "Sample_Game_2/Sample_Game_2_RawTrackingData_Away_Team.csv",
    "Sample_Game_3/Sample_Game_3_events.json",
    "Sample_Game_3/Sample_Game_3_metadata.xml",
    "Sample_Game_3/Sample_Game_3_tracking.txt",
)


def fetch(relative_path, out_dir=OUT_DIR):
    """Download one file unless it is already there. A partial file never keeps the final name."""
    target = os.path.join(out_dir, relative_path.replace("/", os.sep))
    if os.path.isfile(target):
        return target, False
    os.makedirs(os.path.dirname(target), exist_ok=True)
    partial = target + ".part"
    urllib.request.urlretrieve("{}/{}".format(BASE_URL, relative_path), partial)
    os.replace(partial, target)
    return target, True


if __name__ == '__main__':
    for path in FILES:
        target, fetched = fetch(path)
        print("{:<10} {}".format("fetched" if fetched else "present", os.path.relpath(target, ROOT)))
    sys.exit(0)
