# Runbook for Cempaka

You are running a finished pipeline on match data that its author was not allowed to see. Your job
is to execute the steps below in order, stop when a step says stop, and report what you saw.

**Deadline: 17 November 2026.** Reach step 6 as early as you can, and no later than 14 November, so
there is time to fix whatever the real data turns up.

## Hard rules

Breaking any of these can disqualify the team.

1. **Run on this machine only.** Do not send the videos, the tracking, the tagging, the homography,
   anything in `cache/`, or any output CSV to a cloud service or a hosted LLM. If you are a hosted
   model, or you call one, **stop here and tell the team**: the terms ban you from touching this data.
2. **Never write, edit or correct a label.** Every label must come out of `make_submission.py`. Not
   one frame by hand, and not by you. Do not watch the video and adjust anything.
3. **Do not choose between versions using the public 5 minutes.** It has 17 possessions and picks the
   wrong answer even on the baselines (`EVALUATION.md`). Use it to detect faults, never to tune.
4. **Do not change parameters to improve a score.** `possession/params.json` and
   `possession/model.json` are fitted on public games and are final, unless the team decides
   otherwise after a reported fault.
5. **Nothing from `data/`, `cache/` or `evaluation/results/` may enter the public repository.** The
   only exception is the four committed `baseline_*.csv` files.

## What may leave this machine

The printed output of `check_data.py`, `make_submission.py` and `diagnose_public_window.py` is
aggregates only: counts, shares, scores, a checksum. Even so, the team has not yet confirmed that
summary numbers may be shared with a cloud assistant. **Give them to the team, and let the team
decide where they go.** Never print rows of the data to investigate a problem unless you are certain
you are local.

## Step 1. Put the data in place

The tactical data is missing from this machine. `data/` holds only one broadcast file.

Unpack the bundle outside the project folder, keeping its folder structure, then point
`MTAH_DATA_DIR` at it. Outside is safer, because searches run inside the project cannot touch it.
Two frame-level things still land inside the project and are covered by rule 5: `cache\` and
`evaluation\results\TEAMNAME.csv`. Both are ignored by git; neither may be read by a hosted model.

Before the first real run, delete any old parse cache. Rehearsals leave one behind, and the cache is
keyed on the file's size only:

```powershell
Remove-Item -Recurse -Force cache -ErrorAction SilentlyContinue
```

```powershell
$env:MTAH_DATA_DIR = "D:\mtah2027"     # must contain tactical\BAR-ATH_tactical_atd.txt and so on
```

Required: `tactical\BAR-ATH_tactical_atd.txt`, `tactical\BAR-ATH_tactical_atd.xml`,
`tactical\BAR-ATH_tactical_smart_tagging.xml`. The videos, homographies and broadcast files are not
used.

## Step 2. Set up the environment

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-lock.txt
.venv\Scripts\python -m unittest discover -s tests
```

Expected: `Ran 23 tests ... OK`. **If not: stop and report.**

## Step 3. Check the data

```powershell
.venv\Scripts\python check_data.py
```

Expected: every line `PASS`, ending `all checks passed`. The `info` lines are not pass or fail, but
report them all, because the pipeline's author has never seen these values.

| If this fails | It means | Do |
|---|---|---|
| line count, slots per line, lines split | the file is not the expected format | stop and report |
| ball visible, objects tracked | a different or damaged file | check the download, then report |
| parser agrees with baseline_raw.csv | **the most serious one**: frames or teams are read wrongly | stop and report; do not continue |
| home team is listed first | home and away may be swapped | stop and report |
| Smart Tagging entries, kick-off tag | the tag file differs from `DATA.md` | stop and report |
| any other line | the data or `games.py` is not what the code expects | stop and report |

**Worth noting:** the `info` line listing channels and sensors. If it shows anything about ball state,
ball owner or possession, report it straight away, because it would change the method.

The `info` lines quote small pieces of the gated files (team names, channel and sensor ids, tag
values). They go to the team, like everything else, and the team decides whether they go further.

## Step 4. Make the file

Replace `TEAMNAME` with the team name exactly as registered.

```powershell
.venv\Scripts\python make_submission.py --team TEAMNAME
.venv\Scripts\python evaluation\validate_submission.py evaluation\results\TEAMNAME.csv
```

Expected: `sanity checks passed`, then `Valid.` Keep the `sha256` line. Roughly what to expect:

| Measure | Plausible | Note |
|---|---|---|
| possessions | 150 to 600 | public games predict about 240 against about 380 true |
| mean possession | 6 to 30 s | |
| dead share | 0.10 to 0.60 | |
| public-window `spellF1` | above 0.3256 | below this, it loses to a baseline on its easiest stretch |

**A sanity failure means stop and report.** Do not submit, and do not adjust anything to make it pass.

## Step 5. Look for faults

```powershell
.venv\Scripts\python diagnose_public_window.py
```

Expected: `no part looks broken`. A value below 0.5 usually means a sign or team order is wrong
somewhere, so report the FAIL lines. This is a fault-finder only (rule 3).

## Step 6. Report to the team

Report in this form. It contains aggregates only.

```
Unit tests:            pass / fail
check_data.py:         full output, including the info lines
make_submission.py:    possessions, mean, median, dead share, public-window line, sha256
diagnose:              full output
Anything unexpected:   described in words, with no data rows
```

## Step 7. Prove it reproduces

The rules require the same file, not approximately the same. Run this only after the team has
committed the code and created its own public repository.

```powershell
git clone <the team's public repo> C:\temp\fresh
cd C:\temp\fresh
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-lock.txt
$env:MTAH_DATA_DIR = "D:\mtah2027"     # the same bundle as step 1
.venv\Scripts\python make_submission.py --team TEAMNAME --no-cache
```

The `sha256` must equal the one from step 4. **If it differs: stop and report.**

## Step 8. Before submission

Check each of these and report the result:

- [ ] `git ls-files` shows nothing from `data/` or `cache/`, and no `.mp4`, `.npz` or match-derived `.csv`
- [ ] The repository is public, and opening its URL in a logged-out browser works
- [ ] `evaluation\validate_submission.py` was run on the exact CSV being uploaded
- [ ] The CSV and the technical document are named after the team: `TEAMNAME.csv`, `TEAMNAME.pdf` or `.md`
- [ ] The `[TO FILL]` parts of `TECHNICAL_DOCUMENT.md` are completed
- [ ] A team member, not you, uploads through the Google Form sent by email

## What you are free to do

Run commands, read the code, read printed output, explain it to the team, and locally inspect data
if you are running entirely on this machine and the team asks. The method and how it was validated
are explained in `TECHNICAL_DOCUMENT.md`.
