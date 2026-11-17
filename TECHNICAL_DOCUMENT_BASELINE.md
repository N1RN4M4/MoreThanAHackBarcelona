# Frame-level ball possession: nearest player with a majority vote

**Team:** [TEAM NAME] · More than a Hack 2027, Phase 1

## Summary

Our submission is the organisers' reference baseline `baseline_smooth50`, unmodified. It is produced
by `starter/baseline.py` from the organisers' repository: each frame is given to the team of the
tracked player nearest the ball, and the result is smoothed with a centred majority vote of ±50
frames (±2 s at 25 fps). There is no model, no training and no parameter fitted by us.

We state this plainly because the file is not our own method. Its published full-match score is
`spellF1` 0.2433 and `macroF1` 0.6724 (`EVALUATION.md`), and we expect our entry to score exactly
that.

The file contains 199,319 rows: 72,957 `H` (36.6%), 54,842 `A` (27.5%), 71,520 `D` (35.9%) and no
`X`. It reports 669 possessions with a mean duration of 7.6 s.

## The approach, and why

**Step 1: nearest player.** For every frame where the automatic tracking locates the ball, the
script finds the nearest tracked player and assigns the frame to that player's team. Coordinates are
normalised 0 to 1, so both axes are scaled to metres before distances are compared; otherwise
distances along the short axis would be stretched. Frames with no ball position get no team.

**Step 2: majority vote.** Step 1 alone flickers whenever the ball passes near an opponent, which
cuts one real possession into many short ones. Each frame's label is therefore replaced by the
majority team over a centred window of ±50 frames (101 frames, about 4 s). A frame is `D` only when
no frame in its window has a team. Ties go to the home side.

**Why this window.** The organisers publish four variants that differ only in the window:

| Variant | Window | `macroF1` | `spellF1` |
|---|---|---|---|
| `baseline_raw` | none | 0.6532 | 0.0486 |
| `baseline_smooth25` | ±25 frames | 0.6752 | 0.1996 |
| `baseline_smooth50` | ±50 frames | 0.6724 | 0.2433 |
| `baseline_smooth125` | ±125 frames | 0.6356 | 0.2242 |

The ranking is on `spellF1`, and ±50 is the best of the four on the full match. On the public
5-minute window ±125 looks better (0.4242 against 0.3256), but that window holds only 17
possessions and the order reverses on the full match. We followed the published full-match figures
and did not tune anything on the public window.

**Why the baseline and not our own method.** We built a separate pipeline (logistic models over
tracking features, decoded into possessions with a Viterbi pass, fitted on Metrica's public sample
games). It was not run on the competition match before the deadline, so it produced no file we could
submit and it is not part of this entry. [CONFIRM OR REMOVE THIS PARAGRAPH]

## Assets used

| Asset | Used | How |
|---|---|---|
| Automatic tracking (ATD) of the tactical video, with the ball | Yes | Ball and player positions per frame; the metadata file gives each track slot's team |
| Smart Tagging | No | Not read by the baseline |
| Homography | No | Not needed; the tracking is already in pitch coordinates |
| Tactical and broadcast video | No | Not opened |
| Public 5-minute ground truth | No | Not used to choose or tune anything |
| External data or models | No | None |

The home team is the first team listed in the tracking metadata, which is Barcelona for this match.

## Possession rules implemented

- **In possession (`H` / `A`):** the team with the most nearest-player frames in the ±50-frame
  window around the frame.
- **Loose balls and challenges:** no special rule. The ball belongs to whichever side is nearest
  more often in the window, so a contested ball goes to the side that was closer for longer.
- **Dead ball (`D`):** only where the tracking has no ball position for the whole window. There is
  no rule for the whistle, restarts, or a ball that is visible but out of play.
- **Abstention (`X`):** never used.
- **Possession boundaries:** not modelled. A change of possession is wherever the majority flips.

## Limitations

- **Too many, too short possessions.** The file reports 669 possessions of mean 7.6 s. The public
  window has 17 possessions of mean 14.2 s in 5 minutes, which suggests the baseline still splits
  real possessions.
- **Weak dead-ball class.** `D` means "the tracker lost the ball", not "the ball is out of play". A
  ball that is visible while dead is given to a team, and a ball lost in live play can become `D`.
- **Track slots are not players.** The tracking has 46 slots for 22 players, so the nearest slot is
  a tracklet. Any team-assignment error in the tracking passes straight into the label.
- **No use of context.** Smart Tagging, team shape and the videos are ignored.

With more time we would run our own pipeline on the match, which models dead-ball stretches and
decodes whole possessions instead of smoothing frames. [CONFIRM OR REMOVE]

## Reproduction

Requirements: Python with `numpy` and `pandas` (`requirements.txt`), and the organisers' data bundle
unpacked as described in `DATA.md`. The data is not in the repository.

```
pip install -r requirements.txt
python starter/baseline.py
python evaluation/validate_submission.py evaluation/results/baseline_smooth50.csv
```

`starter/baseline.py` writes the four variants to `evaluation/results/`. Our submission is
`baseline_smooth50.csv` from that folder, renamed to `[TEAM NAME].csv` with no other change. The
same file is committed in the organisers' repository.

The script is deterministic. On Windows it writes CRLF line endings and on Linux or macOS LF; the
rows and labels are the same in both cases.
