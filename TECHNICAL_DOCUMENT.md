# Frame-level ball possession from automatic tracking

**Team:** WhiteSpace · More than a Hack 2027, Phase 1

## Summary

We infer who has the ball with a small, fully inspectable statistical pipeline. Three logistic models
turn the automatic tracking into per-frame evidence, and a Viterbi decoder turns that evidence into
whole possessions. The models and parameters were fitted only on Metrica's public sample games.
No model or parameter was fitted or tuned on the competition match's data or ground truth, and no
label was set by hand. The one thing taken from the competition match is published in the
organisers' repository: the baseline scores and the committed `baseline_raw.csv`, which set how
strongly the public games were degraded for validation (see "Noise calibration").

On public games held out from fitting, under noise calibrated to reproduce the published baseline
scores, the method reaches **`spellF1` 0.54 against 0.26** for `baseline_smooth50` on the same data.

**On the competition match:** 199 possessions, mean 11.8 s, median 9.5 s, dead share 0.630, public-window spellF1 0.2625 macroF1 0.6569 (15/17). The full-match score is not known to us.

## The approach, and why

`spellF1` counts possessions as objects, so the decision has to be made over intervals and not over
frames. The first question was therefore how to validate at all, because the public 5 minutes hold
17 possessions and cannot choose between two versions of a method.

**Validation on other matches.** `RULES.md` allows data from other matches, so all choices were made
on Metrica's three public sample games (same vendor, 25 fps; thanks to Metrica Sports for publishing
them: <https://github.com/metrica-sports/sample-data>). Frame labels were derived from their event
data: the team of the last on-ball event owns the ball, and each set piece ends a stoppage. The
result resembles the public ground truth: 3.8 to 4.2 possessions per minute against 3.4.

**Noise calibration.** Those games are corrected data, and their ball is almost never visible when
dead, while the competition match often shows it. Used as they are, they would make the problem look
easy. We therefore degrade them: ball gaps sampled from the real gap lengths (readable from the
committed `baseline_raw.csv`), a visible ball on some dead stretches, a detector locked on a false
hit, lost tracks and cross-team identity switches. The strengths were fitted so that the reference
baseline scores on the degraded games what it is published to score on the real match. This used no
gated data.

| Baseline window | `macroF1` degraded (real) | `spellF1` degraded (real) |
|---|---|---|
| none | 0.630 (0.653) | 0.032 (0.049) |
| ±25 | 0.643 (0.675) | 0.229 (0.200) |
| ±50 | 0.641 (0.672) | 0.276 (0.243) |
| ±125 | 0.609 (0.636) | 0.237 (0.224) |

The fit also reproduces the shape of the problem: `spellF1` peaks at ±50 and `macroF1` at ±25, as on
the real match. This table uses the calibration's noise seed; the results tables below use a separate
held-out seed, which is why `baseline_smooth50` reads 0.276 here and 0.262 there.

**The method.**

1. *Team, ball seen* (47 columns): distances from the ball to each side's nearest player, players
   moving with the ball, majority of the nearest side over several windows, the ball's half, and the
   shape columns below.
2. *Team, ball unseen* (30 columns): the difference between the teams in spread, height, speed,
   forward movement and distance to the nearest opponent, over centred windows of about 1, 4 and 10 s
   (half-widths of 12, 50 and 125 frames), plus who last and
   next controlled the ball. This matters because nearly all missing frames are in long gaps: gaps of
   5 s or more make up 99.7% of them, so interpolation cannot bridge them.
3. *Dead ball* (29 columns): collective speed, how long the ball has been unseen, the ball near the
   lines or still, and team separation.
4. *Decoder:* Viterbi over H, A and D, with costs for switching team and for entering and leaving a
   stoppage. Smart Tagging is added as extra evidence, not copied across as labels.

Every team column is "home minus away", and those models have no intercept, so they cannot learn a
home bias. A unit test confirms that swapping the teams flips every team column exactly. This is what
allows a model fitted on other clubs to apply here.

## Measured results

All are leave-one-game-out on degraded public games. A gap under about 0.04 is noise
(`EVALUATION.md`).

**The honest headline** (`validation/tune.py`): parameters chosen on two games, scored on the third.

| Noise scenario | `spellF1` |
|---|---|
| calibrated | 0.537 |
| more dead balls visible | 0.519 |
| twice the ball lost in play | 0.517 |

**What each part contributes** (`validation/evaluate.py`, calibrated noise, committed parameters):

| Variant | `spellF1` | `macroF1` |
|---|---|---|
| `baseline_smooth50` | 0.262 | 0.637 |
| **full method** | **0.543** | **0.816** |
| no team shape | 0.475 | 0.794 |
| no Smart Tagging | 0.493 | 0.767 |
| poor simulated Smart Tagging | 0.510 | 0.792 |
| no ball-half column | 0.532 | 0.813 |
| no carried control | 0.537 | 0.812 |

Per-game 90% bootstrap intervals for the full method are 0.47 to 0.57, 0.53 to 0.62 and 0.49 to 0.59.

Giving the method perfect knowledge of one part shows where the remaining error lies. Perfect team
knowledge reaches about 0.80 to 0.83, and perfect dead-ball knowledge about 0.59 to 0.62. Team
evidence is the larger gap.

## Assets used and dropped

| Asset | Use |
|---|---|
| Tactical ATD | the main input; positions only, with speeds derived ourselves so features match the validation games |
| Smart Tagging | decoder evidence for team and restarts, and marks everything before a half's kick-off as dead |
| Homography | not used |
| Tactical video | not used |
| Broadcast video and tracking | not used; no alignment is provided, and aligning it is a project of its own |
| Public 5-minute ground truth | a correctness check only; not used for fitting or for choosing anything |

Tried and dropped, because held-out results did not justify them:

- **Gradient boosting** in place of logistic regression: similar accuracy, usually worse calibration.
- **A second-stage model on evidence trends:** at most about 0.02, even with its parameters searched
  on the game it was scored on.
- **Explicit touch detection** from changes in the ball's velocity: 0.001.

## Possession rules implemented

- **A loose ball stays with the last team that had it.** Evidence is carried from who last controlled
  the ball, and on a tie the decoder places a change at the new team's first evidence.
- **A touch without control** should not change possession: it does not outweigh the switching cost
  unless the evidence holds. Very short real possessions are lost the same way (see below).
- **Restarts.** A tagged set piece names the restarting team, except throw-ins, which the tagging
  marks `N/A`. Those are left to the tracking.
- **The second half's lead-in** before kick-off is dead, as `FAQ.md` warns, using the tagged kick-off.
- **`X` is never written on graded frames.** The scorer would fill it with the previous label, which
  the decoder already does where appropriate, and it would cost `macroF1`. Frames outside the halves
  are written as `X`.

## Limitations

- **Short possessions are mostly missed.** On public games under 2 s we recover only about 3 to 11%
  of them, and we predict about 240 possessions against about 380. The method prefers whole, correct
  possessions over catching every brief one, because that scored best. Some cannot be recovered from
  tracking: in the public window, a 1.3 s Barcelona possession has Athletic nearest the ball
  throughout.
- **The validation labels are not the competition's labels.** Ours come from events and theirs from
  corrected tracking, so a real difference in where a possession starts would not show up here.
- **Smart Tagging's contribution is simulated.** The public games have no tags, so the gain above
  comes from imitation tags with invented error rates. The method still gains with poor ones, but the
  real figure is unknown.
- **Only three pairs of teams informed the fit,** none as dominant as a typical Barcelona side. In
  six-minute stretches where one side held at least 60% of live play, accuracy did not fall and we saw
  no bias against the dominant side, but only 20 such stretches exist.
- **With more time:** a better detector of opposition control, and using the tactical video to recover
  the ball on live frames, where the perfect-team result says most of the gain remains.

## Reproduction

Python 3.13.6, with the data bundle available. No GPU, no network, and no public sample games are
needed to regenerate the submission.

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-lock.txt
$env:MTAH_DATA_DIR = "D:\path\to\bundle"                          # or unpack into .\data

.venv\Scripts\python check_data.py                                # the data matches what is expected
.venv\Scripts\python make_submission.py --team <team_name>        # evaluation\results\<team_name>.csv
.venv\Scripts\python evaluation\validate_submission.py evaluation\results\<team_name>.csv
```

macOS or Linux:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
export MTAH_DATA_DIR=/path/to/bundle                              # or unpack into ./data

.venv/bin/python check_data.py
.venv/bin/python make_submission.py --team <team_name>
.venv/bin/python evaluation/validate_submission.py evaluation/results/<team_name>.csv
```

Use `requirements-lock.txt`, not the organisers' unpinned `requirements.txt`, which is for their
starter notebooks.

**Expected checksum of the file:** d73523986d9cffa1b22e228e47e3a24aa138703b02028af59217c4e2f784d924

The pipeline has no random component: models are fitted exactly by Newton's method, and evidence is
rounded before decoding. On a full-size rehearsal bundle in the competition's format, the output was
byte-identical across a fresh parse, a single thread, and AVX2 and AVX-512 disabled. We have not
tested a different operating system.

To regenerate the fitted files from public data (not needed for the submission):

```bash
python validation/download_open_data.py
python validation/tune.py --trials 1000     # possession/params.json, plus the nested result
python validation/train_final.py            # possession/model.json
python validation/evaluate.py               # the contribution table
python -m unittest discover -s tests
```

## Use of LLMs

The method and code were developed by the team, working only from this repository's public materials
and the public sample games. No match data, video or extract of them was provided to any AI assistant.
Runs on the match data were performed locally. No LLM takes part in producing the labels.
