# Get-up: experiment design

Written after six runs and ~45M steps produced a reliable kneel and no stand. The narrative log
lives in the README; this is the plan, the evidence it rests on, and the rules for killing a run.

## What is actually established

| # | Finding | Evidence |
|---|---|---|
| F1 | The ragdoll is mechanically capable of standing and walking. | `Walker_Stage1`, 15M steps, walks. Not a torque or joint-limit problem. |
| F2 | Prone → kneel is learnable by RL from this reward. | `Walker_GetupOnly`, 20M steps: episode length 258 → 838 against a no-recovery floor of 120. |
| F3 | Kneel → stand has never once occurred. | `PeakPosture` ≤ 0.60 in every run that measured it. A stand is ≥ 0.85. |
| F4 | The reward is not inverted. | Standing pays ~2× kneeling per step, ~14× per episode once the kneel is terminal. |
| F5 | Removing 74% of the reward did not change the behaviour. | `posture_focus: 1.0` deleted the locomotion term; the crawl continued unchanged. Behaviours persist without reinforcement — only a better alternative displaces them. |
| F6 | A termination bar that nothing clears produces no gradient. | `Getup5` at bar 0.65 vs peak 0.59: every episode ended identically at 120 decisions, zero advantage variance, 450k steps of no movement. |
| F7 | A competent seed is destroyed in <100k steps by a failure-dominated buffer. | `Getup6` from `Stage1`: `TimeUpright` 0.094 → 0.011 by 100k at LR 3e-4. |
| F8 | Flat entropy is a reliable "policy is static" detector. | Frozen at 0.671 (`Getup3`) and 0.812 (`Getup5`) through long stuck stretches; rising 0.673 → 0.731 while `Getup4` was genuinely moving. |
| F9 | Mean posture is confounded by episode length. | Every episode opens with a low-posture transient, so shorter episodes drag the mean down at unchanged behaviour. Cost two misreadings. `PeakPosture` and `TimeUpright` are not confounded. |
| F10 | `measure: progress` curricula do not survive restarts. | `--initialize-from` resets the step counter, so lesson 0 replayed every run. The collapse ratchet sat inert for four runs and ~22M steps. |
| F11 | **`fallen_tilt` never controlled difficulty.** | `Quaternion.Euler(pitch, yaw, Random.Range(0f, 360f))` — the third argument is roll, sampled over the full circle *regardless of pitch*. "Tilt 0" meant upright pitch with uniformly random roll: on its side or fully inverted about as often as standing. Found by E0, fixed. |

### What F11 invalidates

Every tilt curriculum in this project. Lessons differed only in pitch while roll stayed fully
random, so no lesson was ever meaningfully easier than another — which is exactly why widening and
narrowing the range across five runs changed nothing. It also explains, without any appeal to
learning dynamics:

- **Why `Getup6` collapsed in under 100k steps.** A `Stage1` seed was never handed a pose it could
  stand from. Not catastrophic forgetting from a failure-dominated buffer — the buffer was
  failure-dominated because the task was impossible.
- **Why `GetupOnly` plateaued at a kneel.** From a uniformly random orientation, a kneel is a
  sensible universal intermediate. It was the right answer to the task actually posed.
- **Why the `cos²(tilt)` posture model kept mispredicting thresholds.** It assumed the tilt was the
  whole rotation.

The fix tips by `pitch` about a random **horizontal** axis — magnitude from the curriculum,
direction random — so `dot(hips.up, worldUp)` is now exactly `cos(pitch)` (verified numerically to
0 error) and start posture is `cos(pitch) × height ratio`. The model used for calibration is now
true rather than assumed.

**E0 must be re-run.** Its results below measured the broken start distribution.

## The structural flaw in every run so far

`OnEpisodeBegin` calls `bodyPart.Reset(...)` on all 16 parts — restoring the **authored standing
pose** — and then rotates `hips.rotation` by the sampled tilt. Nothing else changes.

So every start state this project has ever used is *the standing pose, rigidly rotated*. The
"reverse curriculum on tilt" interpolates between standing and lying down. **It never produces the
intermediate poses of a get-up** — hands-and-knees, kneeling, squatting, one knee up. Those states
have only ever been visited transiently, mid-episode, by a policy that then leaves them.

A reverse curriculum is supposed to start the agent near the goal and walk backwards along the
solution trajectory. Tilt does not parameterise that trajectory. It parameterises a rotation
through states that are not on it. That is why widening and narrowing the tilt range keeps
producing nothing: the missing skill is not "recover from a larger angle", it is "extend the legs
under load from a folded pose", and no tilt value ever starts the agent in a folded pose.

## The question

**Is kneel → stand reachable by exploration in this environment, or does it require
demonstrations?**

Everything so far has attacked this while changing reward, curriculum, seed, learning rate and
termination together. The experiments below change one thing at a time and each has a stated kill
criterion, decided before the run.

## Metrics

Primary, both added for this purpose and both immune to episode length:

- **`Walker/PeakPosture`** — highest posture reached in an episode. Answers *did it ever get up*.
- **`Walker/TimeUpright`** — fraction of physics steps above a fixed 0.7. Answers *did it stay up*.

Supporting: `Policy/Entropy` (stuck detector, F8), `Environment/Episode Length` (termination
diagnostic), `Policy/Extrinsic Value Estimate` (is the critic learning). **`Walker/Posture` mean is
demoted** — F9.

Overall success for the project: `TimeUpright` > 0.5 from prone starts.

## E0 — Measure the recovery frontier (no learning)

The one measurement never taken: *at what start tilt does each existing policy stop being able to
stay upright?* Everything downstream depends on it, and it has been guessed at three times.

`config_probe.yaml`, learning rate 1e-8 (policy effectively frozen), collapse cut disabled so
episodes run their full length, tilt stepped through 0 / 10 / 20 / 30 / 45 / 60 / 90 at 50k steps
each. ~350k steps, about four minutes.

Run it against both lineages:

```
mlagents-learn Assets\Python\config_probe.yaml --results-dir=results --run-id=Probe_Stage1    --initialize-from=Walker_Stage1    --torch-device=cuda
mlagents-learn Assets\Python\config_probe.yaml --results-dir=results --run-id=Probe_GetupOnly --initialize-from=Walker_GetupOnly --torch-device=cuda
```

Output: `TimeUpright` and `PeakPosture` as functions of tilt, for a policy that can only stand and
one that can only kneel. Defines **θ\*** = the largest tilt at which `Stage1` still holds
`TimeUpright` > 0.5.

**Also an instrument check.** If `Probe_Stage1` does not show `TimeUpright` > 0.8 at tilt 0, then
`--initialize-from` is not transferring the policy and every conclusion drawn from `Getup6` is
about a broken seed rather than about learning. That possibility has never been excluded, and it
costs four minutes to exclude.

**Kill:** `TimeUpright` < 0.8 at tilt 0 → stop. The problem is transfer, not learning; debug that
first.

## E1 — Can the seed survive fine-tuning at all?

Seed `Stage1`, tilt fixed at `[0, θ*]`, LR 1e-4, `max_steps` set to the length actually being run
so `progress` means something (F10). 2M steps. Collapse bar low enough that a standing policy
clears it and a collapsed one does not — set from E0, not guessed.

This deliberately asks nothing new of the policy. It is entirely a test of F7: whether
fine-tuning preserves a competent seed once the buffer is not dominated by failure.

- **Success:** `TimeUpright` at 2M ≥ its E0 value at θ\*.
- **Kill:** `TimeUpright` falls below half its E0 value at any point before 500k.

If this fails, no curriculum matters — the pipeline erases whatever it is given, and the fix is
optimisation-side (lower LR, smaller `num_epoch`, KL clipping) rather than task-side.

## E2 — Does the frontier extend?

Continue E1's policy, widen tilt one step past θ\*. 2M steps. Everything else held.

This is the actual scientific question in its smallest form: **can RL push the recovery frontier
outward by one increment?** If it can do it once it can probably do it repeatedly, and the answer
is a longer ramp. If it cannot do it once, no amount of ramp design will help.

- **Success:** `TimeUpright` at the new tilt exceeds its E0 baseline by 0.1 or more.
- **Kill:** entropy flat within ±0.005 and `PeakPosture` flat within ±0.02 over 1M steps (F8).

## E3 — Pose-space start states

The change the structural flaw above implies, and the one that has never been tried: start
episodes **in the intermediate poses of a get-up**, not in rotations of the standing pose.

Requires a C# change — an authored crouch pose (thighs and shins flexed, feet planted) applied at
`OnEpisodeBegin` under a `start_pose` environment parameter, alongside the existing tilt path. A
crouch is one leg extension away from standing, which makes it the true penultimate state of the
trajectory.

- **Success:** `PeakPosture` from crouch starts exceeds 0.8.
- **Kill:** no movement in `PeakPosture` over 2M steps.

**E3 failing is the decisive result.** If the agent cannot learn to stand from a crouch — a pose
one joint extension from the goal, with a reward that pays monotonically for the extension — then
kneel → stand is not reachable by exploration here, and E4 is the answer rather than a seventh
curriculum.

## E4 — Demonstrations (fallback, pre-committed)

The config's own decision rule, written before `GetupOnly` and now overdue: record a get-up with
the Demonstration Recorder, then BC pretraining plus GAIL. Reaching E4 is not a failure of the
project; it is the correct outcome if E3 says the behaviour is outside exploration's reach, and
deciding that in advance is what stops it becoming a seventh round of curriculum tuning.

## Rules for running these

1. **One variable per run.** The six runs so far moved reward, seed, curriculum, LR and
   termination together, which is why so few of them concluded anything.
2. **Instrument check at the first summary.** Every run has a stated 50k expectation. Verify it
   before leaving the run alone — F10 cost four runs to a mechanism that was never active, and
   `Getup6` cost 2.35M steps to a seed that may never have loaded.
3. **`max_steps` equals the length actually intended.** `progress` is `step / max_steps`, so a
   20M `max_steps` on a run stopped at 1M means the curriculum never starts.
4. **Kill criteria are written before the run, not after.**
5. **Verify a parameter reached the C# side, not just Python.** An `Environment/Lesson Number/<x>`
   tag only proves it was sent. Proof it was read is a quantity that must move if it took effect.
