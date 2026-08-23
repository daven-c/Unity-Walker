# Unity Walker

An experiment in training a simulated ragdoll robot with [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents) to stand up, walk, run, and chase down a moving target — using reinforcement learning (PPO) instead of hand-authored animation or control logic.

## Demo

<video src="media/MLWalker.mp4" controls width="720">
  Your browser doesn't support embedded video —
  <a href="media/MLWalker.mp4">watch/download it here</a>.
</video>

## Goals

- **Balance** — keep a multi-jointed ragdoll upright under physics simulation.
- **Walk / run** — match a commanded target speed as efficiently as possible.
- **Chase a moving target** — steer toward and reach a target that relocates around the arena.
- **Recover from falls** — get back up after toppling instead of the episode just resetting.

## Project layout

The Unity project lives directly at the repo root:

```
Assets/
├── Examples/Walker/          # Walker scenes, ragdoll prefabs, WalkerAgent.cs
├── Examples/SharedAssets/    # Shared ML-Agents example scripts (joints, targets, sensors, etc.)
├── ML-Agents/                # Training timers
└── Python/                   # Training config + Python requirements
Packages/                     # Unity package manifest (ML-Agents, URP, Input System, etc.)
ProjectSettings/
results/                      # Training runs. Deliberately OUTSIDE Assets/ - see below
```

## The MDP

Everything below is defined in `WalkerAgent.cs` ([source](Assets/Examples/Walker/Scripts/WalkerAgent.cs)), which drives a 16-body-part ragdoll (hips, chest, spine, head, arms, legs) through `ConfigurableJoint` targets.

Formally this is a **POMDP** rather than a clean MDP — the policy sees the summary below, not the full physics state (contact forces, joint velocities, the target's own motion), so some of what determines the next state is hidden from it.

Most observations are expressed in the space of an **orientation cube**: a stabilized transform that tracks the hips' position but points at the target with a level horizon. Working in that frame rather than world space means "forward" always means "toward the target," so the policy doesn't have to relearn the same gait for every compass heading.

### State — 255 continuous values

243 vector observations from `CollectObservations`, plus 12 from four ray sensors (`m_UseChildSensors: 1`).

Global (18):

| Observation | Floats |
|---|---|
| Distance between goal velocity and actual average velocity | 1 |
| Average body velocity, in cube space | 3 |
| Goal velocity, in cube space | 3 |
| Rotation from `hips.forward` to cube forward (quaternion) | 4 |
| Rotation from `head.forward` to cube forward (quaternion) | 4 |
| Target position, in cube space | 3 |

Per body part, ×16 (10 each = 160):

| Observation | Floats |
|---|---|
| Touching ground (bool) | 1 |
| Linear velocity, cube space | 3 |
| Angular velocity, cube space | 3 |
| Position relative to hips, cube space | 3 |

Plus, for the 13 parts that are not hips/handL/handR (5 each = 65): local rotation quaternion (4) and `currentStrength / maxJointForceLimit` (1).

**18 + 160 + 65 = 243**, matching `VectorObservationSize` on Behavior Parameters. `NumStackedVectorObservations: 1` — no frame stacking, so the policy has no memory of previous steps beyond what's in the current observation.

**Ray sensors (12).** Four `RayPerceptionSensor3D` components — `RayPerceptionSensorL0/L1/R0/R1` — each contributing `(2 × raysPerDirection + 1) × (detectableTags + 2) = 1 × 3 = 3` floats. All four are configured identically: a **single** ray, `m_RayLength: 1`, detecting only the **`ground`** tag. They're short-range foot contact probes, not perception of the wider world.

That last point matters for a multi-agent scene: **an agent cannot observe other agents.** The vector observations are entirely self-referential (its own body parts, its own target, its own orientation cube), and the ray sensors only report `ground` within 1 unit. Each ragdoll on its own platform is effectively an independent environment that happens to share a policy.

### Action — 39 continuous values

All in `[-1, 1]`, remapped in `JointDriveController`. **Rotations (26):** each joint's target angle is lerped across that joint's configured angular limits, so the policy can't command anatomically impossible poses.

| Joint | Axes | Floats |
|---|---|---|
| chest, spine | X, Y, Z | 6 |
| footL, footR | X, Y, Z | 6 |
| thighL, thighR | X, Y | 4 |
| armL, armR | X, Y | 4 |
| head | X, Y | 2 |
| shinL, shinR | X | 2 |
| forearmL, forearmR | X | 2 |

**Strengths (13):** one per joint (chest, spine, head, thighL/R, shinL/R, footL/R, armL/R, forearmL/R), scaling that joint's `maximumForce` up to `maxJointForceLimit` (20000). This is why the action-rate penalty matters — the policy can slam every joint to full force and pay nothing for it.

### Reward

Per **physics step** (`FixedUpdate`):

```
matchSpeed × lookAtTarget × posture   +   0.1 × posture
```

| Term | Definition | Range |
|---|---|---|
| `matchSpeed` | `(1 − (‖v_avg − v_goal‖ / v_target)²)²` — sigmoid-ish decay from 1 to 0 as average body velocity deviates from the goal | 0–1 |
| `lookAtTarget` | `(dot(cubeForward, headForward_flattened) + 1) / 2` | 0–1 |
| `posture` | `clamp01(dot(hips.up, worldUp)) × clamp01(height / standingHeight)`, where `height` is head-to-lowest-foot and `standingHeight` is captured from the authored pose in `Initialize` | 0–1 |

Per **physics step**, additionally: `postureShapingWeight × (postureₜ − postureₜ₋₁)` — **potential-based shaping** over posture, weight 50, as a pure difference (γ = 1).

The plain `0.1 × posture` term is a *level* reward: it says where the ragdoll is, not whether it's improving. A prone agent scores ~0 and keeps scoring ~0 until it's already most of the way up, so the first half of every get-up attempt is unrewarded and exploration has nothing to follow. The shaping term pays for the *change* instead, densely, from the first degree of progress. It runs negative while posture falls, so it penalizes falling as well as rewarding recovery. It telescopes exactly to `k × (posture_end − posture_start)` — 0 for standing to standing, +50 for recovering, −50 for falling — so it can't be farmed by oscillating.

**Use γ = 1, not the discounted Ng et al. form.** `k(γ·Φ′ − Φ)` sums over an episode to `k[(γ−1)·ΣΦ + Φ_N − Φ₀]`, and that drift is not negligible when the term is added per *physics* step: at γ = 0.995 and k = 50 it came to −0.25/step merely for being upright, against a posture reward of 0.1/step. Simulated over a full episode it scored **−999.8 for staying standing and −1,168.6 for successfully recovering** — the behavior being taught scored *worse* than never getting up. The pure difference gives up the strict policy-invariance guarantee (which assumes the discounted form) for magnitudes that mean what they should.

Per **decision** (`OnActionReceived`): `− actionRatePenalty × mean(|aₜ − aₜ₋₁|)`, discouraging twitchy joint commands.

On **target contact**: `+1` (`TouchedTarget`).

Two design notes worth carrying forward:

- **Posture multiplies, it doesn't add.** `matchSpeed` is computed from *average body velocity* and `lookAtTarget` from *head yaw* — a ragdoll worming along the ground satisfies both. An earlier version made posture a `+0.1` additive bonus, and the agent converged on crawling, because crawling banked ~1.0/step and was far easier than bipedal gait. Gating means crawling now pays ≈0.
- **The standalone `0.1 × posture` term is the get-up signal.** Once the gated locomotion term is ~0 for a fallen agent, this is the only gradient telling it that standing beats lying there.

### Transitions and episode structure

- **Dynamics:** Unity `PhysX` rigidbody simulation. `DecisionPeriod: 5` with `TakeActionsBetweenDecisions: 0` — the policy acts every 5th physics step and the command is held in between.
- **Episode length:** `MaxStep: 5000` counts *physics* steps, not decisions — `Agent.StepCount` increments once per academy step, which is once per `FixedUpdate`. At `fixedDeltaTime` 0.02 that's **100 simulated seconds**, and at `DecisionPeriod: 5`, **~1000 decisions** per episode. Useful for reading reward magnitudes: reward is added every physics step and peaks around 1.1, so the practical ceiling is ~5,500 per episode.
- **Termination — stage dependent.** ML-Agents bootstraps the value estimate at a step-limit cutoff rather than treating it as terminal, so timing out is never implicitly punished. Ground contact is the switch between the two curriculum stages:
  - *Stage 1 (complete):* `agentDoneOnGroundContact: 1` on the 12 non-ground parts — falling ends the episode.
  - *Stage 2 (current):* all 16 set to `0`, so falling is a recoverable state.

  Stage 2 also **ramps how far fallen starts tip the ragdoll**, via the `fallen_tilt` environment parameter driven by the curriculum in `config.yaml` (30° → 60° → 90°, advancing on reward). Fully prone was always the hardest case, and starting there gave exploration nothing to climb; a shallow lean is a stumble the walking policy can already half-correct. `WalkerAgent` reads it through `Academy.EnvironmentParameters` and falls back to the Inspector's `fallenStartTilt` when no curriculum is supplied.

  Stage 2 adds a second exit: if `postureReward` stays below the collapse bar for `collapsedStepLimit` consecutive physics steps (600, ~12 simulated seconds), the episode ends via **`EndEpisode()`**. The counter resets the instant posture recovers, so an agent making progress keeps its time. The bar is ramped by the `collapse_posture` curriculum (0.2 → 0.7) rather than fixed, because a fixed bar becomes the definition of "good enough" — see log entry 14. This does *not* undo stage 1: ground contact still never terminates, and this fires only after 12 seconds of failing to rise, which is a genuine failure and so is labelled terminal (zero future value) rather than bootstrapped.

  The 4 parts that are **never** set to `1` are `footL`, `footR`, `shinL`, `shinR` — feet and shins legitimately touch the ground while walking, so terminating on their contact would end every episode instantly. The other 12 are `hips`, `spine`, `chest`, `head`, `upper_arm_L/R`, `lower_arm_L/R`, `hand_L/R`, `thighL/R`. Identify them by name rather than by line number when switching stages: the flags appear 16 times in the prefab YAML in a non-obvious order, and the block shifted by 3 lines when `Inference`/`fallenStartProbability`/`actionRatePenalty` were serialized onto it.
- **Initial state distribution** (`OnEpisodeBegin`): body parts reset to the authored pose and hips given a uniformly random yaw. With probability `fallenStartProbability` the ragdoll is tipped over (pitch 70–110°, random roll). That value is **0 in stage 1** — a fallen start combined with fall-is-fatal would end the episode at step 0 — and ~0.3 in stage 2. Target walking speed is resampled uniformly in 0.1–5 m/s when `randomizeWalkSpeedEachEpisode` is on.
- **Discounting:** `gamma: 0.995` and `time_horizon: 1000` in `config.yaml`, with GAE `lambd: 0.95`. The high gamma matters for a task where the payoff for standing up arrives many steps after the effort to do it.

## Training

Training is driven by [`mlagents-learn`](Assets/Python/config.yaml) using PPO:

- 512 hidden units, 3 layers, normalized observations
- 15M max steps, batch size 2048, buffer size 20480
- Checkpoints and TensorBoard event logs are written under `results/` at the repo root (deliberately outside `Assets/`, see below)

Three scenes, all sharing the same `WalkerRagdoll` prefab and therefore the same policy:

| Scene | Agents | Use |
|---|---|---|
| `Walker40` | 40 | Default for training — more agents per environment amortize the Unity↔Python round trip |
| `Walker20` | 20 | Lighter on the machine; the original layout |
| `Solo Walker` | 1 | Watching a single agent's behavior closely |

### Setup

1. Open the repo root in Unity **6000.5.9f1** (or compatible).
2. Set up a Python environment for training (3.9–3.10, since `torch==1.11.0` has no wheels beyond that):
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   pip install -r Assets/Python/requirements.txt
   ```
3. `requirements.txt` installs the **CPU-only** build of torch by default. If you have an NVIDIA GPU, swap in the CUDA build instead:
   ```bash
   pip uninstall torch -y
   pip install torch==1.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
   ```
   Verify it's actually using the GPU: `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"` should print `True` and your GPU's name. Don't expect it pegged, though — the network here is small enough that Unity's physics simulation is the real bottleneck, not the gradient step.
4. Start training. On Windows, use `train.bat` — it always runs from the repo root regardless of your terminal's current directory, so `--results-dir` stays pinned to `results/` instead of landing wherever the shell happened to be (this bit us more than once):
   ```cmd
   train.bat --run-id=<run-name> --torch-device=cuda
   ```
   Add `--resume` to continue an existing run-id instead of starting over:
   ```cmd
   train.bat --run-id=Walker_GetUp --resume --torch-device=cuda
   ```
   On Mac/Linux (or if you'd rather not use the batch file), the equivalent is:
   ```bash
   mlagents-learn Assets/Python/config.yaml --results-dir=results --run-id=<run-name> --torch-device=cuda
   ```
   Drop `--torch-device=cuda` if you're on CPU-only torch.
5. Press Play in the Unity Editor when prompted to connect the environment. Use `Walker40` (or `Walker20`), not `Solo Walker`.
6. Monitor progress with TensorBoard (in a separate terminal, since `mlagents-learn` blocks the one it's running in):
   ```bash
   tensorboard --logdir results
   ```
   Then open the printed URL (typically `http://localhost:6006`).

Training runs live in `results/` at the repo root, **outside `Assets/`** — under `Assets/` the editor writes a `.meta` sibling for every file, including each `events.out.tfevents.*`, which breaks TensorBoard's directory watcher.

The tradeoff is that Unity can't see the models there, so **`sync-models.bat`** copies each run's final `Walker.onnx` into `Assets/Examples/Walker/TFModels/<RunId>.onnx` where it can be dropped onto a Behavior Parameters component. `train.bat` calls it automatically when training finishes; run it by hand if you stopped with Ctrl+C, since answering Y to "Terminate batch job" skips the rest of the script. Re-running overwrites in place, so the `.meta` and its GUID survive and anything already referencing a model picks up the newer weights without re-assigning.

Those copies are gitignored as derived duplicates — `results/` holds the canonical ones. `Assets/Examples/Walker/TFModels/Walker.onnx` is Unity's original stock pretrained model, still wired into the ragdoll's Behavior Parameters by default.

### Running a trained model

Drag the trained `.onnx` model onto the ragdoll's Behavior Parameters component in any Walker scene and set the behavior type to **Inference** to watch it walk without training.

## Training log

What was changed, what was trained on it, and what came out.

### 1. Baseline — stock ML-Agents Walker

The unmodified Unity example. Any torso, head, or hand contact with the ground **ended the episode immediately** (`agentDoneOnGroundContact` on 12 body parts), plus a `SetReward(-1)`. Reward was `matchSpeed × lookAtTarget`.

**`Walker_First_Steps` — ~120k steps.** Walks. Never falls, because falling isn't a state it can occupy. No concept of recovery.

> Removed from the repo in `925199d`. Recoverable via `git checkout 925199d^ -- results/Walker_First_Steps` if a pre-trained walker is ever wanted as a curriculum starting point.

### 2. Fall recovery, first attempt

- Disabled `agentDoneOnGroundContact` **and** `penalizeGroundContact` on all 12 parts — falling no longer ends the episode.
- Added `fallenStartProbability` (0.3): that share of episodes start with the ragdoll already knocked over.
- Added an upright bonus: `+ 0.1 × clamp01(dot(hips.up, worldUp))`.

### 3. Throughput work

- `torch==1.11.0+cpu` → `+cu113`. Training had been running entirely on CPU despite an available GPU — the default PyPI torch wheel is CPU-only.
- Added `engine_settings` to `config.yaml`: `time_scale: 20`, `quality_level: 0`, 84×84 render resolution.

Took a full 15M-step run from impractical to a ~3.5 hour job.

### 4. `Walker_GetUp` — 15,017,000 steps

| Metric | Value |
|---|---|
| Steps | 15,017,000 (`max_steps` reached) |
| Wall clock | ~12,650 s (≈3 h 31 m) |
| Throughput | ~1,180 steps/sec |
| Final mean episode reward | ≈1,472 |
| torch | 1.11.0+cu113 |

**Outcome: it learned to crawl, not walk.** A worming gait along the ground that reached the target perfectly efficiently — and scored well doing it.

### 5. Diagnosis and the posture gate

The crawl was the rational policy, not a training failure:

- `matchSpeedReward` is computed from the **average velocity of all body parts**, and `lookAtTargetReward` from **head yaw flattened to horizontal**. A worming ragdoll satisfies both — worth ~1.0/step.
- Standing upright was worth an additive **0.1**. So the choice was: worm for 1.0, or solve bipedal gait for 1.1.
- `dot(hips.up, worldUp)` doesn't even measure standing — lying flat on your **back** points `hips.up` at the sky and scores ~1.0.

Fix: posture now **gates** locomotion multiplicatively instead of adding to it, and is measured as torso-vertical × head-to-foot height (self-calibrated from the prefab's authored standing pose in `Initialize`):

```csharp
var postureReward = torsoUpright * Mathf.Clamp01(height / m_StandingHeight);
AddReward(matchSpeedReward * lookAtTargetReward * postureReward + 0.1f * postureReward);
```

Crawling now pays ≈0. The standalone `0.1 × postureReward` term remains so a fallen agent still has a gradient to stand — the gated locomotion term is ~0 until it does.

Two flailing fixes went in at the same time: walk speed capped at 5 m/s (it had been sampling up to 10 m/s, roughly a world-record sprint, and velocity matching is the whole objective — so the optimal answer to "sprint at 22 mph" from a ragdoll that can't walk is to hurl limbs), and an **action-rate penalty** on `mean(|aₜ − aₜ₋₁|)`, since nothing previously cost anything for slamming joints to full force or reversing them every decision.

### 6. `Walker_Smooth` — 1,702,000 steps, abandoned

| Checkpoint | Mean reward |
|---|---|
| ~850k | ≈50 |
| 1,702,000 | ≈221 |

At the ~850k mark, reward sat at ≈50 over 5,000-step episodes (~0.01/step) with every agent crawling and none standing. It was still climbing when abandoned — reward more than quadrupled by 1.7M — so this wasn't a hard plateau, just very slow progress toward a behavior that hadn't been discovered yet.

The gate was doing its job: crawling was being denied reward, down from the worm's 0.29/step. But that exposed the *next* problem: **standing up from prone is too hard to discover by random exploration.** With the locomotion term gated to ~0 and only the small posture term live, there was almost no reachable gradient to climb. Denying reward for the wrong behavior doesn't teach the right one if the right one is never stumbled into.

### 7. Two-stage curriculum

Rather than expect one policy to learn balance, gait, *and* recovery simultaneously from a mostly-flat reward, split it:

**Stage 1 — learn to walk, falling is fatal.**
- `agentDoneOnGroundContact: 1` restored on the 12 non-ground parts (torso, head, arms, thighs) (feet and the other 4 ground-legal parts stay `0`).
- `fallenStartProbability: 0` — **required**, since a fallen start plus ground-contact termination would end the episode at step 0.

Termination does the heavy lifting: crawling is not a low-scoring policy here, it's an *impossible* one, so the only way to accumulate reward is to stay up and move. This is the original Unity Walker task, which trains reliably.

**Stage 2 — learn to get up, falling is survivable.**
- Flip `agentDoneOnGroundContact` back to `0` on those 12 parts, set `fallenStartProbability` to ~0.3.
- `--initialize-from` stage 1, so the policy starts already knowing balance and gait and only needs to learn recovery.

The reward function is **identical across both stages** — no code change to switch, just the two prefab fields. That's deliberate: while upright, `postureReward ≈ 1` and the gated reward collapses to the original `matchSpeed × lookAt` plus a constant, so the gate is a no-op in stage 1 and the value function stays meaningful across the transfer. Stage 2 still needs the gate, or the crawl comes straight back.

### 8. `Walker_Stage1` — 15,000,390 steps, complete

| Steps | Mean reward |
|---|---|
| 5,600,000 | 437.6 |
| 13,499,718 | 1,046.3 |
| 13,999,993 | **2,135.8** (peak) |
| 14,499,322 | 1,557.3 |
| 15,000,390 | 1,434.9 (final) |

The curriculum worked. Where the posture-gated flat run (`Walker_Smooth`) was stuck at 221 after 1.7M steps with everything crawling, making falls fatal got the same reward function to 1,435 — because crawling stopped being a *reachable* policy rather than merely an unrewarded one.

Two things worth reading correctly:

- **The late-run swing (1046 → 2136 → 1557 → 1291 → 1435) is sampling noise, not decay.** `randomizeWalkSpeedEachEpisode` draws a target speed in 0.1–5 m/s per episode, and matching 0.5 m/s is far easier than matching 5, so per-episode reward varies widely at fixed policy quality. Note `--initialize-from` loads `checkpoint.pt`, i.e. the **final** checkpoint, not the peak.
- **A mid-run dip after resuming is expected.** Stopping at 5.6M to double the platforms and lower `time_scale` dropped reward from 437 to 101, recovering to 334 over the next 450k steps. Nothing regressed: `--resume` restored the weights (the step counter continued from 5.65M), but every agent restarts its episode simultaneously, so the summary window is briefly dominated by short episodes and the completions stay synchronized for a while. Check `Environment/Episode Length` alongside reward before concluding a restart broke something.

Results also moved from `Assets/Python/results/` to a repo-root `results/` at this point, which ended the TensorBoard `.meta` errors.

### 9. `Walker_Stage2` — in progress

Initialized from stage 1's final checkpoint, with falls survivable and `fallenStartProbability: 0.3`.

| Steps | Mean reward | Episode length |
|---|---|---|
| 50,000 | 1,422.0 | 999.0 |
| 100,000 | 1,225.7 | 999.0 |
| 150,000 | 1,344.1 | 999.0 |
| 200,000 | 1,045.2 | 999.0 |

Reward opens near stage 1's 1,435 — the walking skill transferred intact — but **episode length is pinned at exactly 999**, the cap. (TensorBoard reports episode length in *decisions*; `MaxStep: 5000` physics steps ÷ `DecisionPeriod: 5` = ~1000, so 999 means every episode is running to the limit.)

**Fix applied:** a collapsed-episode timeout (see Termination above). Rather than let a hopeless episode run its full ~999 decisions, it is interrupted after 600 collapsed steps — trading one long look at a single failed pose for several fresh attempts, since every new episode reseeds a different fallen pose. It also restores episode length as a live proxy for recovery success.

That pinned length is the expected consequence of removing termination, and it exposes the next inefficiency. With no early exit, a ragdoll that falls and cannot recover spends the remainder of the episode — up to ~999 decisions, 100 simulated seconds — lying still and earning ~0. If roughly 70% of episodes start upright and score well while the fallen 30% score nothing, a mean near 1,100–1,400 is what you would expect, which matches. It also means episode length currently carries **no** information: it is constant regardless of how well recovery is going.

### 10. `Walker_Stage2c` / `Walker_Stage2d` — curriculum tuning

**`Stage2c` exposed a calibration bug.** The `fallen_tilt` curriculum reached lesson 2 (Prone, 90°) within 250k steps and then flatlined at episode length ~575, reward ~1,150. Mean reward is roughly `W × (0.7 + 0.3p)` for `W` a well-walked episode and `p` the recovery rate — so with 30% fallen starts, a walker that *never* recovers still banks ~0.7W ≈ 1,200. The 1,100 threshold was below that, so inherited walking skill alone advanced the curriculum. `min_lesson_length: 100` compounded it: at 40 agents that many episodes complete inside one summary window. Raised to 1,450/1,500 and 500.

**`Stage2d` fixed the advancing but not the learning.** Lesson correctly held at 0 (30° tilt) for the full 1.5M steps, but episode length sat flat at 490–580 and reward drifted *down* from ~1,230 to ~950. Entropy held at 0.9, so this wasn't policy collapse — it simply wasn't learning.

Episode length turned out to be the informative number. At ~520 it's *below* the 735 you'd expect if only fallen starts were being cut, and solving `0.7 × L + 0.3 × 120 = 520` gives `L ≈ 690` for standing episodes. Since a fall is cut 120 decisions later, **the walker is falling around decision ~400 — roughly 8 simulated seconds.** Stage 1's policy was never a robust walker; stage 2 was compounding a shaky gait with absent recovery.

Hence the potential-based shaping term (see Reward above), which targets both: dense credit for rising posture, negative while posture falls.

### 11. `Walker_Stage2e` — inverted shaping, 5.4M steps wasted

Episode length 432–555 across 5.4M steps, indistinguishable from Stage2d's 490–580, and reward *halved* from ~1,100 to ~500.

The reward drop was the clue. The shaping used the discounted form `k(γ·Φ′ − Φ)` with γ = 0.995 — the trainer's per-*decision* discount — but added it per *physics* step, five times more often. Simulating the episode sum: **−999.8 for an episode that stays standing, −1,168.6 for one that recovers.** The term was inverted; getting up scored worse than staying down. Fixed by using the pure difference (γ = 1).

The lesson generalizes past this bug: **simulate a reward term's episode sum before training on it.** Five lines of Python would have caught this ahead of an hour of GPU time. The runtime version of the same check is watching whether the reward *scale* shifts unexpectedly in the first 100k steps — a new term that changes the magnitude rather than the trend is a red flag.

### 12. `Walker_Stage2f` — 15M steps, the shaping works but the curriculum starves it

First run of four to move at all.

| Phase | Episode length | Reward | Lesson |
|---|---|---|---|
| 0–2M | 526 | 964 | 0 |
| 2–6M | 502 | 870 | 0 |
| 6–10M | 562 | 1,014 | 0 |
| 10–13M | 637 | 1,243 | 0 |
| 13–15M | **700** | **1,471** | 1.46 |

Episode length 502 → 700 and reward 870 → 1,471, sustained rather than noise. Corrected shaping does help — and the mechanism is visible in the split: the −50 for falling taught the walker to *stop falling* (standing episodes now run close to the 999 cap, versus ~690 in Stage2d), while the +50 for recovering has barely been exercised.

Because the curriculum starved it. Lesson 0 → 1 came at **12.90M**, 1 → 2 at **14.10M**, so the agent saw genuinely prone starts for only the final **1M steps**. It hasn't failed at recovery so much as never practiced it.

**Reward is the wrong thing to gate a curriculum on here, in both directions.** At threshold 1,100 (Stage2c) it skipped to the hardest lesson in 250k steps, because a walker that never recovers still banks ~0.7W. At 1,450 it took 13M steps to clear the first lesson, because reward mostly tracks gait quality and improving the gait is what eventually crossed the bar. Reward cannot separate "walks well" from "recovers well", and ML-Agents only offers `reward` or `progress` as measures — so the curriculum now gates on **progress**, giving each lesson a guaranteed budget: 4.5M / 6M / 19.5M against a raised 30M `max_steps`.

### 13. `Walker_GetupOnly` — 20M steps, recovery finally happens

Every episode starts near-prone (80–90°), so 100% of experience is get-up practice against ~15% under the graduated curriculum. With `collapsedStepLimit` at 600 physics steps, an agent that never rises is cut at **120 decisions** — that's the floor, and it makes episode length an unambiguous readout.

| Phase | Episode length | Reward | Entropy |
|---|---|---|---|
| 50k–1.2M | 258 | 17 | 0.85 |
| 1.25–2.4M | 561 | 77 | 0.82 |
| 2.45–3.6M | 662 | 135 | 0.80 |
| 3.65–4.8M | 752 | 188 | 0.78 |
| 4.85–6M | **838** | **249** | 0.77 |

Episode length climbed to 838 against a floor of 120, max 978. After five runs that produced nothing, **recovery is learnable from this reward** — the blocker was never the reward shape or the mechanics, it was that get-up practice was diluted to a fraction of each run.

**The lesson: when a behavior won't emerge, check how much of the data actually contains it** before rewriting the reward. Across Stage2b–2f the agent was spending 85–99% of its experience on walking while the target behavior was recovery. Isolating the skill did in ~90 minutes what five multi-hour runs of reward and curriculum tuning could not.

Caveat on where it got to: it reaches a kneel, not a stand — clear of the 0.2 collapse threshold but well short of upright.

Reward was still climbing at the 6M budget, so `max_steps` was raised and the run continued to 20M:

| Steps | Mean reward |
|---|---|
| 6,000,000 | 249 |
| 18,499,337 | 797.4 |
| 18,999,177 | 716.8 |
| 19,499,787 | 845.8 |
| 19,999,548 | 918.0 |
| 20,000,548 | **954.1** (final) |

Still climbing at the wall. But more steps only bought a *better kneel* — 249 → 954 is the agent perfecting a pose that was never the goal, which is what §14 is about.

### 14. Measuring the kneel — the collapse threshold was the target all along

Rather than guess, `Walker/Posture` was logged to TensorBoard as a histogram. Over the 14M steps to the 20M wall it **settles around 0.52 and is still creeping up.** Standing is 1.0 by construction (`m_StandingHeight` is calibrated from the ragdoll's own pose at `Initialize`), so 0.52 is torso vertical, folded down onto the shins — the kneel, exactly as it looks in playback.

That is worth stating plainly: my earlier estimate, back-solved from mean reward through an assumed rigid-tilt `cos²` model, was ≈0.3. It was wrong by most of the gap. A kneel holds the torso vertical, so `dot(hips.up, up)` stays near 1 and nearly all the loss comes from the height ratio — a shape the tilt model doesn't describe. **Back-solving a quantity out of an aggregate reward is not measurement.** Logging the quantity cost one line and settled it.

One caveat on 0.52 that shaped the fix: it is an episode *mean* over every physics step, and every episode opens with two or three seconds prone at posture ≈0. So 0.52 is a lower bound on the pose actually held once risen — if the transient is a quarter of the episode, the held pose is nearer 0.66. The histogram cannot separate the transient from the plateau, and the two readings imply different thresholds.

Two changes follow:

**The threshold is now curriculum-ramped** — 0.2 → 0.35 → 0.5 → 0.6 → 0.7 — via the `collapse_posture` environment parameter. Any fixed bar becomes a target, because the agent settles at the cheapest pose that clears it; 0.2 means "off the floor", and the cheapest pose clearing "off the floor" is a kneel that survives indefinitely.

A ratchet is also how the ambiguity above gets resolved without resolving it first: it walks the bar up through both candidate ceilings and bites wherever the real one is. **The lesson where episode length drops is the measurement** — that step is the true held-posture ceiling, and it's free.

`measure: progress` is `current_step / max_steps`, so these thresholds only mean what they say on a run that starts at step 0 — which is one reason the next run uses `--initialize-from` rather than `--resume`.

**The collapse cut is now `EndEpisode()` rather than `EpisodeInterrupted()`** — and without this, the ramp above would have been close to inert. `EpisodeInterrupted` resolves to `DoneReason.MaxStepReached`, which *bootstraps* `V(s_T)`. Bootstrapping is right for a cutoff that's an artifact of the harness rather than the task, and it means the agent doesn't perceive the cutoff as costly: the bootstrapped target converges to the infinite-horizon value of kneeling regardless of where the bar sits. Raising the bar would only have shortened episodes — a real effect on the *data distribution*, since shorter episodes mean more prone resets per hour, but no effect on the *objective*. `EndEpisode` marks a terminal state worth zero future reward, which makes failing to rise genuinely cost something.

This is not the lesson stage 2 exists to unlearn. That was "touching the ground is death", which fired on contact and denied any chance to recover; it stays off (`agentDoneOnGroundContact` is 0 on all 16 body parts). This fires only after 600 physics steps of *failing to get back up*, and failing to get up is failure.

**The general lesson: a termination condition is a specification of success, and the agent reads it far more literally than the reward.** `collapsedPostureThreshold` was written as a housekeeping parameter — cut dead episodes, save sim time — and it quietly became the definition of "good enough". Any survival cutoff does this. If a state lets the episode continue, expect the agent to find it and stay there.

**Neither change was in effect for the 20M run.** Training was launched seven minutes before the ramp and the `EndEpisode` swap were written, so the whole 6M→20M stretch ran at a flat 0.2 with the bootstrapping cut. That is legible after the fact because `mlagents-learn` writes the config it actually used to `results/<run-id>/configuration.yaml`, and that file has no `collapse_posture` key at all.

Which makes it a clean measurement of the un-ramped policy rather than a wasted run — but the lesson holds regardless: **a curriculum that silently isn't there looks exactly like a curriculum that isn't working**, and 14M steps is a long time to spend on that distinction. Verify, don't assume.

`configuration.yaml` is *not* how to verify it live, though — `mlagents-learn` writes it at **exit**, within milliseconds of the final model export, not at launch. It's a reliable record afterwards and useless during. Check TensorBoard instead: every environment parameter gets an `Environment/Lesson Number/<name>` scalar, and those appear within the first summary. If the tag isn't there, the parameter isn't being sent.

That only proves Python sent it. To prove the *C# read it*, you need a quantity that must move if it took effect — `posture_focus` was confirmed by mean reward jumping 2.65×, which cannot happen unless `WalkerAgent` recompiled and picked up the new term. A silently-ignored parameter is the failure mode worth designing a check against, because Unity will happily run stale compiled code.

### 15. Next: `Walker_Getup2`, seeded from the kneeler

`--initialize-from=Walker_GetupOnly` rather than `--resume`, for three reasons that happen to coincide:

- **`--resume` cannot work.** The run ended at 20,000,548 against `max_steps: 20000000`; a run at its wall exits immediately. (Second time — see the note on `max_steps` in `config_getup.yaml`.)
- **The LR schedule is spent.** `learning_rate_schedule: linear` has decayed to ~0 at the wall, so even a resumed run would barely move.
- **The curriculum needs a clean axis.** `measure: progress` counts from step 0. Resuming at 20M would drop straight into the last lesson; `--initialize-from` resets the counter, so the ramp runs as written.

The weights carry over, which is the part worth keeping — rising from prone to a stable kneel is most of a get-up. Lesson 0 holds the bar at 0.2 for the first 2M steps so the seeded policy re-stabilises under the terminal cut before the bar starts moving; otherwise a regression can't be attributed between the two changes.

### 16. The target was paying for the crawl

Watching `Walker_Getup2`, the agent knee-walks toward the target on its hands rather than trying to rise. Decomposing the final mean reward of the run that produced that policy — 954 over ~4,190 physics steps per episode (838 decisions × decision period 5) — says why:

| Term | Per physics step | Share |
|---|---|---|
| Locomotion (`matchSpeed × lookAt × posture`) | 0.170 | **74%** |
| Posture level (`0.1 × posture`) | 0.052 | 23% |
| Posture shaping (`50 × Δposture`) | 0.006 | 3% |

Inverting `matchSpeedReward` puts the implied crawl at **~1.8 m/s** against a 5 m/s target. Three-quarters of this policy's income comes from chasing the target, and it collects that income without ever standing.

**Gating locomotion by posture is not the same as not paying for locomotion.** The gate (§ on the worm gait) was working exactly as designed — a kneeler at posture 0.52 keeps 52% of the locomotion term — and 52% of a large term still beats 100% of a small one.

To be clear about what this is and isn't: standing and walking scores ~0.99/step against knee-crawling's ~0.23, so the reward is **not** inverted, and no amount of reward-ratio arithmetic explains the behavior on its own. What the target does is make the *valley* expensive. Rising means rocking back onto the feet — forward velocity goes to zero, `matchSpeedReward` collapses to ~0, and 74% of the income stops for the duration of the attempt. If the attempt fails, the now-terminal collapse cut zeroes the rest of the episode. The cheapest way to stay paid is to keep crawling.

The fix is the same move that made `Walker_GetupOnly` work, applied one level in. That run isolated the **start state** — 100% prone — and left the **objective** mixed. `posture_focus` (env parameter, 0 → 1) fades out the locomotion term and raises the posture coefficient 0.1 → 1.0, so at 1.0 posture is the entire reward and there is nothing to earn by moving at all. It stays flat at 1.0 for the get-up run; the target returns when this policy is folded back into the walking curriculum.

**The target GameObject stays in the scene** — this is a reward-side switch, and that's the right layer for it. `CollectObservations` feeds the target's position through the orientation cube, and other observations are expressed relative to that cube, so deleting the object null-refs `UpdateOrientationObjects` and changes the observation size. A different observation size means the seeded network no longer loads, which breaks `--initialize-from` against every model trained so far. Disable an objective in the reward, not in the scene.

(While checking that: `TouchedTarget()` — the +1 touch bonus — has no caller anywhere in `Assets/`. `TargetController` fires collision UnityEvents but nothing wires one to it, so that bonus has never been paid in this project. The decomposition above is what confirms it independently: three terms summing to 0.228/step against a measured 0.2277 leaves no room for a fourth.)

Reward magnitude is sized so a full stand pays about what standing-and-walking used to. That's cosmetic — PPO normalises advantages — and it does **not** make the curve comparable to earlier runs at intermediate poses: a kneel now earns ~0.52/step against ~0.23 before, so reward jumps ~2× at unchanged behavior. **Read `Walker/Posture`, not reward, while this is on.**

**The general lesson: check what fraction of the reward the unwanted behavior is actually collecting, before assuming the reward ranks it correctly.** Ranking is not the same as incentive. A term can rank the target behavior first and still fund the wrong one, because what the agent follows is the local gradient and what it defends is its current income.

### 17. `Walker_Getup3` — the target wasn't the whole answer

With `posture_focus: 1.0` there is nothing left to earn by moving, and the agent kept knee-crawling anyway:

| Step | Posture | Reward | Reward / physics step | Entropy |
|---|---|---|---|---|
| 50k | 0.515 | 2,527 | 0.513 | 0.671 |
| 150k | 0.520 | 2,605 | 0.521 | 0.671 |
| 300k | 0.518 | 2,625 | 0.526 | 0.672 |

A useful side effect first: with posture as the entire reward, **reward ÷ physics steps *is* mean posture** (0.513 vs 0.515, and so on down the table). The headline metric became a direct readout of the thing we care about, which is worth engineering for on purpose.

Three flat columns is the finding. Posture varies by ±0.006 and entropy by ±0.001 across 300k steps — that is not slow learning, it is a converged policy sitting still. Removing the target removed the *reason* to crawl, and the crawling continued, so the target was never the whole story.

The diagnosis: standing already pays ~2× kneeling per step, so the incentive is not missing. What's missing is that **this policy has never been upright.** Nothing in 20M steps of experience tells it what upright feels like or how to hold it, and the rise is a multi-second coordinated maneuver that undirected action noise will never stumble into. Raising `beta` would inject more noise into 39 joint targets; more noise does not assemble a coordinated motion.

So the fix is **structured exploration instead: start the episode in states the policy cannot reach on its own.** `fallen_tilt` samples uniformly over `[0, 90]` every episode — prone, half-down and standing all in the same buffer, every buffer.

**How that helps is not the obvious story, and the difference matters.** The intuitive version is that the standing agents score higher and the crawlers see it and copy. They can't: all 40 walkers share one network and pool experience into one buffer, and nothing in PPO lets one agent observe another's return. There is no imitation and no comparison.

The route is the **critic**. Standing episodes teach `V()` that high-posture states are worth a lot. Once `V(upright) >> V(kneel)`, an action taken from a kneel that raises posture earns positive *advantage* — it moves toward a state the critic now values — even though its immediate reward barely changed. **The value function is what carries "there's something better over here" back to the states the crawler is actually in.** That's also why level reward alone wasn't enough: advantage is measured against the critic's prediction for that state, so a stander collecting a high reward the critic already expects produces no gradient at all.

Which is why this is a **mix rather than a reverse curriculum** ramping 30 → 90, and worth being concrete about:

1. **The critic needs both ends present at once.** A lesson containing only easy starts teaches `V(upright)` but has no kneel states to apply it to. The useful contrast is within-buffer.
2. **Rising from prone took 20M steps and is the one asset here.** A lesson 0 with no prone starts spends millions of steps not practising it — catastrophic forgetting on exactly the skill worth keeping.
3. **Thresholds are a moving part, and this project's have been miscalibrated three times** (`measure: reward` gaming, a cliff instead of a ramp, thresholds written against the wrong step). Uniform sampling has nothing to calibrate.

Uniform in *tilt* is not uniform in *difficulty* — posture falls off as ~cos²(tilt), so `[0,30]` is the easy third and `[60,90]` the hard third. That spread is the point.

This is also the curriculum `GetupOnly` deleted, and it failed the first time because walking diluted it to ~15% get-up practice. `posture_focus` is exactly what removes that dilution, so **the reason it failed is gone.** The difference now is that mixing difficulty is not the same as mixing *objectives*: prone and standing starts are the same task at different distances from the goal, whereas walking and recovering were two tasks competing for the same buffer.

One honest note on the rescale in §16: raising the posture coefficient 0.1 → 1.0 also cut the shaping term's *relative* weight from ~11% of posture reward to ~1%, diluting the one term that pays for the transition itself. Left alone for now — the level term at 10× does most of what shaping was compensating for, and this run already has enough moving parts — but it's a knob to revisit before adding new ones.

## Notes on reward design and retraining

Generalizable lessons from the above, mostly learned the expensive way.

**Termination conditions are part of the reward function.** The baseline reward never said "be upright" — it said "move fast toward the target while facing it." It got away with that because torso-on-ground states were *unreachable*: touching down ended the episode. Crawling wasn't a low-scoring policy, it was an impossible one. Relaxing a termination condition widens the policy space into a region where the reward was never actually specified, and the agent will find whatever lives there. Budget a reward term for every termination you remove.

**The real penalty was opportunity cost, not the `-1`.** `SetReward(-1)` *overwrites* the step's reward rather than adding, so its worst case was swapping one `+1.0` step for `-1.0` — a swing of 2. Meanwhile terminating at step 100 of 5,000 forfeits ~4,900 points of remaining episode. Termination outweighed the explicit penalty by three orders of magnitude.

**Gate, don't bonus.** If a property is a *prerequisite* for the behavior you want, multiply by it. If it's genuinely optional, add it. An additive bonus only shifts preference at the margin — it loses outright to any easier policy that skips the property and banks the main reward anyway.

**Measure what you actually mean.** `dot(hips.up, worldUp)` is a plausible-looking "uprightness" term that quietly awards full marks for lying on your back. Cheap sanity check: enumerate the degenerate poses and ask what each one scores.

**Removing a bad optimum doesn't create a good one.** Gating crawling to ~0 reward correctly stopped it from paying — and the agent kept crawling anyway, because standing up was never discovered in the first place. A reward can only select among behaviors exploration actually reaches. When the target behavior is a long, precise action sequence with no partial credit along the way, that's a *credit-assignment* problem, and the fix is curriculum or demonstrations, not more reward shaping.

**Deciding whether to reuse a checkpoint.** Two separate questions:

1. *Can it load?* Observation space size, action space size, network architecture (`hidden_units`, `num_layers`, memory), and normalization state must all match. Change any and you must start fresh. Reward-only changes never break this — which is why every change logged above stayed loadable.
2. *Should it load?* A checkpoint encodes a **converged behavior**, and PPO improves locally. Reusing one means searching outward from that behavior's basin.

Two failure modes for (2). **Entropy collapse**: a converged policy has a sharp, near-deterministic action distribution and barely explores — normally desirable, but escaping a bad optimum is exactly when you need exploration. A confidently wrong policy is worse than no policy. **Value staleness**: `--initialize-from` restores the critic too, and it was trained on the *old* reward, so early advantage estimates are wrong and the first updates are partly destructive.

**`--resume` vs `--initialize-from`** — not interchangeable:

| | `--resume` | `--initialize-from` |
|---|---|---|
| Means | same task, continue | different task, borrow the skill |
| Step counter | continues | resets to 0 |
| LR schedule | resumes where it left off | restarts |
| TensorBoard | appends to same curve | new run |

The trap with `learning_rate_schedule: linear`: it decays the LR to **zero** across `max_steps`. `--resume` on a run that already hit `max_steps` trains at a learning rate of approximately nothing — it looks alive, burns hours, and barely updates. Extending a finished run means raising `max_steps` to give the schedule room.

Rule of thumb: **`--initialize-from` when adding a skill, start fresh when removing one.** And since experiments here are ~1,180 steps/sec, just run both for ~500k steps and compare curves. If the initialized run starts higher then flatlines while the fresh run overtakes it, that's entropy collapse. `beta` (entropy regularization, currently `0.005`) is the knob — raising it keeps the policy stochastic and is the standard medicine for fine-tuning out of a local optimum.

## Todo

- **Automate the stage 1 → 2 switch.** `fallen_tilt` now runs through `EnvironmentParameters`, but `agentDoneOnGroundContact` and `fallenStartProbability` are still manual prefab edits between stages. Routing those through the same channel would make the whole curriculum one `mlagents-learn` job.
- **Headless parallel training** — build the project to a standalone player (File → Build) and run `mlagents-learn` against it with `--no-graphics --num-envs=N`. This is the main remaining throughput win.

  Observed symptom on an i9-9900KF (8C/16T): the Editor lags while every logical processor runs moderately busy and spiky, none saturated, and the GPU idle. That rules out a raw compute limit.

  `Assets/ML-Agents/Timers/Walker_timers.json` (written after each Play session) pins it down — from a 1,625 s session:

  | Timer | Total | Share of wall clock |
  |---|---|---|
  | `DecideAction` (blocking gRPC round trip to Python) | 935.07 s | **57.5%** |
  | `AgentSendState` (packaging observations) | 74.90 s | 4.6% |
  | `AgentAct` (applying actions) | 38.88 s | 2.4% |

  Over half the run is Unity **blocked waiting on the trainer** — ~2.1 ms of dead time per decision across 450,013 calls — while the work Unity actually performs for ML-Agents is ~7%. That's why all cores look half-busy: they burst through a physics step, then wait together.

  Neither `time_scale` nor more platforms per scene fixes that, since both add work to an environment that is already spending its time waiting. `--num-envs` does: each environment is a separate OS process, so while one blocks on Python the others keep simulating. Note the two multiply — `--num-envs=4` against this 20-platform scene is 80 agents across 4 processes.

  Sizing on 8 physical cores: budget one for the Python trainer and one for the OS, start at `--num-envs=4`, and scale while watching per-core utilization rather than the aggregate. With ~4× the experience per second, consider raising `buffer_size` (currently 20480) so the trainer isn't updating on a buffer that refills almost instantly.

- **Pack agents per env before adding envs.** The two scale differently, and the shape that wins is **few processes, many agents each**.

  The ~2.08 ms round-trip cost is paid *per exchange*, not per agent — 20 platforms already share one. So agents-per-env amortizes latency, while envs-per-machine buys overlap across cores. Projecting from the measured 3.61 ms/step: going 20 → 40 platforms takes a step to ~5.1 ms but doubles decisions per step (4 → 8), netting roughly **+40% throughput** on a single env, for free and with no build required (just duplicate Platform instances in the scene). The ceiling is wherever physics per step grows enough to dominate — find it empirically by watching steps/sec.

  The degenerate opposite, 80 envs × 1 platform, is worth understanding: it would overlap round trips beautifully, but each Unity player carries 300 MB–1 GB of fixed cost plus its own PhysX scene, ML-Agents spawns a Python subprocess per env (~160 processes), and 80 main threads would fight over 8 cores. Each env would also sit ~90% idle, paying full latency to serve one ragdoll. The concurrency model isn't the problem — per-process overhead is.

- **`time_scale` is capped by the round trip, not by ambition.** It was set to 20 while the machine actually delivered ~5.5×; Unity spent the gap perpetually catching up, which is most of the Editor lag. It's now 8. Re-derive the achieved value after any throughput change: `steps ÷ seconds × fixedDeltaTime` from the root and `DecideAction` entries in the timer file. Separately, the periodic hitch every ~18 s is PPO consuming its buffer (`buffer_size: 20480` at ~1,100 decisions/sec) and running `num_epoch: 3` × 10 minibatches while Unity blocks — that one is inherent to on-policy learning, not a misconfiguration.
