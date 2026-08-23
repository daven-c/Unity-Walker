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

  Stage 2 adds a second exit: if `postureReward` stays below `collapsedPostureThreshold` (0.2) for `collapsedStepLimit` consecutive physics steps (600, ~12 simulated seconds), the episode is cut short via **`EpisodeInterrupted()`**. The counter resets the instant posture recovers, so an agent making progress keeps its time. `EpisodeInterrupted` resolves to `DoneReason.MaxStepReached` and bootstraps the value estimate — `EndEpisode()` would mark a terminal state worth zero future reward and re-teach "the ground is death," which is the lesson stage 2 exists to unlearn.

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

A completed run (`Walker_GetUp`, 15M steps, fall-recovery reward) and its trained `.onnx` model are checked into `results/` for reference/inference. `Assets/Examples/Walker/TFModels/Walker.onnx` is Unity's original stock pretrained model, still wired into the ragdoll's Behavior Parameters by default.

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

- **Automate the curriculum.** `m_ResetParams` in `WalkerAgent` is an assigned-but-unused `EnvironmentParameters` hook, which is exactly what ML-Agents' native curriculum system drives. Wiring `fallenStartProbability` (and a termination toggle) through it would make both stages one `mlagents-learn` job with lessons in `config.yaml`, instead of two manual runs with an Inspector edit between them.
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
