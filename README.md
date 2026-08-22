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
└── Python/                   # Training config, Python requirements, training results
Packages/                     # Unity package manifest (ML-Agents, URP, Input System, etc.)
ProjectSettings/
```

## How the agent works

`WalkerAgent.cs` ([source](Assets/Examples/Walker/Scripts/WalkerAgent.cs)) controls a 16-body-part ragdoll (hips, chest, spine, head, arms, legs) via joint target rotations and per-joint strength, output as continuous actions from a PPO policy. Each step it observes:

- Per-body-part ground contact, velocity, angular velocity, position relative to the hips, and joint strength.
- Its velocity relative to a goal walking speed and direction (via a stabilized "orientation cube" reference frame).
- The target's position relative to that same frame.

Reward is shaped from how closely the ragdoll matches the target walking speed and how well its head/body face the direction of travel — both **gated by posture** (torso vertical × standing height), plus a bonus for touching the target. The gating matters: speed is measured from average body velocity and facing from head yaw, both of which a crawling ragdoll satisfies just fine. An earlier version added posture as a small bonus instead of gating on it, and the agent converged on worming along the ground rather than walking.

Falling no longer ends the episode: ~30% of episodes start with the ragdoll already knocked over so it gets dedicated practice standing back up (see `fallenStartProbability` on `WalkerAgent`). The standalone posture term is what gives a fallen agent a gradient to stand, since the gated locomotion reward stays near zero until it does.

## Training

Training is driven by [`mlagents-learn`](Assets/Python/config.yaml) using PPO:

- 512 hidden units, 3 layers, normalized observations
- 15M max steps, batch size 2048, buffer size 20480
- Checkpoints and TensorBoard event logs are written under `Assets/Python/results/`

Use the `Walker` scene (20 parallel agents) for actual training; `Solo Walker` (1 agent) is for closer inspection of a single agent's behavior.

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
4. Start training. On Windows, use `train.bat` — it always runs from the repo root regardless of your terminal's current directory, so `--results-dir` stays pinned to `Assets/Python/results` instead of landing wherever the shell happened to be (this bit us more than once):
   ```cmd
   train.bat --run-id=<run-name> --torch-device=cuda
   ```
   Add `--resume` to continue an existing run-id instead of starting over:
   ```cmd
   train.bat --run-id=Walker_GetUp --resume --torch-device=cuda
   ```
   On Mac/Linux (or if you'd rather not use the batch file), the equivalent is:
   ```bash
   mlagents-learn Assets/Python/config.yaml --results-dir=Assets/Python/results --run-id=<run-name> --torch-device=cuda
   ```
   Drop `--torch-device=cuda` if you're on CPU-only torch.
5. Press Play in the Unity Editor when prompted to connect the environment. Use the `Walker` scene, not `Solo Walker`.
6. Monitor progress with TensorBoard (in a separate terminal, since `mlagents-learn` blocks the one it's running in):
   ```bash
   tensorboard --logdir Assets/Python/results
   ```
   Then open the printed URL (typically `http://localhost:6006`).

A completed run (`Walker_GetUp`, 15M steps, fall-recovery reward) and its trained `.onnx` model are checked into `Assets/Python/results/` for reference/inference. `Assets/Examples/Walker/TFModels/Walker.onnx` is Unity's original stock pretrained model, still wired into the ragdoll's Behavior Parameters by default.

### Running a trained model

Drag the trained `.onnx` model onto the ragdoll's Behavior Parameters component in the `Walker` or `Solo Walker` scene and set the behavior type to **Inference** to watch it walk without training.

## Training log

What was changed, what was trained on it, and what came out.

### 1. Baseline — stock ML-Agents Walker

The unmodified Unity example. Any torso, head, or hand contact with the ground **ended the episode immediately** (`agentDoneOnGroundContact` on 12 body parts), plus a `SetReward(-1)`. Reward was `matchSpeed × lookAtTarget`.

**`Walker_First_Steps` — ~120k steps.** Walks. Never falls, because falling isn't a state it can occupy. No concept of recovery.

> Removed from the repo in `925199d`. Recoverable via `git checkout 925199d^ -- Assets/Python/results/Walker_First_Steps` if a pre-trained walker is ever wanted as a curriculum starting point.

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

## Notes on reward design and retraining

Generalizable lessons from the above, mostly learned the expensive way.

**Termination conditions are part of the reward function.** The baseline reward never said "be upright" — it said "move fast toward the target while facing it." It got away with that because torso-on-ground states were *unreachable*: touching down ended the episode. Crawling wasn't a low-scoring policy, it was an impossible one. Relaxing a termination condition widens the policy space into a region where the reward was never actually specified, and the agent will find whatever lives there. Budget a reward term for every termination you remove.

**The real penalty was opportunity cost, not the `-1`.** `SetReward(-1)` *overwrites* the step's reward rather than adding, so its worst case was swapping one `+1.0` step for `-1.0` — a swing of 2. Meanwhile terminating at step 100 of 5,000 forfeits ~4,900 points of remaining episode. Termination outweighed the explicit penalty by three orders of magnitude.

**Gate, don't bonus.** If a property is a *prerequisite* for the behavior you want, multiply by it. If it's genuinely optional, add it. An additive bonus only shifts preference at the margin — it loses outright to any easier policy that skips the property and banks the main reward anyway.

**Measure what you actually mean.** `dot(hips.up, worldUp)` is a plausible-looking "uprightness" term that quietly awards full marks for lying on your back. Cheap sanity check: enumerate the degenerate poses and ask what each one scores.

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

- **Retrain against the posture gate.** Start fresh rather than `--resume` — the 15M-step policy is confidently converged on worming, which is the case where a prior actively hurts.
- **Two-stage curriculum.** Stage 1: train a walker with `agentDoneOnGroundContact` on and `fallenStartProbability = 0` (the known-good original task, trains fast). Stage 2: flip both, `--initialize-from` stage 1. Stage 1 fixes the cold start; the gate fixes the local optimum — they're complementary, and stage 2 still needs the gate or the crawl returns. Conveniently the gated reward is backward-compatible with stage 1: while upright, `postureReward ≈ 1` and it collapses to the original reward plus a constant. `m_ResetParams` (an assigned-but-unused `EnvironmentParameters` hook in `WalkerAgent`) is what ML-Agents' native curriculum system drives, if this should run as one automated job instead of two manual runs.
- **Headless parallel training** — build the project to a standalone player (File → Build) and run `mlagents-learn` against it with `--no-graphics --num-envs=N`. Editor Play mode only ever runs one instance of the scene; a headless build lets ml-agents spawn several in parallel. Note the CPU is already saturated at `time_scale: 20`, so the win here is removing Editor overhead more than true parallelism until there's CPU headroom.
