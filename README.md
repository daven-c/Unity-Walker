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

Reward is shaped from how closely the ragdoll matches the target walking speed, how well its head/body face the direction of travel, and how upright it stays — plus a bonus for touching the target. Falling no longer ends the episode: ~30% of episodes start with the ragdoll already knocked over so it gets dedicated practice standing back up (see `fallenStartProbability` on `WalkerAgent`).

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

## Todo

- **Headless parallel training** — build the project to a standalone player (File → Build) and run `mlagents-learn` against it with `--no-graphics --num-envs=N`. Editor Play mode only ever runs one instance of the scene; a headless build lets ml-agents spawn several in parallel, which should beat anything the `engine_settings` speedup in `config.yaml` can do on its own.
