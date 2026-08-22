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
- **Recover from falls** — teach the agent to get back up after it topples (in progress — not yet implemented).

## Project layout

The actual Unity project lives in [`MLAgents-Walker/`](MLAgents-Walker):

```
MLAgents-Walker/
├── Assets/
│   ├── Examples/Walker/          # Walker scenes, ragdoll prefabs, WalkerAgent.cs
│   ├── Examples/SharedAssets/    # Shared ML-Agents example scripts (joints, targets, sensors, etc.)
│   ├── ML-Agents/                # ML-Agents package example content + training timers
│   └── Python/                   # Training config, Python requirements, training results
├── Packages/                     # Unity package manifest (ML-Agents, URP, Input System, etc.)
└── ProjectSettings/
```

Everything else at the repo root is stale leftovers from an earlier prototype and isn't part of this project.

## How the agent works

`WalkerAgent.cs` ([source](MLAgents-Walker/Assets/Examples/Walker/Scripts/WalkerAgent.cs)) controls a 16-body-part ragdoll (hips, chest, spine, head, arms, legs) via joint target rotations and per-joint strength, output as continuous actions from a PPO policy. Each step it observes:

- Per-body-part ground contact, velocity, angular velocity, position relative to the hips, and joint strength.
- Its velocity relative to a goal walking speed and direction (via a stabilized "orientation cube" reference frame).
- The target's position relative to that same frame.

Reward is shaped from how closely the ragdoll matches the target walking speed and how well its head/body face the direction of travel, plus a bonus for touching the target.

## Training

Training is driven by [`mlagents-learn`](MLAgents-Walker/Assets/Python/config.yaml) using PPO:

- 512 hidden units, 3 layers, normalized observations
- 15M max steps, batch size 2048, buffer size 20480
- Checkpoints and TensorBoard event logs are written under `Assets/Python/results/`

### Setup

1. Open `MLAgents-Walker/` in Unity **6000.3.4f1** (or compatible).
2. Set up a Python environment for training:
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   pip install -r MLAgents-Walker/Assets/Python/requirements.txt
   ```
3. From that environment, start training:
   ```bash
   mlagents-learn MLAgents-Walker/Assets/Python/config.yaml --run-id=<run-name>
   ```
4. Press Play in the Unity Editor when prompted to connect the environment.
5. Monitor progress with TensorBoard:
   ```bash
   tensorboard --logdir MLAgents-Walker/Assets/Python/results
   ```

A prior run (`Walker_First_Steps`) and its trained `.onnx` model are checked into `Assets/Python/results/` and `Assets/Examples/Walker/TFModels/` for reference/inference.

### Running a trained model

Drag the trained `.onnx` model onto the ragdoll's Behavior Parameters component in the `Walker` or `Solo Walker` scene and set the behavior type to **Inference** to watch it walk without training.
