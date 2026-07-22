# Unity-Walker — Humanoid Locomotion with ML-Agents

A Unity + ML-Agents reinforcement learning project that teaches a **jointed humanoid ragdoll** to stand up, balance, and walk toward a moving target — starting from a heap on the floor.

Built on Unity's ML-Agents `Walker` example architecture with custom training configs (`config.yaml`, `config_standup.yaml`) and trained checkpoints included.

## What it does

The agent is a physically simulated humanoid with 16 driven joints (hips, chest, spine, head, arms, legs). Every physics step it:

- Observes joint positions/velocities, ground contacts, target direction, and its own linear/angular velocity.
- Outputs continuous target rotations + strength for each joint via `JointDriveController`.
- Is rewarded for velocity alignment to the target direction, upright torso orientation, and target-facing gaze; punished for falling.

Two curricula ship in the repo:

- **`config_standup.yaml`** — earlier curriculum focused on standing up from a collapsed pose (the hard part).
- **`config.yaml`** — full walk-to-target curriculum with randomised walking speeds each episode.

## Highlights

- **Real physics locomotion**, not animation blending — the agent has no baked motion; every step is emergent.
- **Curriculum design** — separate standup and walk configs, since a joint policy trained end-to-end tends to converge to a "safe crouch" local minimum.
- **Trained model included** — inference-ready `.onnx` in `results/Walker_First_Steps/` so the walk is playable without retraining.
- Reuses ML-Agents `SharedAssets` (`JointDriveController`, `BodyPart`, `SensorBase`) as the physics substrate — the interesting file is `Assets/Examples/Walker/Scripts/WalkerAgent.cs`.

## Tech stack

- **Engine:** Unity (2022.3 / Unity 6)
- **RL:** Unity ML-Agents Toolkit (PPO)
- **Language:** C# (agent logic), Python (training via `mlagents-learn`)

## Repo layout

```
MLAgents-Walker/
├── Assets/
│   ├── Examples/Walker/Scripts/WalkerAgent.cs       # Reward + observation logic
│   ├── Examples/SharedAssets/Scripts/*.cs           # Joint driver, sensors, monitor
│   └── Python/
│       ├── config.yaml            # PPO hyperparams for the full walk task
│       ├── config_standup.yaml    # PPO hyperparams for the standup pretext task
│       └── results/               # Training logs + exported .onnx models
├── Packages/           # Unity package manifest (ml-agents, physics)
└── ProjectSettings/
```

## Run

### Play the trained agent

1. Open the project in Unity.
2. Open the Walker scene.
3. Drag `results/Walker_First_Steps/*.onnx` into the agent's **Behavior Parameters → Model** slot.
4. Press Play.

### Train from scratch

```bash
# In a Python 3.9 venv with mlagents installed:
cd MLAgents-Walker/Assets/Python
mlagents-learn config.yaml --run-id=Walker_Retrain
# then press Play in the Unity Editor when prompted
```

Monitor with `tensorboard --logdir results`.

## Status

Personal RL project — a step up in complexity from RollerBall-style toy environments, working through the continuous-control side of ML-Agents.
