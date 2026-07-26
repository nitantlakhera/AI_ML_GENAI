# MiniGPT (TensorFlow) — Diagram Images (PNG)

Rendered from the Mermaid diagrams in this folder's docs via `../render_diagrams.py`.

Regenerate after editing any diagram:
```bash
python render_diagrams.py
```

## Index

### From `NEURAL_NETWORK.md`
| File | Shows |
|------|-------|
| `NEURAL_NETWORK__01__flowchart-td.png` | Full MiniGPT model (Keras layers) |
| `NEURAL_NETWORK__02__flowchart-td.png` | Inside one transformer block |

### From `FLOW.md`
| File | Shows |
|------|-------|
| `FLOW__01__flowchart-lr.png` | Big picture: train vs use |
| `FLOW__02__flowchart-td.png` | File dependency graph |
| `FLOW__03__flowchart-td.png` | Training flow (`model.fit()`) |
| `FLOW__04__sequencediagram.png` | What `fit()` does internally (GradientTape) |
| `FLOW__05__flowchart-td.png` | Forward pass |
| `FLOW__06__flowchart-td.png` | Generation flow |
| `FLOW__07__sequencediagram.png` | Chat call flow |
