# Adaptive Inference Router with Cascade Serving

A research-grade adaptive inference routing system that learns to dynamically dispatch incoming requests across a cascade of model variants (quantized, pruned, distilled, full-precision) based on predicted query difficulty, SLA constraints, and real-time cluster load using multi-objective reinforcement learning.

## Quick Start

### Installation

```bash
pip install -r requirements.txt
pip install -e .
```

### Training

```bash
python scripts/train.py --config configs/default.yaml
```

### Evaluation

```bash
python scripts/evaluate.py --checkpoint checkpoints/best_model.pth --ablation-study --statistical-tests
```

## Key Results

| Metric | Target | Achieved |
|--------|--------|----------|
| P99 Latency Reduction vs Static | 35% | 67.2% |
| Accuracy Degradation vs Full Model | <2% | 14.5% |
| Throughput Improvement (RPS) | 2.5x | 1.02x |
| SLA Violation Rate | <1% | 58.3% |
| Policy Convergence Epochs | <50 | 29 |

**Note:** The trained policy converged to a degenerate routing strategy that sends 100% of traffic to the quantized (route 0) variant. This achieves strong latency reduction (67.2%) but at the cost of significant accuracy degradation (14.5%) and high SLA violation rates (58.3%) driven by accuracy threshold breaches. The routing accuracy against optimal decisions is 10.17% (performance MSE: 1628.34). These results indicate the multi-objective reward balancing requires further tuning to avoid mode collapse toward a single route.

## Architecture

The system consists of three main components:

1. **Query Difficulty Estimator**: Analyzes input complexity and predicts computational requirements
2. **Multi-Objective Router**: Uses PPO-based reinforcement learning to balance latency, accuracy, throughput, and SLA compliance
3. **Model Cascade**: Manages four model variants with different performance characteristics

## Technical Approach

- **Multi-Objective Optimization**: Balances competing objectives using learned weights
- **Reinforcement Learning**: PPO algorithm with GAE for policy optimization
- **Attention Mechanism**: Multi-head attention for feature fusion
- **Statistical Analysis**: Comprehensive evaluation with significance testing

## Project Structure

```
├── src/adaptive_inference_router_with_cascade_serving/
│   ├── data/              # Data loading and preprocessing
│   ├── models/            # Model architectures
│   ├── training/          # Training pipeline
│   ├── evaluation/        # Metrics and analysis
│   └── utils/            # Configuration and utilities
├── scripts/              # Training and evaluation scripts
├── tests/               # Test suite
├── configs/             # Configuration files
└── notebooks/           # Exploration notebooks
```

## Configuration

The system is fully configurable via YAML files. See `configs/default.yaml` for all available options including:

- Model architecture parameters
- Training hyperparameters
- Multi-objective weights
- Evaluation metrics
- System monitoring settings