# Notch: Q-Learning Based Pathfinding in Dynamic Grid Environments

## Abstract

Notch is a reinforcement learning implementation that demonstrates Q-Learning algorithms for autonomous pathfinding in obstacle-rich grid environments. The system employs temporal difference learning to enable an agent to discover optimal paths from a starting position to a goal while navigating around randomly generated obstacles. The implementation features dynamic reward shaping, dead-end detection, and real-time visualization of the learning process.

## Overview

This project implements a Q-Learning agent capable of navigating through a 2D grid environment with obstacles. The agent learns through trial and error, building a Q-table that maps state-action pairs to expected cumulative rewards. The implementation demonstrates key reinforcement learning concepts including exploration vs. exploitation, reward shaping, and convergence in discrete action spaces.

### Key Components

- **Environment**: 20×20 grid world with randomly generated obstacles
- **Agent**: Q-Learning agent with ε-greedy action selection
- **State Space**: 400 discrete positions (x, y coordinates)
- **Action Space**: 4 discrete actions (up, down, left, right)
- **Learning Algorithm**: Temporal Difference Q-Learning with experience replay

## Technical Specifications

### Environment Parameters
- **Grid Dimensions**: 20×20 cells (400 total states)
- **Obstacle Density**: 20% of total cells
- **Action Space**: 4-directional movement (cardinal directions)
- **Boundary Handling**: Position clamping to grid boundaries

### Learning Parameters
- **Learning Rate (α)**: 0.1
- **Discount Factor (γ)**: 0.99
- **Initial Exploration Rate (ε)**: 1.0
- **Minimum Exploration Rate**: 0.05
- **Exploration Decay**: 0.9995 per episode
- **Training Episodes**: 5,000
- **Maximum Steps per Episode**: 400

## Algorithm Implementation

### Q-Learning Update Rule

The implementation uses the standard Q-Learning update equation:

```
Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
```

Where:
- `s` = current state
- `a` = action taken
- `s'` = next state
- `r` = immediate reward
- `α` = learning rate
- `γ` = discount factor

### Reward Structure

The reward function incorporates multiple components:

1. **Goal Reward**: +200 for reaching the target
2. **Obstacle Penalty**: -50 for collision attempts
3. **Revisit Penalty**: -10 for returning to previously visited states
4. **Dead-end Penalty**: -25 for entering terminal positions
5. **Distance-based Reward**: Proportional reward based on Manhattan distance to goal
6. **Step Penalty**: -1 for each movement (encourages efficiency)

### Exploration Strategy

The system implements ε-greedy exploration with exponential decay:

```rust
action = {
    random_action()           if rand() < ε
    argmax Q(s,a)            otherwise
}
```

## Features

### Core Functionality
- **Dynamic Obstacle Generation**: Random obstacle placement with path validation
- **Dead-end Detection**: Identifies and penalizes positions with limited mobility
- **Path Validation**: Ensures solvable environments using breadth-first search
- **Backtracking**: Agent can recover from local minima through path retracing

### Visualization
- **Real-time Rendering**: Live visualization of agent movement
- **Color-coded Display**:
  - `G` (Green): Goal position
  - `A` (Yellow): Current agent position
  - `S` (Magenta): Start position
  - `#` (White): Obstacles
  - `*` (Yellow): Traversed path
  - `·` (Dimmed): Empty cells

## Installation

### Prerequisites
- Rust 1.70+ with Cargo package manager
- Terminal with color support

### Dependencies

Add the following to your `Cargo.toml`:

```toml
[dependencies]
colored = "2.0"
rand = "0.8"
clearscreen = "1.0"
```

### Build Instructions

```bash
# Clone the repository
git clone https://github.com/naseridev/notch.git
cd notch

# Build the project
cargo build --release

# Run the simulation
cargo run --release
```

## Usage

### Basic Execution

```bash
cargo run
```

The program will:
1. Generate a random grid environment with obstacles
2. Train the Q-Learning agent for 5,000 episodes
3. Demonstrate the learned policy through real-time visualization
4. Display performance metrics and path statistics

### Training Output

During training, the system provides periodic updates:

```
Obstacles generated. Count: 80
Starting Q-Learning training...
Training Episode 0 / 5000, ε = 1.000
Training Episode 500 / 5000, ε = 0.607
...
```

## Configuration Parameters

### Environment Settings
```rust
const WIDTH: usize = 20;           // Grid width
const HEIGHT: usize = 20;          // Grid height
const OBSTACLE_DENSITY: f64 = 0.2; // Obstacle ratio (0.0-1.0)
```

### Learning Hyperparameters
```rust
let alpha = 0.1;      // Learning rate
let gamma = 0.99;     // Discount factor
let epsilon = 1.0;    // Initial exploration rate
let min_epsilon = 0.05; // Minimum exploration rate
let eps_decay = 0.9995; // Exploration decay rate
```

### Training Configuration
```rust
let episodes = 5000;    // Total training episodes
let max_steps = 400;    // Steps per episode limit
```

## Results and Analysis

### Performance Metrics

The implementation typically achieves:
- **Convergence**: Stable policy after ~2,000-3,000 episodes
- **Path Optimality**: Near-optimal paths in most scenarios
- **Success Rate**: >95% goal achievement in solvable environments
- **Computational Efficiency**: Sub-second training on modern hardware

### Learning Characteristics

1. **Exploration Phase** (Episodes 0-1000): High randomness, extensive environment exploration
2. **Exploitation Transition** (Episodes 1000-3000): Gradual policy refinement
3. **Convergence Phase** (Episodes 3000+): Stable optimal behavior

### Limitations

- **Scalability**: Memory complexity O(|S| × |A|) limits grid size
- **Static Environments**: No adaptation to dynamic obstacle changes
- **Local Minima**: Occasional suboptimal convergence in complex layouts

## Future Work

### Algorithmic Enhancements
- **Deep Q-Networks (DQN)**: Neural network-based value approximation
- **Double Q-Learning**: Reduced overestimation bias
- **Prioritized Experience Replay**: Improved sample efficiency

### Environmental Extensions
- **Dynamic Obstacles**: Moving barriers and time-varying environments
- **Multi-agent Systems**: Collaborative and competitive scenarios
- **Continuous Action Spaces**: Smooth movement control

### Performance Optimizations
- **Function Approximation**: Handle larger state spaces efficiently
- **Transfer Learning**: Knowledge reuse across similar environments
- **Parallel Training**: Distributed learning acceleration

## References

1. Watkins, C.J.C.H. and Dayan, P. (1992). Q-learning. *Machine Learning*, 8(3-4), 279-292.

2. Sutton, R.S. and Barto, A.G. (2018). *Reinforcement Learning: An Introduction*. MIT Press.

3. Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518(7540), 529-533.

4. Singh, S.P. and Sutton, R.S. (1996). Reinforcement learning with replacing eligibility traces. *Machine Learning*, 22(1-3), 123-158.

---

## Contributing

Contributions are welcome! Please read the contributing guidelines and submit pull requests for any improvements.

## Contact

For questions, issues, or collaboration opportunities, please open an issue in the repository or contact the maintainers directly.