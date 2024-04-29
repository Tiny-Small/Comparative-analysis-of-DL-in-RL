# Comparative Analysis of Deep Learning Architectures in Reinforcement Learning: Case Study on Pac-Man

## Project Summary
This study explores various deep learning architectures within a reinforcement learning framework, applied specifically to the game of Pac-Man. Our objective was to identify the most efficient neural network architecture capable of achieving high scores with minimal training. We evaluated several architectures including Fully Connected Neural Networks, Convolutional Neural Networks (CNNs), and Multi-head Attention Networks, with and without temporal awareness.

Our findings revealed that CNNs, including those with limited perceptual input, displayed remarkable adaptability and efficiency. Despite incorporating advanced features like LSTM layers to capture temporal dynamics, we found that simpler spatial processing models were generally more effective for the *dynamics* of Pac-Man, indicating that complex models with temporal awareness did not significantly enhance performance in our tests.

Further experiments assessed the adaptability of the models to new and varied layouts, indicating that while trained models generally outperformed untrained models, training in more complex environments did not always translate to better performance in simpler settings. This suggests the need for careful consideration of training environments to optimize neural network performance across different scenarios.

Future research will aim to explore the optimization of neural networks in diverse gaming environments and determine under which conditions the incorporation of temporal information can enhance performance, potentially extending beyond gaming to other applications requiring rapid and precise decision-making.


## Video on some of Pac-Man's better gameplay

### Small Classic layout
[![Watch the video on Small Classic layout with Limited Visibility](http://img.youtube.com/vi/34Z7iSKI-qw/0.jpg)](https://youtu.be/34Z7iSKI-qw)

### Original Classic layout
[![Watch the video on Original Classic layout with Limited Visibility](http://img.youtube.com/vi/-HFZcXrcgsg/0.jpg)](https://youtu.be/-HFZcXrcgsg)


## How to run
To run the game with specific configurations, use the following command format:
```
python3 pacman.py -p <AGENT> -n <NUM OF GAMES> -x <NUM OF TRAINING> -l <LEVEL NAME>
```

### Example command to start a game:
```
python3 pacman.py -p PacmanTDQN -n 5 -x 5000 -l originalClassic
```
Where
- `-p PacmanTDQN` specifies the agent type.
- `-n 5` will display 5 games through the graphics.
- `-x 5000` sets the number of training games to 5000.
- `-l originalClassic` specifies the layout to be played. Details of each layout can be found in the layouts folder.

### To Play the Game Interactively:
You can also play the game interactively using the keyboard:
```
python3 pacman.py -p KeyboardAgent -l smallClassic
```
This command configures the game for manual control in a specific layout.

### List of agents
For the `<AGENT>` argument, the following agents are available:
- KeyboardAgent (Manual control using up, down, left, right keys)
- PacmanTDQN


## For Finer Controls

### 1. Step limit
To adjust the step limit, which controls the number of steps Pac-Man can take before the game terminates, modify the value in `game.py` at line 29. It is currently set to 1000.
```
STEPLIMIT = 1000
```

### 2. Scores
The scores for various game actions can be adjusted in `pacman.py`, from lines 55 to 60.
```
EAT_GHOST = 0       # No points for eating a ghost
LOSE_GAME = 500     # Penalty for losing the game
WIN_GAME = 5000     # Reward for winning the game
EAT_FOOD = 50       # Points for eating a food pellet
IDLE_AROUND = 10    # Penalty for idle moves
```

### 3. pacmanTDQN Adjustments
This section details how to adjust `pacmanTDQN_Agents.py` in the agents folder, encompassing everything from neural network choices to visibility settings and learning parameters. Each configuration impacts how the agent learns and performs in the game environment.

#### 3.1 Neural Network Selection
Configure the neural network architecture by uncommenting the appropriate line. This allows you to switch between different models like Fully Connected Networks, CNNs with various depths, and Multi-head Attention Networks with or without temporal awareness.
```
# Neural nets (Choose the neural network)
from dqn.FullyConnected_NN import *
...
```

#### 3.2 Field of Vision
Toggle Pac-Man's field of vision to be restricted or full by modifying the `is_sliced` variable. A value of `True` restricts the vision to a 7x7 grid around Pac-Man.
```
# Field of vision
is_sliced = True
```

#### 3.3 Training and Miscellaneous Settings
Define whether the model is in training or testing mode, specify paths for saving and loading neural network weights, and set up logging.
```
# Training or testing mode
TRAINING_MODE = True
...
# Path settings for saving and loading model weights
save_path_targetqnet = 'saves/FC_NN/target_qnet'
load_file_targetqnet = 'saves/originalClassic/CNN2/target_qnet/ckpt-50'
```

#### 3.4 Model Parameters
Fine-tune model parameters such as batch size, memory size, learning rate, epsilon values, and update frequencies to optimize training and performance.
```
# Model parameters (examples)
'batch_size': 32,
'mem_size': 15000,
'discount': 0.9,
'lr': 0.00025,
'eps_start': 1,
'eps_final': 0.05,
'update_target_freq': 10000,
'DQN_or_DDQN':'DQN',
...
```


## Requirements
```
python==3.10.6
tensorflow==2.15.1
Cython==3.0.10
```

## Special Thanks
Special thanks to [r0b0pp1](https://github.com/r0b0pp1) for her contributions to the project.


## Acknowledgements
- **UC Berkeley**: [Project Overview: Berkeley AI Pac-Man Projects](http://ai.berkeley.edu/project_overview.html) (Accessed: 21 April 2024).
- **Tycho van der Ouderaa (2016)**: *Deep Reinforcement Learning in Pac-man*. Bachelor Thesis, University of Amsterdam. [GitHub Project](https://github.com/tychovdo/PacmanDQN)
