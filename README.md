# Python Chess GUI with PyTorch ML Engine

This project combines a Python-based chess GUI with a machine learning chess engine. The GUI allows you to play chess against either Stockfish (a traditional chess engine) or my custom PyTorch-based ML chess engine. The project is based on the [Python-Easy-Chess-GUI](https://github.com/fsmosca/Python-Easy-Chess-GUI) project.

## Project Overview

The project consists of two main components:
1. A chess GUI that supports UCI protocol engines
2. A PyTorch-based machine learning chess engine that uses neural networks to evaluate positions

## Prerequisites

1. Python 3.7 or higher
2. PyTorch
3. Python-Chess library
4. NumPy
5. zstandard (for processing PGN files)
6. tqdm (for progress bars)

## Installation and Running Instructions

1. Clone the repository and navigate to the project directory:
```bash
git clone <repository-url>
cd PythonChess_2_Assingment_2
```

2. Create and activate a virtual environment:
```bash
# Create virtual environment
python -m venv venv

# Activate on Unix/macOS
source venv/bin/activate

# Activate on Windows
# .\venv\Scripts\activate
```

3. Install the dependencies:
```bash
# Install GUI dependencies
cd Python-Easy-Chess-GUI
pip install -r requirements.txt

# Install ML engine dependencies
cd Engines/ML_Chess_Engine
pip install -r requirements.txt
```

4. Running the Chess GUI:
```bash
# Make sure you're in the GUI directory
cd /path/to/PythonChess_2_Assingment_2/Python-Easy-Chess-GUI

# Activate virtual environment if not already activated
source ../venv/bin/activate  # Unix/macOS
# .\venv\Scripts\activate    # Windows

# Run the GUI
python python_easy_chess_gui.py
```

5. Setting up the ML Engine in GUI:
   - Click on "Mode" in the menu bar
   - Select "Play" to enable game mode
   - Click on "Engine" in the menu bar
   - Select "Set Engine Opponent"
   - Choose "ML_Chess_Engine" from the list
   - The engine is now ready to play

6. Configuring Engine Settings:
   - Click on "Engine" → "Manage" → "Edit"
   - Select "ML_Chess_Engine"
   - Current settings:
     - Search depth: 7 plies
     - Move time: 5000ms (5 seconds)
   - These can be adjusted based on your preference

## Machine Learning Approach

The chess engine uses a hybrid approach combining neural network evaluation with traditional chess search techniques:

1. **Neural Network Architecture**:
   - Input: 768-dimensional feature vector (8x8 board × 12 piece types)
   - Hidden layers:
     - 512 neurons with ReLU + BatchNorm + 30% dropout
     - 256 neurons with ReLU + BatchNorm + 30% dropout
     - 128 neurons with ReLU + BatchNorm + 30% dropout
   - Output: Single neuron with tanh activation (-1 to 1)

2. **Position Representation**:
   - Each square is encoded based on piece type and color
   - Empty squares are represented as zeros
   - Pieces are encoded using a consistent mapping (e.g., 1 for white pawn, -1 for black pawn)

3. **Search Algorithm**:
   - Minimax search with alpha-beta pruning
   - Neural network provides position evaluations
   - Move ordering optimization:
     - Captures are prioritized
     - Higher value captures are tried first
     - Promotions are given high priority

4. **Time Management**:
   - Default move time: 5 seconds
   - Adjusts based on remaining time and increment
   - Early cutoff if time limit is reached

## Training Process

The engine learns from Lichess game databases:

1. **Data Processing**:
   - Reads zstandard-compressed PGN files
   - Samples positions from games
   - Converts positions to feature vectors

2. **Training Details**:
   - Batch size: 1024
   - Optimizer: Adam (learning rate 0.001)
   - Loss function: Mean Squared Error
   - 10 epochs of training
   - Trained on 1 million positions

## Project Structure

### ML Engine Components
Located in `Engines/ML_Chess_Engine/`:

#### Core Source Files (`src/`)
- `model.py`: Neural network implementation
  - Defines `ChessNet` class with the neural architecture
  - Implements `ChessModel` class for training and evaluation
  - Handles model saving/loading and batch normalization
  - Provides evaluation interface for the engine

- `data_processor.py`: Position processing and feature extraction
  - Converts chess positions to neural network inputs
  - Implements board-to-feature vector conversion (768 dimensions)
  - Handles piece encoding and position normalization
  - Provides utilities for processing PGN game files

- `uci_engine.py`: Main engine implementation
  - Implements the UCI (Universal Chess Interface) protocol
  - Contains the core search algorithm (alpha-beta pruning, depth 7)
  - Manages time controls and move selection
  - Provides fallback mechanisms for error handling
  - Integrates neural evaluation with search

#### Support Files
- `train.py`: Training script
  - Handles dataset creation from PGN files
  - Manages the training loop and optimization
  - Implements position sampling and augmentation
  - Saves the trained model checkpoints

- `run_engine.sh`: Engine execution script
  - Sets up the Python environment
  - Configures necessary paths
  - Launches the UCI engine interface

#### Model Files
- `models/chess_model.pth`: The trained PyTorch model
  - Contains trained network weights
  - Includes batch normalization parameters
  - ~6.4MB in size

#### Dependencies
- `requirements.txt`: Python package dependencies
  - PyTorch for neural network operations
  - python-chess for game logic
  - zstandard for PGN processing
  - NumPy for numerical operations
  - tqdm for progress tracking

## Troubleshooting

If you encounter any issues:

1. **Engine Not Found**:
   - Verify the engine path in `pecg_engines.json`
   - Check that `run_engine.sh` has execute permissions
   - Ensure virtual environment is activated

2. **Model Loading Issues**:
   - Verify `chess_model.pth` exists in the models directory
   - Check PyTorch installation matches your system (CPU/GPU)
   - The engine will fallback to material evaluation if model fails

3. **Performance Issues**:
   - Reduce search depth if moves are too slow
   - Adjust move time in engine settings
   - Check system resources and CPU usage

## Credits

- [Python-Easy-Chess-GUI](https://github.com/fsmosca/Python-Easy-Chess-GUI) for the GUI
- [PyTorch](https://pytorch.org/) for machine learning capabilities
- [python-chess](https://python-chess.readthedocs.io/) for chess logic
- [Lichess](https://lichess.org/) for the training database 