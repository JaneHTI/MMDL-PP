1. Installation
conda env create -f environment.yml
conda activate MMDLPP

2. Usage
- demo_main.py # Model training and testing, along with detailed specifications of configuration parameters and core functional components.
- lib          # Data loading and model setting.
- data         # Data
- checkpoints  # Trained model
- utils        # Function for result analysis.

Executing the script demo_main.py directly enables evaluation of the model—pretrained on the ABCD dataset—on the provided sample data; all required configuration parameters are pre-specified within the source code.