1. Installation
conda env create -f environment.yml
conda activate MMDLPP

2. Usage
- demo_main.py # Model training and testing, along with detailed specifications of configuration parameters and core functional components.
- lib          # Data loading and model setting.
- data         # Data
- checkpoints  # Trained model
- utils        # Function for result analysis.

3. Demo
Executing the script demo_main.py directly enables evaluation of the pretrained model—trained on the ABCD dataset—on the provided sample data. All required configuration parameters are pre-specified within the source code. The output includes the probability of psychopathology risk for each sample, along with aggregated statistical results stored in the checkpoint file.
