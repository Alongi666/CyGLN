# CyGLN
Paper Code ”Global-Local Evolution Modeling with Cyclic Patterns for Temporal Knowledge Graph“

1、You need to run the file generate_data.py to generate the graph data needed for our model:

python generate_data.py --data=DATA_NAME

2、In order to speed up training and testing, for ICEWS18, ICEWS05-15, and GDELT datasets, data in the required format can be constructed in advance before training and testing:

python save_data.py --data=DATA_NAME

Training and Testing
Then you can run the file main.py to train and test our model. The detailed commands can be found in {dataset}.sh. Some important hyper-parameters can be found in long_config.yaml and short_config.yaml.

Requirements
Make sure you have the following dependencies installed:

Python~=3.9.0
dgl~=1.1.2.cu116
torch~=1.12.1+cu116
numpy~=1.26.3
tqdm~=4.65.0
pandas~=2.1.4
scipy~=1.13.0
