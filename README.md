# Outerplanar GNNs
## Setup
Clone this repository and open the directory

Add this directory to the python path. Let `$PATH` be the path to where this repository is stored (i.e. the result of running `pwd`).
```
export PYTHONPATH=$PYTHONPATH:$PATH
```

Create a conda environment (this assume miniconda is installed)
```
conda create --name GNNs
```

Activate environment
```
conda activate GNNs
```

Install dependencies
```
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 -c pytorch
conda install -c pyg pyg=2.2.0
pip install -r requirements.txt
```

## Replicating the experiments
Results can be found in the results directory.

Baselines (GIN):
```
python Exp/run_experiment.py -grid Configs/Eval/GIN_zinc.yaml -dataset ZINC --candidates 48 --repeats 10
python Exp/run_experiment.py -grid Configs/Eval/GIN_molhiv.yaml -dataset ogbg-molhiv --candidates 16 --repeats 10
```

New models (CAT+GIN):
```
python Exp/run_experiment.py -grid Configs/Eval/cat_molhiv.yaml -dataset ogbg-molhiv --candidates 16 --repeats 10
python Exp/run_experiment.py -grid Configs/Eval/cat_zinc.yaml -dataset ZINC --candidates 48 --repeats 10
```
