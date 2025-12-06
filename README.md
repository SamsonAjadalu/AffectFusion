# RAVDESS live audio emotion predictor
Binary emotion classifier trained on python 3.9.23, tensorflow 2.15.0. Source preprocessing and training notebook for submission found in RAVDESS.

## Py and Tf versions
Create conda environment:
```
conda create -n tf_env python=3.9.23
conda activate tf_env
```
Make sure you are running with **python** not the system **py**. py uses the latest python and will not cooperate with tf 2.15.0

## Live inference demo
Install demo requirements:
```
pip install -r demo-requirements.txt
```

## Training and source code
Install training requirements:
```
pip install - training-requirements.txt
```
