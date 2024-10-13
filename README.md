# Automatic Speech Recognition (ASR) 

<p align="center">
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#final-metrics">Final Metrics</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

## About

This repository contains my implementation of DeepSpeech2. The model has been trained for 91 epochs, which is approximately 12 hours, on the LibriSpeech dataset.

More details with the description of the model, experiments runs and parameters used can be found in my report [here](https://api.wandb.ai/links/aavgustyonok/rxllkqko).

See the task assignment [here](https://github.com/markovka17/dla/tree/2024/hw1_asr).

## Installation

Follow these steps to install the project:

0. (Optional) Create and activate new environment using [`conda`](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) or `venv` ([`+pyenv`](https://github.com/pyenv/pyenv)).

   a. `conda` version:

   ```bash
   # create env
   conda create -n project_env python=PYTHON_VERSION

   # activate env
   conda activate project_env
   ```

   b. `venv` (`+pyenv`) version:

   ```bash
   # create env
   ~/.pyenv/versions/PYTHON_VERSION/bin/python3 -m venv project_env

   # alternatively, using default python version
   python3 -m venv project_env

   # activate env
   source project_env
   ```

1. Install all required packages

   ```bash
   pip install -r requirements.txt
   ```

2. Download deepspeech and language model weights:
   ```bash
   python get_weights.py
   ```

## How To Use

To run inference (evaluate the model or save predictions):

```bash
python inference.py HYDRA_CONFIG_ARGUMENTS
```
By default (without any additional arguments) inference is run on the LibriSpeech test-other dataset. If you want to test the model on the test-other data, you need to specify corresponding parameter (it can be also found in src/configs/datasets/test.yaml). Be careful with the paths to the directories with models weights (by default they will be saved in asr/saved directory, as can be seen in inference.yaml). 

## Final Metrics

|Method|WER test-clean| WER test-other |
|------|--------------|----------------|
|basic model| 18.4 | 39.5 |
|basic model + my beam search | 17.9 | 38.9 |
| basic model + lm beam search | 11.2 | 29.1 |

## Credits

This repository is based on a [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template).

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
