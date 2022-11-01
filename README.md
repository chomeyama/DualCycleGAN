# Dual-CycleGAN for NonParallel High-Quality Audio Super-Resolution

This repositry provides official pytorch implementation of [Dual-CycleGAN](https://arxiv.org/abs/2210.15887).<br>
Specifically, Dual-CycleGAN enables you to train a high-quality super resolution (SR) model (e.g., 16kHz -> 48kHz) only with low-resolution audio signals of the target domain with high-resolution audio signals of another domain.

Please check the [DEMO](https://chomeyama.github.io/DualCycleGAN-Demo/) for more information.

## Environment setup

```bash
cd DualCycleGAN
pip install -e .
```

This will install the core library (`dual-cyclegan` and its submodules) and CLI tools (e.g., `dual-cyclegan-train`).
Please refer to the [Parallel WaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN) repo for more details.

## Folder architecture

- **egs**:
The folder for projects.
- **egs/tts_16kHz.gt_48kHz**:
The folder of the "TTS + Ground truth" project example.
- **dual_cyclegan**:
The folder of the source codes.

## Run

In this repo, hyperparameters are managed using [Hydra](https://hydra.cc/docs/intro/).<br>
Hydra provides an easy way to dynamically create a hierarchical configuration by composition and override it through config files and the command line.

### Dataset preparation

Make dataset and list files denoting paths to each audio files according to your own dataset (E.g., `egs/tts_16kHz.gt_48kHz/data/list/tts_train_16kHz.list`). Note that list files for each training/validation/evaluation are required.

### Training

```bash
# Train a model customizing the hyperparameters as you like
$ dual-cyclegan-train data=tts_16kHz.gt_48kHz model=dual_cyclegan train=dual_cyclegan out_dir=exp/dual_cyclegan
```

### Inference

```bash
# Infer with a trained model from a checkpoint file
$ dual-cyclegan-infer data=tts_16kHz.gt_48kHz model=dual_cyclegan out_dir=exp/dual_cyclegan/wav checkpoint_path=exp/dual_cyclegan/checkpoint-600000steps.pkl
```

### Monitor training progress

```bash
tensorboard --logdir exp
```

### Details of the list files

List files contain the path to audio files used for training, validation, and evaluation, respectively. <br>
A set of the three files indicates distribution of a single dataset. <br>
Please note that Dual-CycleGAN requires two sets of list files because it is trained on two kinds of datasets. <br>
Please check `egs` directory for examples of the intended directory structure.

### Details of the list files

List files contain the path to audio files used for training, validation, and evaluation, respectively. <br>
A set of the three files indicates distribution of a single dataset. <br>

```
# An example of the project directory structure
cd DualCycleGAN/egs/tts_16kHz.gt_48kHz/
  |- data
  |    |- list
  |        |- tts_train_no_dev.list
  |        |- tts_dev.list
  |        |- tts_eval.list
  |        |- gt_train_no_dev.list
  |        |- gt_dev.list
  |        |- gt_eval.list
  |- exp
       |- dual_cyclegan
```

```
# An example of the list file
path_to_your_own_dataset_dir/audio1.wav
path_to_your_own_dataset_dir/audio2.wav
path_to_your_own_dataset_dir/audio3.wav
...
```
