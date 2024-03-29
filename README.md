# Intra Order-preserving Functions for Calibration of Multi-Class Neural Networks

This repository is the official implementation of NeurIPS 2020 paper: [Intra Order-preserving Functions for Calibration of Multi-Class Neural Networks](https://arxiv.org/abs/2003.06820)

![Order preserving/invariant architecture](https://github.com/AmirooR/IntraOrderPreservingCalibration/blob/main/architecture.001.png?raw=true)

## Requirements

To create the environment, you should have python3 and virtualenv installed, run these commands inside the source code directory:

```bash
virtualenv -p python3 venv
source venv/bin/activate                                 
pip3 install torch torchvision tqdm easydict scikit-learn
```

## Data
Use this [google drive link](https://drive.google.com/file/d/18QvhqtfUY77xV7tZa25q4h4BLsbeFmxV/view?usp=sharing) to download all of the logits in pickle format.

## Setup

Put the corresponding logits to the data folder. 
The configs have the following path format:
`exp_dir/{dataset}/{model}/{method}/config.json`,
where dataset, model, and method should be replaced with the desired dataset, model, and method's name.
(e.g., ["exp_dir/CIFAR100/ResNet110/OI/config.json"](https://github.com/AmirooR/IntraOrderPreservingCalibration/blob/main/exp_dir/CIFAR100/ResNet110/OI/config.json))

## Training

To train the calibrator network:

```bash
python calibrate.py --exp_dir exp_dir/{dataset}/{model}/{method}
```

## Evaluation

To evaluate the trained networks:

```bash
python evaluate.py --exp_dir exp_dir/{dataset}/{model}/{method}
```

## Results

The results will be saved in json format in the config dirname. As an example: `"ensemble/post_metrics_test_ensemble_best_ece.json"` corresponds to the ECE values reported in Table 1 of the paper and `"cross_val_test_post_metrics_best_ece.json"` corresponds to the results without ensemble (by averaging the metrics over different folds). Note that the results might be slightly different from the reported numbers in the paper due to randomness in training. 

### Update

There was a [bug](https://github.com/AmirooR/IntraOrderPreservingCalibration/issues/3) in our evaluation code. Our results slightly improved for the ECE metric after fixing the issue. Thanks to [@futakw](https://github.com/futakw). The updated table is shown below:

![ECE Table](https://github.com/AmirooR/IntraOrderPreservingCalibration/blob/main/ECETable.png?raw=true)



## Cite

If you make use of this code in your own work, please cite our paper:

```
@inproceedings{rahimi2020intra,
  title={Intra Order-preserving Functions for Calibration of Multi-Class Neural Networks},
  author={Rahimi, Amir and Shaban, Amirreza and Cheng, Ching-An and Hartley, Richard and Boots, Byron},
  booktitle={Advances in Neural Information Processing Systems},
  year={2020}
}
```
