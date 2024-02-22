# complexNDM
This repository includes the code for the paper _Parallelizable Complex Neural Dynamics Models for Temperature Estimation_.

Since Pytorch does not support parallel scanning, this repo is built by TensorFlow, and the parallel scanning algorithm is implemented by the [```tfp.math.scan_associative```](https://www.tensorflow.org/probability/api_docs/python/tfp/math/scan_associative).

```mode``` is used to select the algorithm execution way, default is parallel ```scan```.
# Quick start
## Dataset
Download the dataset from this [website](https://www.kaggle.com/wkirgsn/electric-motor-temperature) and put it into the [Data](https://github.com/XinyuanLiao/complexNDM/tree/main/Data) folder.
## Configuration
```
python==3.7.0
numpy==1.21.6
pandas==1.3.4
tensorflow==2.4.1
tensorflow_probability==0.12.1
```
## Run
```
python trainer.py
```
