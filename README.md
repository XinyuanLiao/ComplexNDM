# complexNDM
This repository includes the code for the paper _Parallelizable Complex Neural Dynamics Models for Temperature Estimation with Hardware Acceleration_.

Since Pytorch does not support parallel scanning and JAX does not support Windows x86, this repo is built by TensorFlow, and the parallel scanning algorithm is implemented by the [```tfp.math.scan_associative```](https://www.tensorflow.org/probability/api_docs/python/tfp/math/scan_associative).

# Quick start
## Dataset
Download the dataset from this [website](https://www.kaggle.com/wkirgsn/electric-motor-temperature) and put it into the [Data](https://github.com/XinyuanLiao/complexNDM/tree/main/Data) folder.
## Configuration
```
pip install -r requirements.txt
```
details:
```
tensorflow-gpu==2.10.0
tensorflow-probability==0.16.0
pandas==1.4.2
numpy==1.21.0
h5py==3.6.0
```
## Run
### Parallel computing
```
python trainer.py --scan True
```
### Serial computing
```
python trainer.py --scan False
```
### Phase priors
```
python trainer.py --phase 0.314  # phase range is [-np.pi/10, np.pi/10]
python trainer.py --phase 3.14  # phase range is [-np.pi, np.pi]
```
