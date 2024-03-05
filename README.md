<div align="center">
<h1>ComplexNDN: Complex Neural Dynamics Models</h1>

![Static Badge](https://img.shields.io/hexpm/l/plug)
![languages](https://img.shields.io/github/languages/top/XinyuanLiao/complexNDN)
![Size](https://img.shields.io/github/languages/code-size/XinyuanLiao/complexNDN)
![Static Badge](https://img.shields.io/badge/Framework-TensorFlow-orange)
![Static Badge](https://img.shields.io/badge/Platform-Win_|_Mac-pink)
</div>

This repository includes the code for the paper _Parallelizable Complex Neural Dynamics Models for Temperature Estimation with Hardware Acceleration_.

Since Pytorch does not support parallel scanning and JAX does not support Windows x86, this repo is built by TensorFlow, and the parallel scanning algorithm is implemented by the [```tfp.math.scan_associative```](https://www.tensorflow.org/probability/api_docs/python/tfp/math/scan_associative).

<p align="center">
  <img src="https://github.com/XinyuanLiao/complexNDM/blob/main/Figs/frame.jpg" width="1000px"/>
</p>

# Quick start
## Dataset
Download the dataset from [Kaggle](https://www.kaggle.com/wkirgsn/electric-motor-temperature) and put it into the [Data](https://github.com/XinyuanLiao/complexNDM/tree/main/Data) folder.
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
Run the training program from the command line.

```
# parallel computing; estimation_length=128; phase range is [-np.pi/10, np.pi/10]; hidden_size=32
python trainer.py --scan True --estimation_length 128 --phase 0.314 --hidden_size 32
```

# Star History

[![Star History Chart](https://api.star-history.com/svg?repos=XinyuanLiao/complexNDM&type=Date)](https://star-history.com/#XinyuanLiao/complexNDM&Date)


# Cite as
```
@misc{liao2024parallelzable,
      title={Parallelizable Complex Neural Dynamics Models for Temperature Estimation with Hardware Acceleration},
      author={Xinyuan Liao, Shaowei Chen, Shuai Zhao},
      Url= {https://github.com/XinyuanLiao/complexNDN}, 
      year={2024}
}
```
