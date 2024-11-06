<div align="center">
<h1>ComplexNDN: Complex Neural Dynamics Models</h1>

![Static Badge](https://img.shields.io/hexpm/l/plug)
![GitHub top language](https://img.shields.io/github/languages/top/xinyuanliao/complexndm)
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/xinyuanliao/complexndm)
![Static Badge](https://img.shields.io/badge/Framework-TensorFlow_v2-orange)
![Static Badge](https://img.shields.io/badge/Test_Platform-Windows-pink)
</div>

This repository includes the code for the paper _Parallelizable Complex Neural Dynamics Models for Temperature Estimation with Hardware Acceleration_.

Since _**PyTorch**_ does not support parallel scanning and _**JAX**_ does not support Windows x64_86, this repo is built by _**TensorFlow-gpu**_, and the parallel prefix sum scanning algorithm is implemented by the [```tfp.math.scan_associative```](https://www.tensorflow.org/probability/api_docs/python/tfp/math/scan_associative). The parallel prefix sum algorithm accelerates the training process by at least **1.6 times** and the inference process by at least **1.8 times**. The parallel algorithm reduces the time complexity from _**O(N)**_ to _**O(logN)**_ for serial calculations. As the estimation length increases, the acceleration effect becomes more obvious.

<p align="center">
  <img src="https://github.com/XinyuanLiao/complexNDM/blob/main/Figs/frame.jpg" width="1000px"/>
</p>

# Quick start
## Dataset
Download the dataset from [Kaggle](https://www.kaggle.com/datasets/wkirgsn/electric-motor-temperature) and put it into the [Data](https://github.com/XinyuanLiao/complexNDM/tree/main/Data) folder.
<p align="center">
  <img src="https://github.com/XinyuanLiao/complexNDM/blob/main/Figs/dataset.jpg" width="1400px"/>
</p>

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
python trainer.py --scan True --estimation_length 128 --phase 0.314 --hidden_size 32
```

# Star History

[![Star History Chart](https://api.star-history.com/svg?repos=XinyuanLiao/complexNDM&type=Date)](https://star-history.com/#XinyuanLiao/complexNDM&Date)


# Cite as
```
@misc{liao2024parallelizable,
      title={Parallelizable Complex Neural Dynamics Models for Temperature Estimation with Hardware Acceleration},
      author={Xinyuan Liao, Shaowei Chen, Shuai Zhao},
      Url= {https://github.com/XinyuanLiao/complexNDN}, 
      year={2024}
}
```
