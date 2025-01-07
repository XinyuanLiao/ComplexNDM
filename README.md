<div align="center">
<h1>ComplexNDN: Complex Neural Dynamics Models</h1>

![Static Badge](https://img.shields.io/hexpm/l/plug)
![GitHub top language](https://img.shields.io/github/languages/top/xinyuanliao/complexndm)
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/xinyuanliao/complexndm)
![Static Badge](https://img.shields.io/badge/Framework-JAX-orange)
![Static Badge](https://img.shields.io/badge/Test_Platform-Linux-pink)
</div>

This repository includes the code for the paper _Parallelizable Complex Neural Dynamics Models for Temperature Estimation with Hardware Acceleration_.

Since _**PyTorch**_ does not support parallel scanning and _**JAX**_ does not support Windows x64_86, this repo is built by _**JAX**_, and the parallel prefix sum scanning algorithm is implemented by the [```jax.lax.associative_scan```](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.associative_scan.html). The parallel algorithm reduces the time complexity from _**O(N)**_ to _**O(logN)**_ for serial calculations. 
<p align="center">
  <img src="https://github.com/XinyuanLiao/complexNDM/blob/main/Figs/frame.jpg" width="1000px"/>
</p>

# Quick start

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
kagglehub==0.3.6
```
## Run
Run the training program from the command line.

```
python trainer.py --scan --is_PILF
```

# Star History

[![Star History Chart](https://api.star-history.com/svg?repos=XinyuanLiao/complexNDM&type=Date)](https://star-history.com/#XinyuanLiao/complexNDM&Date)


# Cite as
```
@article{liao5033161parallelizable,
  title={Parallelizable Complex Neural Dynamics Models for Temperature Estimation with Hardware Acceleration},
  author={Liao, Xinyuan and Chen, Shaowei and Zhao, Shuai},
  journal={Available at SSRN 5033161},
  year={2024}
}
```
