<div align="center">
<h1>ComplexNDN: Complex Neural Dynamics Models</h1>

![Static Badge](https://img.shields.io/hexpm/l/plug)
![languages](https://img.shields.io/github/languages/top/XinyuanLiao/complexNDN)
![Size](https://img.shields.io/github/languages/code-size/XinyuanLiao/complexNDN)
![Static Badge](https://img.shields.io/badge/Framework-TensorFlow-orange)
![Static Badge](https://img.shields.io/badge/Test_Platform-Windows_x64-pink)
</div>

This repository includes the code for the paper _Parallelizable Complex Neural Dynamics Models for Temperature Estimation with Hardware Acceleration_.

Since PyTorch does not support parallel scanning and JAX does not support Windows x86, this repo is built by TensorFlow, and the parallel prefix sum scanning algorithm is implemented by the [```tfp.math.scan_associative```](https://www.tensorflow.org/probability/api_docs/python/tfp/math/scan_associative).

<p align="center">
  <img src="https://github.com/XinyuanLiao/complexNDM/blob/main/Figs/frame.jpg" width="1000px"/>
</p>

# Quick start
## Dataset
Download the dataset from [Kaggle](https://www.kaggle.com/datasets/wkirgsn/electric-motor-temperature) and put it into the [Data](https://github.com/XinyuanLiao/complexNDM/tree/main/Data) folder.
<p align="center">
  <img src="https://github.com/XinyuanLiao/complexNDM/blob/main/Figs/dataset.jpg" width="1200px"/>
</p>

### Context
>The data set comprises several sensor data collected from a permanent magnet synchronous motor (PMSM) deployed on a test bench. The PMSM represents a german OEM's prototype model. Test bench measurements were collected by the LEA department at Paderborn University.

### Content
>All recordings are sampled at 2 Hz. The data set consists of multiple measurement sessions, which can be distinguished from each other by column "profile_id". A measurement session can be between one and six hours long.
>
>The motor is excited by hand-designed driving cycles denoting a reference motor speed and a reference torque.
>
>Currents in d/q-coordinates (columns "i_d" and i_q") and voltages in d/q-coordinates (columns "u_d" and "u_q") are a result of a standard control strategy trying to follow the reference speed and torque.
>
>Columns "motor_speed" and "torque" are the resulting quantities achieved by that strategy, derived from set currents and voltages.
>
>Most driving cycles denote random walks in the speed-torque-plane in order to imitate real world driving cycles to a more accurate degree than constant excitations and ramp-ups and -downs would.
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
```
```
python trainer.py --scan True --estimation_length 128 --phase 0.314 --hidden_size 32
```

## Training
```
train shape:  (145772, 144, 14)
valid shape:  (32, 144, 14)
test shape:  (53, 144, 14)
Model: "complex_ndm"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 C (cDense)                  multiple                  264

 f_0 (complexMLP)            multiple                  6304

 f_u (complexMLP)            multiple                  4576

=================================================================
Total params: 11,176
Trainable params: 11,040
Non-trainable params: 136
_________________________________________________________________
Epoch 0/10000
142/142 [==============================] - 29s 125ms/step - Loss_inf: 0.3893 - Loss_smth: 0.0196
Valid Loss RMSE: 20.4429

Epoch 1/10000
142/142 [==============================] - 10s 70ms/step - Loss_inf: 0.0521 - Loss_smth: 0.0122
Valid Loss RMSE: 13.3631

Epoch 2/10000
142/142 [==============================] - 10s 72ms/step - Loss_inf: 0.0304 - Loss_smth: 0.0095
Valid Loss RMSE: 10.8444

Epoch 3/10000
142/142 [==============================] - 10s 71ms/step - Loss_inf: 0.0231 - Loss_smth: 0.0079
Valid Loss RMSE: 9.7771
```

# Parallel Computing
The parallel prefix sum algorithm accelerates the training process by at least **1.6 times** and the inference process by at least **1.8 times**. The parallel algorithm reduces the time complexity of model inference from _**O(N)**_ to _**O(logN)**_ for serial calculations. As the estimation length increases, the acceleration effect becomes more obvious.
<p align="center">
  <img src="https://github.com/XinyuanLiao/complexNDM/blob/main/Figs/para.jpg" width="500px"/>
</p>

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
