# RAdam in Tensorflow
On The Variance Of The Adaptive Learning Rate And Beyond in Tensorflow

This repo is based on pytorch impl [repo](https://github.com/LiyuanLucasLiu/RAdam)

*WIP*

# Explanation
The learning rate warmup for Adam is a must-have trick for stable training in certain situations (or eps tuning). But the underlying mechanism is largely unknown. In our study, we suggest one fundamental cause is **the large variance of the adaptive learning rates**, and provide both theoretical and empirical support evidence.

In addition to explaining **why we should use warmup**, we also propose **RAdam**, a theoretically sound variant of Adam.

# Requirement
* Python 3.x
* Tensorflow 1.x (maybe 2.x)

## Usage

```python
# learning can be either a scalar or a tensor

# use exclude_from_weight_decay feature, 
# if you wanna selectively disable updating weight-decayed weights

optimizer = RAdamOptimizer(
    ...
)
```

You can simply test the optimizers on MNIST Dataset w/ below model!

For `RAdam` optimizer,
```python
python3 mnist_test --optimizer "radam"
```

## Results

Testing Accuracy & Loss among the optimizers on the several data sets w/ under same condition.

### MNIST DataSet

![acc](./assets/mnist_acc.png)

*Optimizer* | *Test Acc* | *Time* | *Etc* |
:---: | :---: | :---: | :---: |
RAdam | **xx.xx%** | m s | |
AdaBound | **97.77%** | 5m 45s |  |
AMSBound | 97.72% | 5m 52s |  |
Adam | 97.62% | 4m 18s |  |
AdaGrad | 90.15% | **4m 07s** |  |
SGD | 87.88% | 5m 26s | |
Momentum | 87.88% | 4m 26s | w/ nestrov |

# Citation

```
@article{liu2019radam,
  title={On the Variance of the Adaptive Learning Rate and Beyond},
  author={Liu, Liyuan and Jiang, Haoming and He, Pengcheng and Chen, Weizhu and Liu, Xiaodong and Gao, Jianfeng and Han, Jiawei},
  journal={arXiv preprint arXiv:1908.03265},
  year={2019}
}
```

# Author

Hyeongchan Kim / [kozistr](http://kozistr.tech)