# Profiling results

## Resnet50

### Setup

* `Dataset` - [CIFAR10 small image dataset](https://www.cs.toronto.edu/~kriz/cifar.html).
* `Machine` - [AWS EC2 P2.16xlarge](https://aws.amazon.com/ec2/instance-types/p2/).
* `Batchsize` - 32 per GPU.

Example:
1 GPU   -> Batchsize = 32
2 GPU   -> Batchsize = 64
4 GPU   -> Batchsize = 128
8 GPU   -> Batchsize = 256
16 GPU  -> Batchsize = 512

MXNet distributes the data batch across all GPUs. i.e., if batch size is 512 and number of GPUs is 16, then each GPU get 512 / 16 images.

* `Epochs` - 10

`Note:` Parameters are not tuned to get the best possible accuracy. We run these tests for few epochs and take the average time per epoch to calculate the number of images processed per second.

### Keras 1.2.2 with MXNet backend

|                               | GPU (1) | GPU (2) | GPU (4) | GPU (8) | GPU (16) |
|-------------------------------|:-------:|:-------:|:-------:|:-------:|:--------:|
| Training time (secs)          | 1609.32 | 971.94  | 514.96  | 418.09  | 413.66   |
| Average time per epoch (secs) | 160.93  | 97.19   | 51.49   | 41.80   | 41.36    |
| Images processed per sec      | 372.82  | 617.32  | 1165.12 | 1435.06 | 1450.43  |
| Training accuracy             | 0.69    | 0.69    | 0.69    | 0.68    | 0.54     |
| Test accuracy                 | 0.67    | 0.52    | 0.60    | 0.59    | 0.42     |
| Maximum Memory Consumption    | 5086    | 5048    | 5012    | 9786    | 15187    |

### Reduce Randomization
* `random seed` - 1337
* `Epochs` - 100

#### MXNet Backend
|                               | GPU (1) | GPU (2) | GPU (4) | GPU (8) | GPU (16) |
|-------------------------------|:-------:|:-------:|:-------:|:-------:|:--------:|
| Training time (secs)          | 15860.5 | 9456.02 | 4475.55 | 3429.55 | 3040.19  |
| Average time per epoch (secs) | 158.61  | 94.56   | 44.76   | 34.30   | 304.02   |
| Images processed per sec      | 378.30  | 634.52  | 1340.62 | 1749.50 | 1973.56  |
| Training accuracy             | 0.95    | 0.95    | 0.95    | 0.95    | 0.94     |
| Test accuracy                 | 0.80    | 0.75    | 0.78    | 0.75    | 0.77     |
| Maximum Memory Consumption    | 5091    | 5855    | 5126    | 9826    | 15267    |

#### Tensorflow Backend
|                               | GPU (1) | GPU (2) | GPU (4) | GPU (8) | GPU (16) |
|-------------------------------|:-------:|:-------:|:-------:|:-------:|:--------:|
| Training time (secs)          | 23749.9 | 18029.7 | 12230.6 | 10911.9 | 10907.5  |
| Average time per epoch (secs) | 237.50  | 180.30  | 122.31  | 109.12  | 109.08   |
| Images processed per sec      | 252.63  | 332.78  | 490.57  | 549.86  | 550.08   |
| Training accuracy             | 0.95    | 0.94    | 0.90    | 0.84    | 0.77     |
| Test accuracy                 | 0.80    | 0.76    | 0.76    | 0.75    | 0.73     |
| Maximum Memory Consumption    | 10946   | 21890   | 43778   | 87544   | 175088   |



## Inception_v3

### Setup

* `Dataset` - [CIFAR10 small image dataset resized to 299x299](https://www.cs.toronto.edu/~kriz/cifar.html)
* `Machine` - [AWS EC2 P2.8xlarge](https://aws.amazon.com/ec2/instance-types/p2/)
* `Batchsize` - 32
* `Epochs` - 5

### Keras 1.2.2 with MXNet backend

|                               | GPU (1) | GPU (2) | GPU (4) | GPU(8) |
|-------------------------------|:-------:|:-------:|:-------:|:------:|
| Training time (secs)          |10738    |5935     |3451     |2839    |
| Average time per epoch (secs) |2147     |1187     |690      |568     |
| Training accuracy             |0.78     |0.80     |0.81     |0.81    |
| Test accuracy                 |0.73     |0.75     |0.80     |0.78    |
| Maximum Memory Consumption    |10319    |9542     |10168    |12183   |

### Reduce Randomization

* `Epochs` - 25

#### MXNet Backend
|                               | GPU (1) | GPU (2) | GPU (4) | GPU(8) | GPU(16) |
|-------------------------------|:-------:|:-------:|:-------:|:------:|:-------:|
| Training time (secs)          |54609    |29342    |17316    |14170   |12971    |
| Average time per epoch (secs) |2184     |1174     |693      |567     |519      |
| Training accuracy             |0.99     |0.99     |0.99     |0.99    |0.99     |
| Test accuracy                 |0.83     |0.86     |0.85     |0.87    |0.82     |
| Maximum Memory Consumption    |10896    |10961    |11846    |13701   |17628    |

#### Tensorflow Backend
|                               | GPU (1) | GPU (2) | GPU (4) | GPU(8) | GPU(16) |
|-------------------------------|:-------:|:-------:|:-------:|:------:|:-------:|
| Training time (secs)          |121480   |72657    |49758    |44042   |78247    |
| Average time per epoch (secs) |4859     |2906     |1990     |1761    |3130     |
| Training accuracy             |0.97     |0.97     |0.96     |0.96    |0.96     |
| Test accuracy                 |0.71     |0.81     |0.83     |0.71    |0.79     |
| Maximum Memory Consumption    |10946    |21890    |43778    |87544   |175088   |