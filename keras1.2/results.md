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
