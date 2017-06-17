# Profiling results

## Resnet50

### Setup

`Dataset` - [CIFAR10 small image dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
`Machine` - [AWS EC2 P2.16xlarge](https://aws.amazon.com/ec2/instance-types/p2/)

`Batchsize` - 64
`Epochs` - 10

`Note:` Parameters are not tuned to get the best possible accuracy. We run these tests for few epochs and take the average time per epoch to calculate the number of images processed per second.

### Keras 1.2.2 with MXNet backend

|                               | GPU (1) | GPU (2) | GPU (4) | GPU(8) | GPU(16) |
|-------------------------------|:-------:|:-------:|:-------:|:------:|:-------:|
|  Training time (secs)         |         |         |         |        |         |
| Average time per epoch (secs) |         |         |         |        |         |
| Images processed per sec      |         |         |         |        |         |
| Training accuracy             |         |         |         |        |         |
| Test accuracy                 |         |         |         |        |         |
| Maximum Memory Consumption    |         |         |         |        |         |


## Inception_v3

### Setup

`Dataset` - [CIFAR10 small image dataset resized to 299x299](https://www.cs.toronto.edu/~kriz/cifar.html)
`Machine` - [AWS EC2 P2.8xlarge](https://aws.amazon.com/ec2/instance-types/p2/)

`Batchsize` - 32
`Epochs` - 5

### Keras 1.2.2 with MXNet backend

|                               | GPU (1) | GPU (2) | GPU (4) | GPU(8) |
|-------------------------------|:-------:|:-------:|:-------:|:------:|
|  Training time (secs)         |10738    |5935     |3451     |2839    |
| Average time per epoch (secs) |2147     |1187     |690      |568     |
| Training accuracy             |0.78     |0.80     |0.81     |0.81    |
| Test accuracy                 |0.73     |0.75     |0.80     |0.78    |
| Maximum Memory Consumption    |10319    |9542     |10168    |12183   |
