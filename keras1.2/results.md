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
