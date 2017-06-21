import os

import tensorflow as tf

from keras import backend as K
from keras.models import Model
from keras.layers import Input, merge
from keras.layers.core import Lambda

# Before running the integration tests, users are expected to set these
# environment variables.
IS_GPU = (os.environ['MXNET_KERAS_TEST_MACHINE'] == 'GPU')
GPU_NUM = int(os.environ['GPU_NUM']) if IS_GPU else 0
KERAS_BACKEND = os.environ['KERAS_BACKEND']

def slice_batch(x, n_gpus, part):
    sh = K.shape(x)
    L = sh[0] / n_gpus
    if part == n_gpus - 1:
        return x[part*L:]
    return x[part*L:(part+1)*L]

def prepare_tensorflow_multi_gpu_model(model, kwargs):
    multi_input = True if type(model.input_shape) is list else False
    multi_output = True if len(model.outputs) > 1 else False
    x = [Input(shape[1:]) for shape in model.input_shape] if multi_input else Input(model.input_shape[1:])
    towers = []
    outputs = []
    for _ in range(len(model.outputs)):
        outputs.append([])
    for g in range(GPU_NUM):
        with tf.device('/gpu:' + str(g)):
            slice_g = [Lambda(slice_batch, lambda shape: shape, arguments={'n_gpus':GPU_NUM, 'part':g})(y) for y in x] \
                      if multi_input \
                      else Lambda(slice_batch, lambda shape: shape, arguments={'n_gpus':GPU_NUM, 'part':g})(x)
            output_model = model(slice_g)
            if multi_output:
                for num in range(len(output_model)):
                    outputs[num].append(output_model[num])
            else:
                 towers.append(output_model)

    with tf.device('/cpu:0'):
        merged = []
        if multi_output:
            merged = []
            for output in outputs:
                merged.append(merge(output, mode='concat', concat_axis=0))
        else:
            merged = merge(towers, mode='concat', concat_axis=0)

    model = Model(input= x if type(x) is list else [x], output=merged)
    model.compile(**kwargs)
    return model

def prepare_mxnet_multi_gpu_model(model, kwargs):
    gpu_list = []
    for i in range(GPU_NUM):
        gpu_list.append('gpu(%d)' % i)
    kwargs['context'] = gpu_list
    model.compile(**kwargs)
    return model

def prepare_gpu_model(model, **kwargs):
    if KERAS_BACKEND == 'mxnet':
        return(prepare_mxnet_multi_gpu_model(model, kwargs))
    elif KERAS_BACKEND == 'tensorflow' and GPU_NUM > 1:
        return(prepare_tensorflow_multi_gpu_model(model, kwargs))
    else:
        model.compile(**kwargs)
        return model

def prepare_cpu_model(model, **kwargs):
    model.compile(**kwargs)
    return model

def make_model(model, **kwargs):
    """
        Compiles the Keras Model object for given backend type and machine type.
        Use this function to write one Keras code and run it across different machine type.

        If environment variable - MXNET_KERAS_TEST_MACHINE is set to CPU, then Compiles
        Keras Model for running on CPU.

        If environment variable - MXNET_KERAS_TEST_MACHINE is set to GPU, then Compiles
        Keras Model running on GPU using number of GPUs equal to number specified in
        GPU_NUM environment variable.

        Currently supports only MXNet as Keras backend.
    """
    if(IS_GPU):
        model = prepare_gpu_model(model, **kwargs)
        return model
    else:
        model = prepare_cpu_model(model, **kwargs)
        return model
