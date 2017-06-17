from __future__ import division

import six
from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    merge,
    Dense,
    Flatten
)
from keras.layers.convolutional import (
    Convolution2D,
    MaxPooling2D,
    AveragePooling2D,
    ZeroPadding2D
)
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K
from keras.models import Sequential

bn_axis = 1 if K.backend() == 'mxnet' else -1

def conv_factory(nb_filter, nb_row=1, nb_col=1, stride=(1,1), pad=(0, 0), border_mode='valid'):
    conv = Convolution2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col, subsample=stride,
                         border_mode=border_mode)
    def func(input):
        input = ZeroPadding2D(pad)(input)
        bn = BatchNormalization(axis=bn_axis)(conv(input))
        return Activation("relu")(bn)
    return func

def inception7a(data,
                num_1x1,
                num_3x3_red, num_3x3_1, num_3x3_2,
                num_5x5_red, num_5x5,
                pool, proj):
    tower_1x1 = conv_factory(num_1x1)(data)
    tower_5x5 = conv_factory(num_5x5_red)(tower_1x1)
    tower_5x5 = conv_factory(num_5x5, 5, 5, pad=(2,2))(tower_5x5)
    tower_3x3 = conv_factory(num_3x3_red)(data)
    tower_3x3 = conv_factory(num_3x3_1, 3, 3, pad=(1,1))(tower_3x3)
    tower_3x3 = conv_factory(num_3x3_2, 3, 3, pad=(1,1))(tower_3x3)
    pooling = MaxPooling2D(pool_size=(3,3), strides=(1,1))(ZeroPadding2D((1,1))(data)) \
              if pool == 'max' else \
              AveragePooling2D(pool_size=(3,3), strides=(1,1))(ZeroPadding2D((1,1))(data))
    cproj = conv_factory(proj)(pooling)
    concat = merge([tower_1x1, tower_5x5, tower_3x3, cproj], mode='concat', concat_axis=1)
    return concat

# First Downsample
def inception7b(data,
                num_3x3,
                num_d3x3_red, num_d3x3_1, num_d3x3_2,
                pool):
    tower_3x3 = conv_factory(num_3x3, 3, 3, stride=(2,2), pad=(0,0))(data)
    tower_d3x3 = conv_factory(num_d3x3_red)(data)
    tower_d3x3 = conv_factory(num_d3x3_1, 3, 3, pad=(1,1))(tower_d3x3)
    tower_d3x3 = conv_factory(num_d3x3_2, 3, 3, stride=(2,2))(tower_d3x3)
    pooling = MaxPooling2D(pool_size=(3,3), strides=(2,2))(data) \
              if pool == 'max' else \
              AveragePooling2D(pool_size=(3, 3), strides=(2, 2))(data)
    concat = merge([tower_3x3, tower_d3x3, pooling], mode='concat', concat_axis=1)
    return concat

def inception7c(data,
                num_1x1,
                num_d7_red, num_d7_1, num_d7_2,
                num_q7_red, num_q7_1, num_q7_2, num_q7_3, num_q7_4,
                pool, proj):
    tower_1x1 = conv_factory(num_1x1)(data)
    tower_d7 = conv_factory(num_d7_red)(data)
    tower_d7 = conv_factory(num_d7_1, 1, 7, pad=(0,3))(tower_d7)
    tower_d7 = conv_factory(num_d7_2, 7, 1, pad=(3,0))(tower_d7)
    tower_q7 = conv_factory(num_q7_red)(data)
    tower_q7 = conv_factory(num_q7_1, 7, 1, pad=(3,0))(tower_q7)
    tower_q7 = conv_factory(num_q7_2, 1, 7, pad=(0,3))(tower_q7)
    tower_q7 = conv_factory(num_q7_3, 7, 1, pad=(3,0))(tower_q7)
    tower_q7 = conv_factory(num_q7_4, 1, 7, pad=(0,3))(tower_q7)
    pooling = MaxPooling2D(pool_size=(3,3), strides=(1,1))(ZeroPadding2D((1,1))(data)) \
              if pool == 'max' else \
              AveragePooling2D(pool_size=(3, 3), strides=(1,1))(ZeroPadding2D((1,1))(data))
    cproj = conv_factory(proj)(pooling)
    concat = merge([tower_1x1, tower_d7, tower_q7, cproj], mode='concat', concat_axis=1)
    return concat

def inception7d(data,
                num_3x3_red, num_3x3,
                num_d7_3x3_red, num_d7_1, num_d7_2, num_d7_3x3,
                pool):
    tower_3x3 = conv_factory(num_3x3_red)(data)
    tower_3x3 = conv_factory(num_3x3, 3, 3, stride=(2,2))(tower_3x3)
    tower_d7_3x3 = conv_factory(num_d7_3x3_red)(data)
    tower_d7_3x3 = conv_factory(num_d7_1, 1, 7, pad=(0,3))(tower_d7_3x3)
    tower_d7_3x3 = conv_factory(num_d7_2, 7, 1, pad=(3,0))(tower_d7_3x3)
    tower_d7_3x3 = conv_factory(num_d7_3x3, 3, 3, stride=(2,2))(tower_d7_3x3)
    pooling = MaxPooling2D(pool_size=(3,3), strides=(2,2))(data) \
              if pool == 'max' else \
              AveragePooling2D(pool_size=(3, 3), strides=(2, 2))(data)
    concat = merge([tower_3x3, tower_d7_3x3, pooling], mode='concat', concat_axis=1)
    return concat

def inception7e(data,
                num_1x1,
                num_d3_red, num_d3_1, num_d3_2,
                num_3x3_d3_red, num_3x3, num_3x3_d3_1, num_3x3_d3_2,
                pool, proj):
    tower_1x1 = conv_factory(num_1x1)(data)
    tower_d3 = conv_factory(num_d3_red)(data)
    tower_d3_a = conv_factory(num_d3_1, 1, 3, pad=(0,1))(tower_d3)
    tower_d3_b = conv_factory(num_d3_2, 3, 1, pad=(1,0))(tower_d3)
    tower_3x3_d3 = conv_factory(num_3x3_d3_red)(data)
    tower_3x3_d3 = conv_factory(num_3x3, 3, 3, pad=(1,1))(tower_3x3_d3)
    tower_3x3_d3_a = conv_factory(num_3x3_d3_1, 1, 3, pad=(0,1))(tower_3x3_d3)
    tower_3x3_d3_b = conv_factory(num_3x3_d3_2, 3, 1, pad=(1,0))(tower_3x3_d3)
    pooling = MaxPooling2D(pool_size=(3,3), strides=(1,1))(ZeroPadding2D((1,1))(data)) \
              if pool == 'max' else \
              AveragePooling2D(pool_size=(3, 3), strides=(1, 1))(ZeroPadding2D((1,1))(data))
    cproj = conv_factory(proj)(pooling)
    concat = merge([tower_1x1, tower_d3_a, tower_d3_b, tower_3x3_d3_a, tower_3x3_d3_b, cproj],
                   mode='concat', concat_axis=1)
    return concat

def make_symbol(input_shape, num_classes=1000):
    K.set_image_dim_ordering('th')
    data = Input(input_shape)
    #stage 1
    conv = conv_factory(32, 3, 3, stride=(2,2))(data)
    conv_1 = conv_factory(32, 3, 3)(conv)
    conv_2 = conv_factory(64, 3, 3, pad=(1,1))(conv_1)
    pool1 = MaxPooling2D(pool_size=(3,3), strides=(2,2))(conv_2)
    #stage 2
    conv_3 = conv_factory(80)(pool1)
    conv_4 = conv_factory(192, 3, 3)(conv_3)
    pool2 = MaxPooling2D(pool_size=(3,3), strides=(2,2))(conv_4)
    #stage 3
    in3a = inception7a(pool2, 64,
                       64, 96, 96,
                       48, 64,
                       "avg", 32)
    in3b = inception7a(in3a, 64,
                       64, 96, 96,
                       48, 64,
                       "avg", 64)
    in3c = inception7a(in3b, 64,
                       64, 96, 96,
                       48, 64,
                       "avg", 64)
    in3d = inception7b(in3c, 384,
                       64, 96, 96,
                       "max")
    # stage 4
    in4a = inception7c(in3d, 192,
                       128, 128, 192,
                       128, 128, 128, 128, 192,
                       "avg", 192)
    in4b = inception7c(in4a, 192,
                       160, 160, 192,
                       160, 160, 160, 160, 192,
                       "avg", 192)
    in4c = inception7c(in4b, 192,
                       160, 160, 192,
                       160, 160, 160, 160, 192,
                       "avg", 192)
    in4d = inception7c(in4c, 192,
                       192, 192, 192,
                       192, 192, 192, 192, 192,
                       "avg", 192)
    in4e = inception7d(in4d, 192, 320,
                       192, 192, 192, 192,
                       "max")
    # stage 5
    in5a = inception7e(in4e, 320,
                       384, 384, 384,
                       448, 384, 384, 384,
                       "avg", 192)
    in5b = inception7e(in5a, 320,
                       384, 384, 384,
                       448, 384, 384, 384,
                       "max", 192)
    # pool
    pool = AveragePooling2D(pool_size=(8,8), strides=(1,1))(in5b)
    flatten = Flatten()(pool)
    out = Dense(output_dim=num_classes, activation='softmax')(flatten)
    model = Model(input=data, output=out)
    return model
