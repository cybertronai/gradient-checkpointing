"""
The core Pixel-CNN model
"""

import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import arg_scope
import pixel_cnn_pp.nn as nn

def model_spec(x, h=None, init=False, ema=None, dropout_p=0.5, nr_resnet=5, nr_filters=160, nr_logistic_mix=10, resnet_nonlinearity='concat_elu', energy_distance=False):
    """
    We receive a Tensor x of shape (N,H,W,D1) (e.g. (12,32,32,3)) and produce
    a Tensor x_out of shape (N,H,W,D2) (e.g. (12,32,32,100)), where each fiber
    of the x_out tensor describes the predictive distribution for the RGB at
    that position.
    'h' is an optional N x K matrix of values to condition our generative model on
    """

    counters = {}
    with arg_scope([nn.conv2d, nn.deconv2d, nn.gated_resnet, nn.dense], counters=counters, init=init, ema=ema, dropout_p=dropout_p):

        # parse resnet nonlinearity argument
        if resnet_nonlinearity == 'concat_elu':
            resnet_nonlinearity = nn.concat_elu
        elif resnet_nonlinearity == 'elu':
            resnet_nonlinearity = tf.nn.elu
        elif resnet_nonlinearity == 'relu':
            resnet_nonlinearity = tf.nn.relu
        else:
            raise('resnet nonlinearity ' + resnet_nonlinearity + ' is not supported')

        with arg_scope([nn.gated_resnet], nonlinearity=resnet_nonlinearity, h=h):

            # ////////// up pass through pixelCNN ////////
            xs = nn.int_shape(x)
            x_pad = tf.concat([x,tf.ones(xs[:-1]+[1])],3) # add channel of ones to distinguish image from padding later on
            u_list = [nn.down_shift(nn.down_shifted_conv2d(x_pad, num_filters=nr_filters, filter_size=[2, 3]))] # stream for pixels above
            ul_list = [nn.down_shift(nn.down_shifted_conv2d(x_pad, num_filters=nr_filters, filter_size=[1,3])) + \
                       nn.right_shift(nn.down_right_shifted_conv2d(x_pad, num_filters=nr_filters, filter_size=[2,1]))] # stream for up and to the left

            for rep in range(nr_resnet):
                u_list.append(nn.gated_resnet(u_list[-1], conv=nn.down_shifted_conv2d))
                ul_list.append(nn.gated_resnet(ul_list[-1], u_list[-1], conv=nn.down_right_shifted_conv2d))

            u_list.append(nn.down_shifted_conv2d(u_list[-1], num_filters=nr_filters, stride=[2, 2]))
            ul_list.append(nn.down_right_shifted_conv2d(ul_list[-1], num_filters=nr_filters, stride=[2, 2]))

            for rep in range(nr_resnet):
                u_list.append(nn.gated_resnet(u_list[-1], conv=nn.down_shifted_conv2d))
                ul_list.append(nn.gated_resnet(ul_list[-1], u_list[-1], conv=nn.down_right_shifted_conv2d))

            u_list.append(nn.down_shifted_conv2d(u_list[-1], num_filters=nr_filters, stride=[2, 2]))
            ul_list.append(nn.down_right_shifted_conv2d(ul_list[-1], num_filters=nr_filters, stride=[2, 2]))

            for rep in range(nr_resnet):
                u_list.append(nn.gated_resnet(u_list[-1], conv=nn.down_shifted_conv2d))
                ul_list.append(nn.gated_resnet(ul_list[-1], u_list[-1], conv=nn.down_right_shifted_conv2d))

            # remember nodes
            for t in u_list+ul_list:
                tf.add_to_collection('checkpoints', t)

            # /////// down pass ////////
            u = u_list.pop()
            ul = ul_list.pop()
            for rep in range(nr_resnet):
                u = nn.gated_resnet(u, u_list.pop(), conv=nn.down_shifted_conv2d)
                ul = nn.gated_resnet(ul, tf.concat([u, ul_list.pop()],3), conv=nn.down_right_shifted_conv2d)
                tf.add_to_collection('checkpoints', u)
                tf.add_to_collection('checkpoints', ul)

            u = nn.down_shifted_deconv2d(u, num_filters=nr_filters, stride=[2, 2])
            ul = nn.down_right_shifted_deconv2d(ul, num_filters=nr_filters, stride=[2, 2])

            for rep in range(nr_resnet+1):
                u = nn.gated_resnet(u, u_list.pop(), conv=nn.down_shifted_conv2d)
                ul = nn.gated_resnet(ul, tf.concat([u, ul_list.pop()],3), conv=nn.down_right_shifted_conv2d)
                tf.add_to_collection('checkpoints', u)
                tf.add_to_collection('checkpoints', ul)

            u = nn.down_shifted_deconv2d(u, num_filters=nr_filters, stride=[2, 2])
            ul = nn.down_right_shifted_deconv2d(ul, num_filters=nr_filters, stride=[2, 2])

            for rep in range(nr_resnet+1):
                u = nn.gated_resnet(u, u_list.pop(), conv=nn.down_shifted_conv2d)
                ul = nn.gated_resnet(ul, tf.concat([u, ul_list.pop()],3), conv=nn.down_right_shifted_conv2d)
                tf.add_to_collection('checkpoints', u)
                tf.add_to_collection('checkpoints', ul)

            if energy_distance:
                f = nn.nin(tf.nn.elu(ul), 64)

                # generate 10 samples
                fs = []
                for rep in range(10):
                    fs.append(f)
                f = tf.concat(fs, 0)
                fs = nn.int_shape(f)
                f += nn.nin(tf.random_uniform(shape=fs[:-1] + [4], minval=-1., maxval=1.), 64)
                f = nn.nin(nn.concat_elu(f), 64)
                x_sample = tf.tanh(nn.nin(nn.concat_elu(f), 3, init_scale=0.1))

                x_sample = tf.split(x_sample, 10, 0)

                assert len(u_list) == 0
                assert len(ul_list) == 0

                return x_sample

            else:
                x_out = nn.nin(tf.nn.elu(ul),10*nr_logistic_mix)

                assert len(u_list) == 0
                assert len(ul_list) == 0

                return x_out

