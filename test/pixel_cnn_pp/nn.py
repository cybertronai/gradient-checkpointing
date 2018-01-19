"""
Various tensorflow utilities
"""

import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import add_arg_scope

def int_shape(x):
    return list(map(int, x.get_shape()))

def concat_elu(x):
    """ like concatenated ReLU (http://arxiv.org/abs/1603.05201), but then with ELU """
    axis = len(x.get_shape())-1
    return tf.nn.elu(tf.concat([x, -x], axis))

def log_sum_exp(x):
    """ numerically stable log_sum_exp implementation that prevents overflow """
    axis = len(x.get_shape())-1
    m = tf.reduce_max(x, axis)
    m2 = tf.reduce_max(x, axis, keepdims=True)
    return m + tf.log(tf.reduce_sum(tf.exp(x-m2), axis))

def log_prob_from_logits(x):
    """ numerically stable log_softmax implementation that prevents overflow """
    axis = len(x.get_shape())-1
    m = tf.reduce_max(x, axis, keepdims=True)
    return x - m - tf.log(tf.reduce_sum(tf.exp(x-m), axis, keepdims=True))

def energy_distance(x, x_sample):
    l1 = 0.
    for xs in x_sample:
        l1 += tf.reduce_sum(tf.pow(1e-10 + tf.reduce_sum(tf.square(127.5*(x-xs)),3), 0.75))
    l1 /= len(x_sample)

    l2 = 0.
    n = 0
    for i in range(len(x_sample)):
        for j in range(i+1,len(x_sample)):
            l2 += tf.reduce_sum(tf.pow(1e-10 + tf.reduce_sum(tf.square(127.5*(x_sample[i] - x_sample[j])), 3), 0.75))
            n += 1
    l2 /= n

    return 2.*l1 - l2

def discretized_mix_logistic_loss(x,l,sum_all=True):
    """ log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval """
    xs = int_shape(x) # true image (i.e. labels) to regress to, e.g. (B,32,32,3)
    ls = int_shape(l) # predicted distribution, e.g. (B,32,32,100)
    nr_mix = int(ls[-1] / 10) # here and below: unpacking the params of the mixture of logistics
    logit_probs = l[:,:,:,:nr_mix]
    l = tf.reshape(l[:,:,:,nr_mix:], xs + [nr_mix*3])
    means = l[:,:,:,:,:nr_mix]
    log_scales = tf.maximum(l[:,:,:,:,nr_mix:2*nr_mix], -7.)
    coeffs = tf.nn.tanh(l[:,:,:,:,2*nr_mix:3*nr_mix])
    x = tf.reshape(x, xs + [1]) + tf.zeros(xs + [nr_mix]) # here and below: getting the means and adjusting them based on preceding sub-pixels
    m2 = tf.reshape(means[:,:,:,1,:] + coeffs[:, :, :, 0, :] * x[:, :, :, 0, :], [xs[0],xs[1],xs[2],1,nr_mix])
    m3 = tf.reshape(means[:, :, :, 2, :] + coeffs[:, :, :, 1, :] * x[:, :, :, 0, :] + coeffs[:, :, :, 2, :] * x[:, :, :, 1, :], [xs[0],xs[1],xs[2],1,nr_mix])
    means = tf.concat([tf.reshape(means[:,:,:,0,:], [xs[0],xs[1],xs[2],1,nr_mix]), m2, m3],3)
    centered_x = x - means
    inv_stdv = tf.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1./255.)
    cdf_plus = tf.nn.sigmoid(plus_in)
    min_in = inv_stdv * (centered_x - 1./255.)
    cdf_min = tf.nn.sigmoid(min_in)
    log_cdf_plus = plus_in - tf.nn.softplus(plus_in) # log probability for edge case of 0 (before scaling)
    log_one_minus_cdf_min = -tf.nn.softplus(min_in) # log probability for edge case of 255 (before scaling)
    cdf_delta = cdf_plus - cdf_min # probability for all other cases
    mid_in = inv_stdv * centered_x
    log_pdf_mid = mid_in - log_scales - 2.*tf.nn.softplus(mid_in) # log probability in the center of the bin, to be used in extreme cases (not actually used in our code)

    # now select the right output: left edge case, right edge case, normal case, extremely low prob case (doesn't actually happen for us)

    # this is what we are really doing, but using the robust version below for extreme cases in other applications and to avoid NaN issue with tf.select()
    # log_probs = tf.select(x < -0.999, log_cdf_plus, tf.select(x > 0.999, log_one_minus_cdf_min, tf.log(cdf_delta)))

    # robust version, that still works if probabilities are below 1e-5 (which never happens in our code)
    # tensorflow backpropagates through tf.select() by multiplying with zero instead of selecting: this requires use to use some ugly tricks to avoid potential NaNs
    # the 1e-12 in tf.maximum(cdf_delta, 1e-12) is never actually used as output, it's purely there to get around the tf.select() gradient issue
    # if the probability on a sub-pixel is below 1e-5, we use an approximation based on the assumption that the log-density is constant in the bin of the observed sub-pixel value
    log_probs = tf.where(x < -0.999, log_cdf_plus, tf.where(x > 0.999, log_one_minus_cdf_min, tf.where(cdf_delta > 1e-5, tf.log(tf.maximum(cdf_delta, 1e-12)), log_pdf_mid - np.log(127.5))))

    log_probs = tf.reduce_sum(log_probs,3) + log_prob_from_logits(logit_probs)
    if sum_all:
        return -tf.reduce_sum(log_sum_exp(log_probs))
    else:
        return -tf.reduce_sum(log_sum_exp(log_probs),[1,2])

def sample_from_discretized_mix_logistic(l,nr_mix):
    ls = int_shape(l)
    xs = ls[:-1] + [3]
    # unpack parameters
    logit_probs = l[:, :, :, :nr_mix]
    l = tf.reshape(l[:, :, :, nr_mix:], xs + [nr_mix*3])
    # sample mixture indicator from softmax
    sel = tf.one_hot(tf.argmax(logit_probs - tf.log(-tf.log(tf.random_uniform(logit_probs.get_shape(), minval=1e-5, maxval=1. - 1e-5))), 3), depth=nr_mix, dtype=tf.float32)
    sel = tf.reshape(sel, xs[:-1] + [1,nr_mix])
    # select logistic parameters
    means = tf.reduce_sum(l[:,:,:,:,:nr_mix]*sel,4)
    log_scales = tf.maximum(tf.reduce_sum(l[:,:,:,:,nr_mix:2*nr_mix]*sel,4), -7.)
    coeffs = tf.reduce_sum(tf.nn.tanh(l[:,:,:,:,2*nr_mix:3*nr_mix])*sel,4)
    # sample from logistic & clip to interval
    # we don't actually round to the nearest 8bit value when sampling
    u = tf.random_uniform(means.get_shape(), minval=1e-5, maxval=1. - 1e-5)
    x = means + tf.exp(log_scales)*(tf.log(u) - tf.log(1. - u))
    x0 = tf.minimum(tf.maximum(x[:,:,:,0], -1.), 1.)
    x1 = tf.minimum(tf.maximum(x[:,:,:,1] + coeffs[:,:,:,0]*x0, -1.), 1.)
    x2 = tf.minimum(tf.maximum(x[:,:,:,2] + coeffs[:,:,:,1]*x0 + coeffs[:,:,:,2]*x1, -1.), 1.)
    return tf.concat([tf.reshape(x0,xs[:-1]+[1]), tf.reshape(x1,xs[:-1]+[1]), tf.reshape(x2,xs[:-1]+[1])],3)

def get_var_maybe_avg(var_name, ema, **kwargs):
    ''' utility for retrieving polyak averaged params '''
    v = tf.get_variable(var_name, **kwargs)
    if ema is not None:
        v = ema.average(v)
    return v

def get_vars_maybe_avg(var_names, ema, **kwargs):
    ''' utility for retrieving polyak averaged params '''
    vars = []
    for vn in var_names:
        vars.append(get_var_maybe_avg(vn, ema, **kwargs))
    return vars

def adam_updates(params, cost_or_grads, lr=0.001, mom1=0.9, mom2=0.999):
    ''' Adam optimizer '''
    updates = []
    if type(cost_or_grads) is not list:
        grads = tf.gradients(cost_or_grads, params)
    else:
        grads = cost_or_grads
    t = tf.Variable(1., 'adam_t')
    for p, g in zip(params, grads):
        mg = tf.Variable(tf.zeros(p.get_shape()), p.name + '_adam_mg')
        if mom1>0:
            v = tf.Variable(tf.zeros(p.get_shape()), p.name + '_adam_v')
            v_t = mom1*v + (1. - mom1)*g
            v_hat = v_t / (1. - tf.pow(mom1,t))
            updates.append(v.assign(v_t))
        else:
            v_hat = g
        mg_t = mom2*mg + (1. - mom2)*tf.square(g)
        mg_hat = mg_t / (1. - tf.pow(mom2,t))
        g_t = v_hat / tf.sqrt(mg_hat + 1e-8)
        p_t = p - lr * g_t
        updates.append(mg.assign(mg_t))
        updates.append(p.assign(p_t))
    updates.append(t.assign_add(1))
    return tf.group(*updates)

def get_name(layer_name, counters):
    ''' utlity for keeping track of layer names '''
    if not layer_name in counters:
        counters[layer_name] = 0
    name = layer_name + '_' + str(counters[layer_name])
    counters[layer_name] += 1
    return name

@add_arg_scope
def dense(x, num_units, nonlinearity=None, init_scale=1., counters={}, init=False, ema=None, **kwargs):
    ''' fully connected layer '''
    name = get_name('dense', counters)
    with tf.variable_scope(name):
        V = get_var_maybe_avg('V', ema, shape=[int(x.get_shape()[1]),num_units], dtype=tf.float32,
                              initializer=tf.random_normal_initializer(0, 0.05), trainable=True)
        g = get_var_maybe_avg('g', ema, shape=[num_units], dtype=tf.float32,
                              initializer=tf.constant_initializer(1.), trainable=True)
        b = get_var_maybe_avg('b', ema, shape=[num_units], dtype=tf.float32,
                              initializer=tf.constant_initializer(0.), trainable=True)

        # use weight normalization (Salimans & Kingma, 2016)
        x = tf.matmul(x, V)
        scaler = g / tf.sqrt(tf.reduce_sum(tf.square(V), [0]))
        x = tf.reshape(scaler, [1, num_units]) * x + tf.reshape(b, [1, num_units])

        if init: # normalize x
            m_init, v_init = tf.nn.moments(x, [0])
            scale_init = init_scale/tf.sqrt(v_init + 1e-10)
            with tf.control_dependencies([g.assign(g*scale_init), b.assign_add(-m_init*scale_init)]):
                x = tf.nn.l2_normalize(x, axis=0)

        # apply nonlinearity
        if nonlinearity is not None:
            x = nonlinearity(x)

        return x

@add_arg_scope
def conv2d(x, num_filters, filter_size=[3,3], stride=[1,1], pad='SAME', nonlinearity=None, init_scale=1., counters={}, init=False, ema=None, **kwargs):
    ''' convolutional layer '''
    name = get_name('conv2d', counters)
    with tf.variable_scope(name):
        V = get_var_maybe_avg('V', ema, shape=filter_size+[int(x.get_shape()[-1]),num_filters], dtype=tf.float32,
                              initializer=tf.random_normal_initializer(0, 0.05), trainable=True)
        g = get_var_maybe_avg('g', ema, shape=[num_filters], dtype=tf.float32,
                              initializer=tf.constant_initializer(1.), trainable=True)
        b = get_var_maybe_avg('b', ema, shape=[num_filters], dtype=tf.float32,
                              initializer=tf.constant_initializer(0.), trainable=True)

        # use weight normalization (Salimans & Kingma, 2016)
        W = tf.reshape(g, [1, 1, 1, num_filters]) * tf.nn.l2_normalize(V, [0, 1, 2])

        # calculate convolutional layer output
        x = tf.nn.bias_add(tf.nn.conv2d(x, W, [1] + stride + [1], pad), b)

        if init:  # normalize x
            m_init, v_init = tf.nn.moments(x, [0,1,2])
            scale_init = init_scale / tf.sqrt(v_init + 1e-10)
            with tf.control_dependencies([g.assign(g * scale_init), b.assign_add(-m_init * scale_init)]):
                x = tf.nn.l2_normalize(x, axis=[0,1,2])

        # apply nonlinearity
        if nonlinearity is not None:
            x = nonlinearity(x)

        return x

@add_arg_scope
def deconv2d(x, num_filters, filter_size=[3,3], stride=[1,1], pad='SAME', nonlinearity=None, init_scale=1., counters={}, init=False, ema=None, **kwargs):
    ''' transposed convolutional layer '''
    name = get_name('deconv2d', counters)
    xs = int_shape(x)
    if pad=='SAME':
        target_shape = [xs[0], xs[1]*stride[0], xs[2]*stride[1], num_filters]
    else:
        target_shape = [xs[0], xs[1]*stride[0] + filter_size[0]-1, xs[2]*stride[1] + filter_size[1]-1, num_filters]
    with tf.variable_scope(name):
        V = get_var_maybe_avg('V', ema, shape=filter_size+[num_filters,int(x.get_shape()[-1])], dtype=tf.float32,
                              initializer=tf.random_normal_initializer(0, 0.05), trainable=True)
        g = get_var_maybe_avg('g', ema, shape=[num_filters], dtype=tf.float32,
                              initializer=tf.constant_initializer(1.), trainable=True)
        b = get_var_maybe_avg('b', ema, shape=[num_filters], dtype=tf.float32,
                              initializer=tf.constant_initializer(0.), trainable=True)

        # use weight normalization (Salimans & Kingma, 2016)
        W = tf.reshape(g, [1, 1, num_filters, 1]) * tf.nn.l2_normalize(V, [0, 1, 3])

        # calculate convolutional layer output
        x = tf.nn.conv2d_transpose(x, W, target_shape, [1] + stride + [1], padding=pad)
        x = tf.nn.bias_add(x, b)

        if init:  # normalize x
            m_init, v_init = tf.nn.moments(x, [0,1,2])
            scale_init = init_scale / tf.sqrt(v_init + 1e-10)
            with tf.control_dependencies([g.assign(g * scale_init), b.assign_add(-m_init * scale_init)]):
                x = tf.nn.l2_normalize(x, axis=[0,1,2])

        # apply nonlinearity
        if nonlinearity is not None:
            x = nonlinearity(x)

        return x

@add_arg_scope
def nin(x, num_units, **kwargs):
    """ a network in network layer (1x1 CONV) """
    s = int_shape(x)
    x = tf.reshape(x, [np.prod(s[:-1]),s[-1]])
    x = dense(x, num_units, **kwargs)
    return tf.reshape(x, s[:-1]+[num_units])

''' meta-layer consisting of multiple base layers '''

@add_arg_scope
def gated_resnet(x, a=None, h=None, nonlinearity=concat_elu, conv=conv2d, init=False, counters={}, ema=None, dropout_p=0., **kwargs):
    xs = int_shape(x)
    num_filters = xs[-1]

    c1 = conv(nonlinearity(x), num_filters)
    if a is not None: # add short-cut connection if auxiliary input 'a' is given
        c1 += nin(nonlinearity(a), num_filters)
    c1 = nonlinearity(c1)
    if dropout_p > 0:
        c1 = tf.nn.dropout(c1, keep_prob=1. - dropout_p)
    c2 = conv(c1, num_filters * 2, init_scale=0.1)

    # add projection of h vector if included: conditional generation
    if h is not None:
        with tf.variable_scope(get_name('conditional_weights', counters)):
            hw = get_var_maybe_avg('hw', ema, shape=[int_shape(h)[-1], 2 * num_filters], dtype=tf.float32,
                                    initializer=tf.random_normal_initializer(0, 0.05), trainable=True)
        if init:
            hw = hw.initialized_value()
        c2 += tf.reshape(tf.matmul(h, hw), [xs[0], 1, 1, 2 * num_filters])

    a, b = tf.split(c2, 2, 3)
    c3 = a * tf.nn.sigmoid(b)
    return x + c3

''' utilities for shifting the image around, efficient alternative to masking convolutions '''

def down_shift(x):
    xs = int_shape(x)
    return tf.concat([tf.zeros([xs[0],1,xs[2],xs[3]]), x[:,:xs[1]-1,:,:]],1)

def right_shift(x):
    xs = int_shape(x)
    return tf.concat([tf.zeros([xs[0],xs[1],1,xs[3]]), x[:,:,:xs[2]-1,:]],2)

@add_arg_scope
def down_shifted_conv2d(x, num_filters, filter_size=[2,3], stride=[1,1], **kwargs):
    x = tf.pad(x, [[0,0],[filter_size[0]-1,0], [int((filter_size[1]-1)/2),int((filter_size[1]-1)/2)],[0,0]])
    return conv2d(x, num_filters, filter_size=filter_size, pad='VALID', stride=stride, **kwargs)

@add_arg_scope
def down_shifted_deconv2d(x, num_filters, filter_size=[2,3], stride=[1,1], **kwargs):
    x = deconv2d(x, num_filters, filter_size=filter_size, pad='VALID', stride=stride, **kwargs)
    xs = int_shape(x)
    return x[:,:(xs[1]-filter_size[0]+1),int((filter_size[1]-1)/2):(xs[2]-int((filter_size[1]-1)/2)),:]

@add_arg_scope
def down_right_shifted_conv2d(x, num_filters, filter_size=[2,2], stride=[1,1], **kwargs):
    x = tf.pad(x, [[0,0],[filter_size[0]-1, 0], [filter_size[1]-1, 0],[0,0]])
    return conv2d(x, num_filters, filter_size=filter_size, pad='VALID', stride=stride, **kwargs)

@add_arg_scope
def down_right_shifted_deconv2d(x, num_filters, filter_size=[2,2], stride=[1,1], **kwargs):
    x = deconv2d(x, num_filters, filter_size=filter_size, pad='VALID', stride=stride, **kwargs)
    xs = int_shape(x)
    return x[:,:(xs[1]-filter_size[0]+1):,:(xs[2]-filter_size[1]+1),:]
