import tensorflow as tf
import numpy as np

def gather_axis(params, indices, axis=0):
    return tf.stack(tf.unstack(tf.gather(tf.unstack(params, axis=axis), indices)), axis=axis)