# TensorFlow example
import keras
from keras import backend as K
import numpy as np
tf_session = K.get_session()
val = np.array([[1, 2], [3, 4]])
kvar = K.variable(value=val)
input = keras.backend.placeholder(shape=(2, 4, 5))
K.shape(kvar).shape[0]
#<tf.Tensor 'Shape_8:0' shape=(2,) dtype=int32>
K.shape(input)
K.int_shape(kvar)[1]

# <tf.Tensor 'Shape_9:0' shape=(3,) dtype=int32>
  # To get integer shape (Instead, you can use K.int_shape(x))
  >>> K.shape(kvar).eval(session=tf_session)
  array([2, 2], dtype=int32)
  >>> K.shape(input).eval(session=tf_session)
  array([2, 4, 5], dtype=int32)
