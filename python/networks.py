# Import packages
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input, Subtract, Activation, Layer
from tensorflow.keras import backend
from tensorflow.keras.initializers import Ones
import numpy as np
from dsle_mpc import data


class Exploglayer(Layer):

    def __init__(self, nhiddenLayers=10):
        super(Exploglayer, self).__init__()
        # Activation functions
        log_activ = backend.log
        exp_activ = backend.exp

        # Define exponential and logarithmic layers
        self.layerexp = Dense(nhiddenLayers, activation=Activation(exp_activ))  # first hidden layer
        self.layerlog = Dense(1, use_bias=False, kernel_initializer=Ones, activation=Activation(log_activ),
                              trainable=False)  # output layer

    def call(self, inputs):
        x = self.layerexp(inputs)
        return self.layerlog(x)


class Dslenet(Model):

    def __init__(self, nhiddenLayers=10):
        super(Dslenet, self).__init__()
        self.top = Exploglayer(nhiddenLayers)
        self.bottom = Exploglayer(nhiddenLayers)

    def call(self, inputs):
        # Top layers
        xtop = self.top(inputs)
        # Bottom layers
        xbottom = self.bottom(inputs)

        return Subtract()([xtop, xbottom])

    def fit(self, data, *args, **kwargs):
        # Save nromalization parameters
        self.layernorminput = data.layernorminput
        self.layernormoutput = data.layernormoutput

        x = data.input_norm
        y = data.output_norm

        # Reserve 67% of data for validation
        if not ('validation_split' in kwargs) and not ('validation_data' in kwargs):
            kwargs['validation_split'] = 0.67

        # Set default to 100 epochs
        if not ('epochs' in kwargs):
            kwargs['epochs'] = 100
        super(Dslenet, self).fit(x, y, *args, **kwargs)

        # Save weights
        self.mean_in = tf.make_ndarray(tf.make_tensor_proto(self.layernorminput.mean))
        self.var_in = tf.make_ndarray(tf.make_tensor_proto(self.layernorminput.variance))
        self.mean_out = float(tf.make_ndarray(tf.make_tensor_proto(self.layernormoutput.mean)))
        self.var_out = float(tf.make_ndarray(tf.make_tensor_proto(self.layernormoutput.variance)))
        weights = self.get_weights()
        self.kWeightsTop = weights[0]
        self.kWeightsBtm = weights[3]
        self.kBiasTop = weights[1]
        self.kBiasBtm = weights[4]

    def compile(self, *args, **kwargs):
        # Choose mean squared error as default loss function
        if not ('loss' in kwargs):
            kwargs['loss'] = 'mse'
        super(Dslenet, self).compile(*args, **kwargs)

    def export_weights_npz(self):
        np.savez('params',
                 kWeightsTop=self.kWeightsTop,
                 kWeightsBtm=self.kWeightsBtm,
                 kBiasTop=self.kBiasTop,
                 kBiasBtm=self.kBiasBtm,
                 kMeanIn=self.mean_in,
                 kVarIn=self.var_in,
                 kMeanOut=self.mean_out,
                 kVarOut=self.var_out)
