# Import packages
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input, Subtract, Activation, Layer
from tensorflow.keras import backend
from tensorflow.keras.initializers import Ones
import numpy as np

class Exploglayer(Layer):

    def __init__(self, nhiddenLayers=10):
        super(Exploglayer, self).__init__()
        # Activation functions
        log_activ = backend.log
        exp_activ = backend.exp

        # Define exponential and logarithmic layers
        self.layerexp = Dense(nhiddenLayers, activation=Activation(exp_activ), name='exp')  # first hidden layer
        self.layerlog = Dense(1, use_bias=False, kernel_initializer=Ones, activation=Activation(log_activ),
                              trainable=False, name='log')  # output layer

    def call(self, inputs):
        x = self.layerexp(inputs)
        return self.layerlog(x)

class Dslenet(Model):

    def __init__(self, data, nhiddenLayers=10):
        super(Dslenet, self).__init__()
        self.layernorminput = data.layernorminput
        self.layernorminput._name = 'prepro_input'

        self.layernormoutput = data.layernormoutput
        self.layernormoutput._name = 'prepro_output'

        self.top = Exploglayer(nhiddenLayers)
        self.top._name = 'top'
        self.bottom = Exploglayer(nhiddenLayers)
        self.bottom._name = 'btm'

    def call(self, inputs):
        # Preprocessing layer
        inputs = self.layernorminput(inputs)
        # Top layers
        xtop = self.top(inputs)
        # Bottom layers
        xbottom = self.bottom(inputs)
        # Postprocessing layer
        output = Subtract()([xtop, xbottom])
        return self.layernormoutput(output)

    def fit(self, data, *args, **kwargs):
        # Define in- and outputs
        x = data.input
        y = data.output
        super(Dslenet, self).fit(x, y, *args, **kwargs)

    def export_weights_npz(self, filename):
        # Export weights and parameters of preprocessing layer to npz file
        # Can be imported by C++
        np.savez(filename,
                 kWeightsTop=self.get_layer('top').get_weights()[0],
                 kWeightsBtm=self.get_layer('btm').get_weights()[0],
                 kBiasTop=self.get_layer('top').get_weights()[1],
                 kBiasBtm=self.get_layer('btm').get_weights()[1],
                 kMeanIn=self.get_layer('prepro_input').mean,
                 kVarIn=self.get_layer('prepro_input').variance,
                 kMeanOut=self.get_layer('prepro_output').mean,
                 kVarOut=self.get_layer('prepro_output').variance)