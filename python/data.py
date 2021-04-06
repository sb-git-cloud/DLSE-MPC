from scipy.io import loadmat
from tensorflow.keras.layers.experimental.preprocessing import Normalization

class Data:
    def __init__(self, filename, m_input='u', m_output='y'):
        # Load data
        data = loadmat(filename)

        # Normalize input data
        self.input = data[m_input]
        self.layernorminput = Normalization()
        self.layernorminput.adapt(self.input)
        self.input_norm = self.layernorminput(self.input)

        # Normalize output data
        self.output = data[m_output]
        self.layernormoutput = Normalization()
        self.layernormoutput.adapt(self.output)
        self.output_norm = self.layernormoutput(self.output)
