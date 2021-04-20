Difference of Log-Sum-Exp Neural Networks to Solve Data-Driven Model  Predictive Control Tracking Problems - Python code base
========

The python version is currently in beta stage. It draws on cvxpy and tensorflow and is separated into three parts.

+ "data.py" contains the data class whose main purpose is to load and extract statistics from the input and output data. The 
data object also serves as an input for the neural network.

+ "networks.py" provides the DSLE neural network class, composed of two log-sum-exp layers.

+ "algorithm.py" includes the core algorithm being the DSLEA and the MPC extension.

### data.py
Data.py contains the Data class and in addition to the tensorflow requires the scipy package to load the data
(currently only *.mat files). By instantiating a data object via a path to a mat-file, in addition to saving the data 
itself, we also initialize a normalization layer. The latter computes the mean and variance (both over time) for each 
input and output. This information will be used in the neural network model.

### networks.py
Networks.py contains two classes. Class "LogSumExpLayer" inherits from tensorflow.keras.layers.Layer and represents the
upper and lower half of the DSLE network, i.e. the log of the sum of the exponentials.
The other class "DsleNet" uses this layer to compose the log-sum-exp neural network, by inhereting from 
tensorflow.keras.Model. It additionally uses the data object to add a preprocessing layer that subtracts the mean and
divides by the variance, for each inout and output variable. Please see the tensorflow/keras documentation for details
on the model and layer classes.

### algorithms.py
This file has two classes, the "DsleaProblem" and the "MpcProblem". The prior describes the general DSLEA found in the
paper, using cvxpy to solve the convex optimization problem. The class "MpcProblem" bvuilds on "DsleaProblem" to solve
an MPC in a standard form.

### Example code train network
    data = data.Data('data.mat')  # load data with u as inputs and y as outputs
    model = networks.Dslenet(data, 3)  # 3 hidden neurons for top/bottom half respectively
    model.compile(loss='mse')  # define loss function
    model.fit(data, epochs=100)  # fit data

### MPC code snippet
Once the network is trained, we can generate a suboptimal control sequence by first defining the constraints, e.g. 
    
    lb = np.array([0,0,0,0,0,0])  # lower bound for input sequence with MPC horizon N=3
    ub = np.array([1,2,1,2,1,2])  # upper bound for input sequence with MPC horizon N=3

and then instantiating the MPC using **kwargs via
    
    mpc = networks.MpcProblem(model, 5, 2, lb=lb, ub=ub)  # state dimension = 5, inout dimensions = 2

and at each time instance calling

    u0 = mpc_control  # initialize optimization problem with old optimal solution ([u_0^T, u_1^T, ... u_{N-1}^T])
    mpc_control = mps.solve(current_state, u0)
