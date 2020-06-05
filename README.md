# DLSE-MPC
Aprroximation of MPC problems via DLSE networks

1. Generate Dataset:

Run the code "Generate_test_data.m" to generate the training dataset.
The code has to be run several times with different values of the parameters yr in line 15 (we tested for yr = 2e4, yr = 2.5e4, and 3e4).

2. Train the DLSE networks:

Run the code "trainDLSEreactor.m" to train the DLSE networks.
The code has to be run several times with different values of the parameters yr in line 5 (we tested for yr = 2e4, yr = 2.5e4, and 3e4).

3. Test the DLSE approximation:

Run the code "optimMPCreactor.m" to test the trained DLSE networks.
Run the code "plainMPC.m" to run the classical MPC.
