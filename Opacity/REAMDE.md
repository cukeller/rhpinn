# Deep neural networks to calculate Rosseland mean and 500nm continuum opacities 

To create the Rosseland mean and 500-nm continuum opacity deep neural networks, execute the following commands

    python opacity_nn_rosseland.py
    python opacity_nn_500nm.py

This creates the neural network files `opacity_rosseland.keras` and `opacity_500nm.keras` and plots the comparison between input table values and deep neural network output along with the residuals between the two.
