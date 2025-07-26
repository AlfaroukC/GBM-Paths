# This program simulates and plots multiple paths of a financial assets price and its logarithm over
# time using the GBM (Geometric Brownian Motion) model.

# The GBM is a continuous time stochastic process used to model asset prices, 
# assuming that the log of the price follows a random walk with a consistent 
# drift and unpredictable volatility.

# NoOfPaths = An integer representing the number of simulation paths for the model to generate.
# NoOfSteps = An integer representing the number of discrete time steps into which the total time horizon T is divided.
# T = A float representing the total time horizon for the simulation (years).
# r = A float representing the risk free interest rate (or drift rate for the log price in the GBM model) expressed as an annual decimal.
# sigma = A float representing the volatility of the asset's returns, expressed as an annual decimal.
# S_0 = A float representing the initial (spot) price of the underlying asset at the start of the simulation.

# Z = An array of random numbers drawn from a SND(mean 0, std 1). These represent the "Brownian Motion" component in the asset's price 
# movement at each time step for each path. 
# X = An array that stores the log of the asset prices for each path over time. 
# S = An array that stores the simulated asset prices for each path over time. This is the exponential of the X array (S = np.exp(X))
# time = An array that stores the discrete time points for the simulation ranging from 0 to t.

# dt = A float representing the length of each discrete time step. (T / NoOfSteps)
# i = The current time step index.
# paths = A dictionary used to return the calculated time array, X array(log prices), and S array(Actual/Simulated prices) from the GeneratePathsGBMABM function.

# timeGrid = The array of time points, extracted from Paths["time"] so it's the same as the time variable.

import numpy as np
import matplotlib.pyplot as plt

def GeneratePathsGBMABM(NoOfPaths, NoOfSteps, T, r, sigma, S_0, normalisation, seed=None):

    if seed is not None:
        np.random.seed(seed)

    Z = np.random.normal(0.0,1.0,[NoOfPaths, NoOfSteps])
    X = np.zeros([NoOfPaths, NoOfSteps+1])
    S = np.zeros([NoOfPaths, NoOfSteps+1])
    time = np.zeros([NoOfSteps+1])

    X[:,0] = np.log(S_0) # The initial condition for the log price.

    dt = T / float(NoOfSteps)
    for i in range(0,NoOfSteps):
        if NoOfPaths > 1 and normalisation == True:
            # The samples from the normal have a mean of 0 and a variance of 1.
            # This normalisation step ensures that for each time step (t), the random 
            # samples used across all the paths have exactly a mean of 0 and std of 1.
            # This however isn't strictly necessary for GBM which relies on unbiased normal samples.
            Z[:,i] = (Z[:,i] - np.mean(Z[:,i])) / np.std(Z[:,i])
            
        # This is the discretised formula for Geometric Brownian Motion.
        # X(t+dt) = X(t) + (mu - 0.5 * sigma^2) * dt + sigma * sqrt(dt) * Z
        X[:,i+1] = X[:,i] + (r - 0.5 * sigma **2) * dt + sigma * np.power(dt, 0.5) * Z[:,i]
        time[i+1] = time[i] + dt

    S = np.exp(X) # Convert the log prices back to the actual prices
    return {"time":time, "X":X, "S":S}


def Log_Asset_PricePaths():

    NoOfPaths = 25
    NoOfSteps = 500
    T = 1
    r = 0.05
    sigma = 0.4
    S_0 = 100

    # Simulation without normalisation
    paths_no_norm = GeneratePathsGBMABM(NoOfPaths, NoOfSteps, T, r, sigma, S_0, False, seed = 52)
    Plot_Path(paths_no_norm["time"], paths_no_norm["X"], paths_no_norm["S"], "(No Z-Norm)")
    
    # Simulation with normalisation
    paths_with_norm = GeneratePathsGBMABM(NoOfPaths, NoOfSteps, T, r, sigma, S_0, True, seed = 53)
    Plot_Path(paths_with_norm["time"], paths_with_norm["X"], paths_with_norm["S"], "(With Z-Norm)")
    
    plt.show(block=True)

def Plot_Path(time, X, S, title_suffix):

    plt.figure(figsize=(12,6))
    plt.subplot(1, 2, 1) # 1 row, 2 columns, 1st plot
    plt.plot(time, np.transpose(X))
    plt.grid(True)
    plt.xlabel("Time")
    plt.ylabel("Log Price X(t)")
    plt.title(f"Log Price {title_suffix}")

    plt.subplot(1, 2, 2) # 1 row, 2 columns, 2nd plot
    plt.plot(time, np.transpose(S))
    plt.grid(True)
    plt.xlabel("Time")
    plt.ylabel("Asset Price S(t)")
    plt.title(f"Asset Price {title_suffix}")
    plt.suptitle(f"Simulation {title_suffix}")
Log_Asset_PricePaths()