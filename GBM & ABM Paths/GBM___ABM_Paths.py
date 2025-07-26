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
# S_0 = A float representing the initial (spot) price of the underlying asset at the start of the simulation in pounds(GBP).

# Z = An array of random numbers drawn from a SND(mean 0, std 1). These represent the "Brownian Motion" component in the asset's price 
# movement at each time step for each path. 
# X = An array that stores the log of the asset prices for each path over time. 
# S = An array that stores the simulated asset prices for each path over time. This is the exponential of the X array (S = np.exp(X))
# time = An array that stores the discrete time points for the simulation ranging from 0 to t.

# dt = A float representing the length of each discrete time step. (T / NoOfSteps)
# i = The current time step index.
# paths = A dictionary used to return the calculated time array, X array(log prices), and S array(Actual/Simulated prices) from the GeneratePathsGBMABM function.


import numpy as np
import matplotlib.pyplot as plt

def GenerateGBMPaths(NoOfPaths, NoOfSteps, T, R, Sigma, S_0, Normalisation):

    if NoOfPaths <= 0:
        raise ValueError(f"NoOfPaths ({NoOfPaths}) must be positive.")
    if NoOfSteps <= 0:
        raise ValueError(f"NoOfSteps ({NoOfSteps}) must be positive.")
    if T <= 0:
        raise ValueError(f"T ({T}) must be positive.")
    if S_0 <= 0:
        raise ValueError(f"S_0 ({S_0}) must be positive.")
    if Sigma < 0:
        raise ValueError("Volatility (sigma) cannot be negative.")
    if not isinstance(Normalisation, bool):
        raise ValueError("Normalisation can only be true or false.")

    np.random.seed(52)

    Z = np.random.normal(0.0,1.0,[NoOfPaths, NoOfSteps])

    time = np.linspace(0, T, NoOfSteps + 1)
    
    X = np.zeros([NoOfPaths, NoOfSteps + 1])
    X[:,0] = np.log(S_0) # The initial condition for the log price.

    dt = T / float(NoOfSteps)
    
    if Normalisation:
            # The samples from the normal have a mean of 0 and a variance of 1.
            # This normalisation step ensures that for each time step (t), the random 
            # samples used across all the paths have exactly a mean of 0 and std of 1.
            # This however isn't strictly necessary for GBM which relies on unbiased normal samples.
            # The normalisation helps to reduce variability in small samples but may not reflect real world randomness.
            Z = (Z - np.mean(Z, axis = 0)) / np.std(Z, axis = 0)
            
    # Discretised formula for Geometric Brownian Motion.
    # X(t+dt) = X(t) + (mu - 0.5 * sigma^2) * dt + sigma * sqrt(dt) * Z
    Increments = (R - 0.5 * Sigma ** 2) * dt + Sigma * np.sqrt(dt) * Z
    X[:, 1:] = X[:, 0:1] + np.cumsum(Increments, axis=1)

    S = np.exp(X) # Convert the log prices back to the actual prices
    return {"time":time, "X":X, "S":S}


def Log_Asset_PricePaths():

    NoOfPaths = 25
    NoOfSteps = 500
    T = 1
    R = 0.05
    Sigma = 0.4
    S_0 = 100

    # Simulation without normalisation
    Paths_No_Norm = GenerateGBMPaths(NoOfPaths, NoOfSteps, T, R, Sigma, S_0, False)
    Plot_Path(Paths_No_Norm["time"], Paths_No_Norm["X"], Paths_No_Norm["S"], "(No Z-Norm)")
    
    # Simulation with normalisation
    Paths_With_Norm = GenerateGBMPaths(NoOfPaths, NoOfSteps, T, R, Sigma, S_0, True)
    Plot_Path(Paths_With_Norm["time"], Paths_With_Norm["X"], Paths_With_Norm["S"], "(With Z-Norm)")
    
    plt.show(block=True)

def Plot_Path(time, X, S, title_suffix):

    plt.figure(figsize=(12,6))
    plt.subplot(1, 2, 1) # 1 row, 2 columns, 1st plot
    plt.plot(time, np.transpose(X), alpha=0.5)
    plt.grid(True)
    plt.xlabel("Time")
    plt.ylabel("Log Price X(t) (GBP)")
    plt.title(f"Log Price {title_suffix}")

    plt.subplot(1, 2, 2) # 1 row, 2 columns, 2nd plot
    plt.plot(time, np.transpose(S), alpha=0.5)
    plt.grid(True)
    plt.xlabel("Time")
    plt.ylabel("Asset Price S(t) (GBP)")
    plt.title(f"Asset Price {title_suffix}")
    plt.suptitle(f"Simulation {title_suffix}")

Log_Asset_PricePaths()