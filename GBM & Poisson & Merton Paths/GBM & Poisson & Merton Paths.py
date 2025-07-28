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
# GBM Path Generator
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

    NoOfPaths = 5
    NoOfSteps = 500
    T = 1
    R = 0.05
    Sigma = 0.4
    MU = 0.15
    S_0 = 100
    # M = Money savings account.
    M = lambda r,T: np.exp(R*T)

    # Simulation without normalisation
    Paths_No_Norm = GenerateGBMPaths(NoOfPaths, NoOfSteps, T, R, Sigma, S_0, False)
    Plot_Path(Paths_No_Norm["time"], Paths_No_Norm["X"], Paths_No_Norm["S"], "(No Z-Norm)")
    
    # Simulation with normalisation
    Paths_With_Norm = GenerateGBMPaths(NoOfPaths, NoOfSteps, T, R, Sigma, S_0, True)
    Plot_Path(Paths_With_Norm["time"], Paths_With_Norm["X"], Paths_With_Norm["S"], "(With Z-Norm)")

    # Martingale property checker
    S = Paths_With_Norm["S"]
    # ES = Expected Asset Price
    ES = np.mean(S[:, -1])
    print(ES)
    # ESM = Expected Discounted Asset Price
    ESM = np.mean(S[:, -1]/M(R,T))
    print(ESM)

    # Monte Carlo Paths
    PathsQ = GenerateGBMPaths(NoOfPaths, NoOfSteps, T, R, Sigma, S_0, True)
    S_Q = PathsQ["S"]
    PathsP = GenerateGBMPaths(NoOfPaths, NoOfSteps, T, MU, Sigma, S_0, True)
    S_P = PathsP["S"]
    time = PathsQ["time"]

    #Discounted Stock Paths
    S_QDisc = np.zeros([NoOfPaths, NoOfSteps + 1])
    S_PDisc = np.zeros([NoOfPaths, NoOfSteps + 1])
    i = 0
    for i, ti in enumerate(time):
        S_QDisc[:, i] = S_Q[:, i]/M(R, ti) 
        S_PDisc[:, i] = S_P[:, i]/M(R, ti)

    Plot_Discounted_Stock_Path(time, S_0, S, R, T, M, MU, S_QDisc, S_PDisc)
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
def Plot_Discounted_Stock_Path(time, S_0, S, R, T, M, MU, QDISC, PDISC):

    # S(T)/M(T) with Stock growing with rate R
    plt.figure(3)
    plt.grid(True)
    plt.xlabel("Time")
    plt.ylabel("S(t)")
    eSM_Q = lambda t: S_0 * np.exp(R * t) / M(R, t)
    plt.plot(time, eSM_Q(time), 'r--')
    plt.plot(time, np.transpose(QDISC), 'Blue')
    plt.legend(['E^Q[S(t)/M(t)]','paths S(t)/M(t)'])

    # S(T)/M(T) with Stock growing with rate MU
    plt.figure(4)
    plt.grid(True)
    plt.xlabel("Time")
    plt.ylabel("S(t)")
    eSM_P = lambda t: S_0 * np.exp(MU * t) / M(R, t)
    plt.plot(time, eSM_P(time), 'r--')
    plt.plot(time, np.transpose(PDISC), 'Blue')
    plt.legend(['E^P[S(t)/M(t)]','paths S(t)/M(t)'])
Log_Asset_PricePaths()
# End of GBM Path Generator

# Poisson Path Generator
def GeneratePoissonPaths(NoOfPaths, NoOfSteps, T, XIP):
    # Create empty matrices Maybe this is a waste and should be optimised later.
    X = np.zeros([NoOfPaths, NoOfSteps + 1])
    XC = np.zeros([NoOfPaths, NoOfSteps + 1])
    Time = np.zeros([NoOfSteps + 1])
    DT = T / NoOfSteps
    Z = np.random.poisson(XIP * DT, [NoOfPaths, NoOfSteps])

    for i in range(0, NoOfSteps):
        # Normalisation
        X[:, i + 1] = X[:, i] + Z[:, i]
        XC[:, i + 1] = XC[:, i] - XIP * DT + Z[:, i]
        Time[i + 1] = Time[i] + DT

    Paths = {"Time":Time, "X":X, "XComp":XC}
    return Paths
def PoissonCalculation():
    NoOfPaths = 25
    NoOfSteps = 500
    T = 30 
    XIP = 1

    Paths = GeneratePoissonPaths(NoOfPaths, NoOfSteps, T, XIP)
    TimeGrid = Paths["Time"]
    X = Paths["X"]
    XC = Paths["XComp"]

    plt.figure(5)
    plt.plot(TimeGrid, np.transpose(X), '-b')
    plt.title("Poisson Process with X")
    plt.grid(True)
    plt.xlabel("Time")
    plt.ylabel("X(t)")

    plt.figure(6)
    plt.plot(TimeGrid, np.transpose(XC), '-b')
    plt.title("Poisson Process with XC")
    plt.grid()
    plt.xlabel("time")
    plt.ylabel("X(t)")
PoissonCalculation()
# End of Poisson Path Generator

# Merton Path Generator
def GenerateMertonPaths(NoOfPaths, NoOfSteps, S0, T, XIP, MuJ, SigmaJ, R, Sigma):
    # Create empty matrices Maybe this is a waste and should be optimised later
    X = np.zeros([NoOfPaths, NoOfSteps + 1])
    S = np.zeros([NoOfPaths, NoOfSteps + 1])
    Time = np.zeros([NoOfSteps + 1])
    DT = T / NoOfSteps
    X[:, 0] = np.log(S0)
    S[:, 0] = S0

    # Expectation E(e^J) for J~N(MuJ, sigmaJ**2)
    EeJ = np.exp(MuJ + 0.5 * SigmaJ ** 2)

    ZPois = np.random.poisson(XIP * DT, [NoOfPaths, NoOfSteps])
    Z = np.random.normal(0.0, 1.0, [NoOfPaths, NoOfSteps])
    J = np.random.normal(MuJ, SigmaJ, [NoOfPaths, NoOfSteps])

    for i in range(0, NoOfSteps):
        Z[:, i] = (Z[:, i] - np.mean(Z[:, i])) / np.std(Z[:, i])

        X[:, i + 1] = X[:, i] + (R - XIP * (EeJ - 1) - 0.5 * Sigma ** 2) * DT + Sigma * np.sqrt(DT) * Z[:, i] + J[:, i] * ZPois[:, i]
        Time[i + 1] = Time[i] + DT

    S = np.exp(X)
    Paths = {"Time":Time, "X":X, "S":S}
    return Paths
def MertonCalculation():
    NoOfPaths = 25
    NoOfSteps = 500
    T = 5
    XIP = 1
    MuJ = 0
    SigmaJ = 0.7
    Sigma = 0.2
    S0 = 100
    R = 0.05
    Paths = GenerateMertonPaths(NoOfPaths, NoOfSteps, S0, T, XIP, MuJ, SigmaJ, R, Sigma)
    TimeGrid = Paths["Time"]
    X = Paths["X"]
    S = Paths["S"]

    plt.figure(7)
    plt.plot(TimeGrid, np.transpose(X))
    plt.title("Merton Process with X")
    plt.grid(True)
    plt.xlabel("Time")
    plt.ylabel("X(t)")

    plt.figure(8)
    plt.plot(TimeGrid, np.transpose(S))
    plt.title("Merton Process with S")
    plt.grid()
    plt.xlabel("time")
    plt.ylabel("X(t)")
MertonCalculation()
plt.show()