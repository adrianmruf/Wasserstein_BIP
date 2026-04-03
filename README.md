Readme
=
**Python code accompanying the paper 'Well-posedness of Bayesian Inverse Problems
for Hyperbolic Conservation Laws' by Siddhartha Mishra, David Ochsner, Adrian M. Ruf, and Franziska Weber**

03.04.2026

### General Setup
The code in this repository employs a Metropolis–Hastings Markov Chain Monte Carlo (MCMC) method to sample from the posterior distribution. The proposal kernel is a standard random walk $v^{(n)} = u^{(n)} + \beta \xi^{(n)}$, where $\xi^{(n)} \sim \mathcal{N}(0, \mathcal{C})$ and $\beta$ is the step size. Monotone finite volume schemes are used as the forward solvers to compute the numerical solution operators for the observation operators. 

---

### 1. Experiment 1 - Inverse problem for the shock location and amplitude in a Riemann problem for Burgers’ equation

*   **Problem:** Infers the initial datum of a Riemann problem for Burgers' equation $w_t + (\frac{w^2}{2})_x = 0$ on the domain $(-1, 1)$ with outflow boundary conditions.
*   **Parameters:** We infer 3 parameters $u = (\delta_1, \delta_2, \sigma_0)$ corresponding to the left state amplitude ($1+\delta_1$), right state amplitude ($\delta_2$), and the shock location ($\sigma_0$).
*   **Forward Model:** Computed using the Rusanov finite volume scheme with a grid discretization parameter of $\Delta x = 2/128$. Observation data is taken at time $T=1$ over 5 measurement intervals.
*   **MCMC Setup:** 
    *   Chain length: $N = 2500$
    *   Burn-in: $b = 500$
    *   Sample interval: $\tau = 20$ (every 20th state is used after burn-in)
    *   Step size $\beta(k)$: Piecewise linear decay formulation to ensure short burn-in and steady-state convergence.
*   **Outputs:** Includes code to compute the Wasserstein distance to evaluate convergence with respect to both chain length and spatial discretization $\Delta x$.

### 2. Experiment 2 - Inverse problem for the transport speed and jump amplitude for a Riemann problem with flux discontinuity

*   **Problem:** Infers parameters for a conservation law with a spatially discontinuous flux: $w_t + (k(x)f(w) + (1 - k(x))g(w))_x = 0$. The equation switches from the Transport equation to Burgers' equation across the interface at $x=0$.
*   **Parameters:** We infer 2 parameters $u = (\delta, a)$ corresponding to the left state of the initial Riemann datum ($\delta$) and the transport speed ($a$).
*   **Forward Model:** Computed using a finite volume scheme adapted for flux discontinuities, with grid discretization $\Delta x = 2/128$ and CFL-like parameter $\lambda = 0.4$. Observations are taken at time $T=1$ over 6 measurement intervals.
*   **MCMC Setup:** 
    *   Chain length: $N = 2500$
*   **Outputs:** Computes the posterior histograms and MAP estimators, as well as Wasserstein error tracking to demonstrate experimental orders of convergence.

### 3. Experiment 3 - An inverse problem for systems of conservation laws (Sod's shock tube)

*   **Problem:** Demonstrates that the MCMC framework extends practically to systems of conservation laws by inferring the initial discontinuity in the 1D Euler equations (Sod's shock tube problem).
*   **Parameters:** We infer 6 parameters $u = (\delta_L, \gamma_L, \beta_L, \delta_R, \gamma_R, \beta_R)$ representing the perturbations in the left and right states for density, velocity, and pressure.
*   **Forward Model:** Computed using the HLLC Riemann solver scheme with a grid discretization of $\Delta x = 1/128$. Observations are taken at time $T=0.2$ over 5 spatial measurement intervals.
*   **MCMC Setup:** 
    *   Chain length: $N = 1500$
    *   Burn-in: $b = 500$
    *   Sample interval: $\tau = 10$
    *   Step size: Constant $\beta = 0.0005$

---

Copyright (C) 2026, Adrian M Ruf

The code is based upon `https://github.com/ochsnerd/ip_mcmc`
