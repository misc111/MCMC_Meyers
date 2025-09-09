Level Chain Ladder — Meyers (Flask + MCMC)

Overview

This app demonstrates the Level Chain Ladder (LCL) method from Meyers with an interactive UI:
- Run an MCMC sampler (Metropolis–Hastings) with live-updating trace via Server‑Sent Events.
- Posterior predictive simulation to generate missing triangle cells.
- Pick a simulation index to view a simulated cumulative triangle.
- View a histogram for the distribution of total ultimate or total reserve.

Model

- Likelihood: log C_{w,d} ~ Normal(alpha_w + beta_d, sigma_d)
- Constraints: beta_D = 0; sigma monotone via sigma_d = sum_{i=d}^D a_i with a_i in [0,1]
- Priors:
  - alpha_w ~ Normal(log(Premium_w) + logELR, sd = 10^(-1/2))
  - beta_d ~ Uniform(-5, 5), d < D
  - a_i ~ Beta(2, 2)  (light prior discouraging extremes and σ collapse)
  - logELR ~ Uniform(-1, 0.5)
- Posterior sampled via Metropolis–Hastings in 4 blocks: alpha (component‑wise scalar updates), beta, a, logELR

Project Structure

- app.py — Flask app, SSE endpoints, routes
- lcl_mcmc.py — Data model, MCMC implementation, predictive simulation
- templates/index.html — UI layout
- static/js/app.js — UI logic, SSE, simple canvas charts
- static/css/style.css — Styling

Run Locally

1) Ensure Python 3.9+ with Flask and NumPy installed:
   pip install flask numpy

2) Start the app:
   python app.py

3) Open in a browser:
   http://127.0.0.1:5000/

Usage

- Use Sample Data: loads a synthetic 10x10 cumulative triangle with NaNs in the missing upper-right.
- MCMC Control: set iterations, burn-in, thin, adapt length, number of chains, and a trace parameter (e.g., logELR). Click Run MCMC to start. The trace plot will live update.
- Posterior Predictive: after MCMC finishes, click Simulate Reserves to simulate missing cells for a subset of draws. Choose which histogram to view (Total Ultimate or Total Reserve) and load any simulation index to view its triangle.

Notes

- The sampler uses reflective Gaussian random-walk proposals within the constrained domains for beta ([-5,5]), a ([0,1]), and logELR ([-1,0.5]).
- Proposal scales auto-tune during the adaptive phase toward target acceptance rates (0.25–0.40 for vectors, ~0.5 for scalars).
- For demonstration, convergence diagnostics (R-hat, ESS) are not calculated; use the live trace qualitatively, or extend as needed.
- The histogram shows distributions of total ultimate (sum of C[:, D]) and total reserve (sum over AY of max(0, ultimate - latest observed cumulative)).
