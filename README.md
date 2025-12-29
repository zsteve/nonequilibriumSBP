## Code: Learning non-equilibrium diffusions with Schrödinger bridges: from exactly solvable to simulation-free

This repository provides code accompanying the paper "Learning non-equilibrium diffusions with Schrödinger bridges: from exactly solvable to simulation-free", presented at NeurIPS 2025.

`fm.py` provides an implementation of the mvOU-OTFM algorithm (Alg. 1): `LinearEntropicOTFM` and `LinearBridgeMatcher`.
`EntropicOTFM` and `BridgeMatcher` are corresponding implementations in the case of a Brownian reference.
`GaussianOUSB` implements computation of the Gaussian Schrödinger bridge using the formulas of Theorem 2. 

`notebooks` provide code for reproducing experiments from the paper. 

If this paper and code are useful for your own research, please consider citing our work:

```
Learning non-equilibrium diffusions with Schr\"odinger bridges: from exactly solvable to simulation-free
Stephen Y. Zhang and Michael Stumpf
The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025
```



