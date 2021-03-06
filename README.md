<p align="center">
<img width="400px" src="green-flux-logo.png"/>
</p>

[![Build Status](https://github.com/EddieCueto/GreenFlux.jl/workflows/CI/badge.svg)](https://github.com/EddieCueto/GreenFlux.jl/actions)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://EddieCueto.github.io/GreenFlux.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://EddieCueto.github.io/GreenFlux.jl/dev)
[![Build Status](https://travis-ci.com/EddieCueto/GreenFlux.jl.svg?branch=master)](https://travis-ci.com/EddieCueto/GreenFlux.jl) 
[![Coverage](https://codecov.io/gh/EddieCueto/GreenFlux.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/EddieCueto/GreenFlux.jl)
<!-- [![Build Status](https://github.com/EddieCueto/GreenFlux.jl/workflows/CI/badge.svg)](https://github.com/EddieCueto/GreenFlux.jl/actions) -->

GreenFlux adds Green AI functionalities to [Flux](https://github.com/FluxML/Flux.jl) so you can make more precise reports on your peer reviewed papers or conference submissions of the electricity consumed by your model during training and number of `non-embedding Floating Point OPerations` that your Flux model will require during forward and back propagation.

To install just do:
```julia
] add GreenFlux
```
**Due to the limited hardware and driver accessibility this framework has been tested and works currently in Linux.**

Tested Successfully On

GPUs:

    - NVIDIA GeForce GTX 1050

CPUs:

    - Intel(R) Core(TM) i7-7700HQ CPU @ 2.80GHz 

OS:

    - Ubuntu 18.04.1
