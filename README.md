# VoiceConversion

[![Build Status](https://travis-ci.org/r9y9/VoiceConversion.jl.svg?branch=master)](https://travis-ci.org/r9y9/VoiceConversion.jl)
[![Build status](https://ci.appveyor.com/api/projects/status/byf0ru7b8d7hf2dn/branch/master?svg=true)](https://ci.appveyor.com/project/r9y9/voiceconversion-jl/branch/master)
[![Coverage Status](https://coveralls.io/repos/r9y9/VoiceConversion.jl/badge.svg)](https://coveralls.io/r/r9y9/VoiceConversion.jl)
[![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](LICENSE.md)

`VoiceConverision.jl` is a statistical voice conversion library for Julia.

Please note that this package is still under developement. Both bug reports and feature requests are welcome.

## Features

### Parameter conversion

- Frame-by-frame parameter conversion using joint Gaussian Mixture Models (GMMs) of source and target feature space [src/gmmmap.jl](src/gmmmap.jl)
- Trajectory parameter conversion based on maximum likelihood criterion w/o considering Gloval Variance (GV) [Toda 2007] [src/trajectory_gmmmap.jl](src/trajectory_gmmmap.jl)

### Waveform modification

- WORLD-based vocoding
- Direct waveform modification using Mel log Spectrum Approximation (MLSA) digital filter based on spectrum differencial [Kobayashi 2014]

## Supported Platforms

- Linux
- Mac OS X
- Windows

## Installation

Run the following commands on your julia interactive settion (REPL):

```julia
julia> Pkg.clone("https://github.com/r9y9/VoiceConversion.jl")
```

## Demonstration using [CMU Arctic](http://festvox.org/cmu_arctic/)

Please check [examples/cmu_arctic/cmu_arctic_demo.sh](examples/cmu_arctic/cmu_arctic_demo.sh).

## References

- [[Toda 2007] T. Toda, A. W. Black, and K. Tokuda, “Voice conversion based on maximum likelihood estimation of spectral parameter trajectory,” IEEE
Trans. Audio, Speech, Lang. Process, vol. 15, no. 8, pp. 2222–2235,
Nov. 2007.](http://isw3.naist.jp/~tomoki/Tomoki/Journals/IEEE-Nov-2007_MLVC.pdf)
- [[Kobayashi 2014] Kobayashi, Kazuhiro, et al. "Statistical Singing Voice Conversion with Direct Waveform Modification based on the Spectrum Differential." Fifteenth Annual Conference of the International Speech Communication Association. 2014.](http://isw3.naist.jp/~kazuhiro-k/resource/kobayashi14IS.pdf)
