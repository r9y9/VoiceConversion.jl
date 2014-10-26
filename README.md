# VoiceConversion

[![Build Status](https://travis-ci.org/r9y9/VoiceConversion.jl.svg?branch=master)](https://travis-ci.org/r9y9/VoiceConversion.jl)

VoiceConversion.jl is a repository of my statistical voice conversion research experiments.

## Features

### Parameter conversion

- Frame-by-frame parameter conversion using joint Gaussian Mixture Models (GMMs) of source and target feature space [src/gmmmap.jl](src/gmmmap.jl)
- Trajectory parameter conversion based on maximum likelihood criterion w/o considering Gloval Variance (GV) [Toda 2007] [src/trajectory_gmmmap.jl](src/trajectory_gmmmap.jl)

### Waveform modification

- WORLD-based vocoding
- Direct waveform modification using Mel log Spectrum Approximation (MLSA) digital filter based on spectrum differencial [Kobayashi 2014]

## Installation

```julia
Pkg.clone("https://github.com/r9y9/VoiceConversion.jl")
Pkg.build("VoiceConversion")
```

All dependencies are resolved with `Pkg.clone` and `Pkg.build`.

## References

- [[Toda 2007] T. Toda, A. W. Black, and K. Tokuda, “Voice conversion based on maximum likelihood estimation of spectral parameter trajectory,” IEEE
Trans. Audio, Speech, Lang. Process, vol. 15, no. 8, pp. 2222–2235,
Nov. 2007.](http://isw3.naist.jp/~tomoki/Tomoki/Journals/IEEE-Nov-2007_MLVC.pdf)
- [[Kobayashi 2014] Kobayashi, Kazuhiro, et al. "Statistical Singing Voice Conversion with Direct Waveform Modification based on the Spectrum Differential." Fifteenth Annual Conference of the International Speech Communication Association. 2014.](http://isw3.naist.jp/~kazuhiro-k/resource/kobayashi14IS.pdf)
