# VoiceConversion

[![Build Status](https://travis-ci.org/r9y9/VoiceConversion.jl.svg?branch=master)](https://travis-ci.org/r9y9/VoiceConversion.jl)

VoiceConversion.jl is a repository of my statistical voice conversion research experiments.

## Parameter conversion

- Frame-by-frame parameter conversion using joint Gaussian Mixture Models (GMMs) of source and target feature space
- Trajectory parameter conversion based on maximum likelihood criterion [Toda 2007]

## Waveform modification

- WORLD-based vocoding
- Direct waveform modification using Mel log Spectrum Approximation (MLSA) digital filter based on spectrum differencial [Kobayashi 2014]

## References

- [[Toda 2007] T. Toda, A. W. Black, and K. Tokuda, “Voice conversion based on maximum likelihood estimation of spectral parameter trajectory,” IEEE
Trans. Audio, Speech, Lang. Process, vol. 15, no. 8, pp. 2222–2235,
Nov. 2007.](http://isw3.naist.jp/~tomoki/Tomoki/Journals/IEEE-Nov-2007_MLVC.pdf)
- [[Kobayashi 2014] 小林 和弘, 戸田 智基, Graham Neubig, Sakriani Sakti, 中村 哲. “差分スペクトル補正に基づく統計的歌声声質変換”, 日本音響学会2014年春季研究発表会(ASJ). 東京. 2014年3月.](http://www.phontron.com/paper/kobayashi14asj.pdf)
