# SparseSCVB0

Basic implementation of SparseSCVB0 algorithm for LDA in [Rashidul Islam, and James Foulds. "Scalable Collapsed Inference for High-Dimensional Topic Models." Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers). 2019](https://www.aclweb.org/anthology/N19-1291).

## Prerequisites

* [Julia](https://julialang.org/) (tested on v0.6.4.1)
* Optional Julia packages: MAT and JLD (to save the generated results)

The code is tested on windows and linux operating systems. It should work on any other platform.

## Data format

The input is a single file, with one line per document where each word is separated by a space. Words in each document are represented by one-based dictionary indices.  The demo is provided on NIPS corpus [NIPS corpus, due to Sam Roweis](https://cs.nyu.edu/~roweis/data.html). See more in data folder where NIPS.txt and NIPSdict.txt contain the corpus and dictionary, respectively. 
