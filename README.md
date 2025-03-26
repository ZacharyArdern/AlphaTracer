# AlphaTracer
ultra-fast structure inference from existing structural models

The aim of AlphaTracer is to speed up protein structure inference by approx. 10-fold over ESMfold and to increase the feasibility of medium-scale protein structure inference (full proteomes) locally on CPU or with minimal GPU access. The method we use is to rapidly find close homologs and 'trace' the relevant portion of the protein structure, filling in the gaps with a method such as ESMfold.

The current code is the alpha version of AlphaTracer. It is available as a Google Colab notebook at: 

The full code and a manuscript with benchmarking results are in preparation. 
The current alpha pipeline takes an average of [~2s]/protein of 200aas as compared to [~4s] for ESMfold but a further speed up of at least 4-fold is expected


<!-- GETTING STARTED -->
## Getting Started

To get a local copy of AlphaTracer up and running follow these steps (availability of full code for local install is under development).

### Prerequisites
* diamond
* polars
* int2cart
* sidechainnet
* esmfold
database of sequences for pre-computed protein models (e.g. AlphaFold or ESM Atlas)

### Installation

* install int2cart
  ```sh
  git clone https://github.com/THGLab/int2cart.git
  cd int2cart
  pip install -e .
  ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage
AlphaTracer consists of three modules:

Alpha_find
Alpha_gapfill
Alpha_stitch


Readme adapted from: https://github.com/othneildrew/Best-README-Template 

