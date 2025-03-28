# AlphaTracer

${{\color{darkblue}\Huge{\textsf{ultra-fast\ structure\ inference\ by\ tracing\ existing\ structural\ models\ \}}}}\$

The aim of AlphaTracer is to speed up protein structure inference by at least 10-fold over the current SOTA for rapid structure inference (ESMfold) and to increase the feasibility of medium-scale and large-scale protein structure inference locally on CPU or with minimal GPU access. The method we use is to rapidly find close homologs in the **AlphaFold Database** and 'trace' the relevant portion of the protein structure, filling in the gaps with **ESMfold** and "stitching" the segments together using **int2cart**. 

The current version produces Cα co-ordinates, i.e. an "alpha-carbon trace". This data allows the use of various tools including **Reseek** for protein structure comparisons. <a href="https://github.com/rcedgar/reseek" target="_blank">Reseek Github Repo</a>

The code available here is the alpha version of AlphaTracer (v0.1), beginning with the first two modules:  
(1) **AlphaSearch**   
(2) **AlphaTrace**   
for (1) finding homologs in the AlphaFold DB and (2) producing Cα traces of input sequences which are highly similar to the homologs. 

The full code and a manuscript with benchmarking results are in preparation. 
The current code takes an average of ~0.1s/protein of 200aas as compared to >1s for ESMfold, and further speed-up is expected. 
The speed advantage over using only ESMfold will depend on whether relatively similar sequences already have structures in the AlphaFold Database.


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
AlphaTracer consists of four modules:

AlphaSearch,
AlphaTrace,
AlphaGaps,
AlphaStitch

  ```sh
bash alpha_search_v0.1.bash input_seqs.fa alpha_search.dmnd 8
  ```

  ```sh
python3 alpha_trace_v0.1.py
  ```


<!-- Roadmap -->
## Roadmap 

Updates planned:
Complete the final two modules and provide AlphaGaps as a Jupyter Notebook / Google Colab Notebook  
Further improve running speed of the code  
Explore use of ESM Atlas and clustered sequence databases  

<!-- CONTACT -->
## Contact
The author, Zachary Ardern, can be contacted via <a href="https://zacharyardern.com" target="_blank">zacharyardern.com</a>

<br>
<br>
<br>

Readme adapted from: https://github.com/othneildrew/Best-README-Template 

