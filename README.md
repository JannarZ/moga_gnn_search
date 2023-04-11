# Multi-Objective Genetic Algorithm (MOGA) search using Graph Neural Network (GNN) Surrogate

This repository contains code for MOGA search of the physical realistic grain boundary (GB) structures of 2D materials as mentioned in the paper "Graph Neural Network Guided Evolutionary Search of Grain Boundaries in 2D Materials". Other details for GA search of 2D materials can be found in the second paper in [Citation](#citation) section.

## Usage
To perform the MOGA searching, python scripts in the [ga_scripts](ga_scripts/) musted be copied to the folder where the search is performed. Two structure input files similar to [12left](example/12left), [12right](example/12right) should also present in the searching folder, as they are treated as the bulk parts of 2D materials. For the two bulk part structures, same format as POSCAR file used by [VASP](https://www.vasp.at/) is adopted. A graph neural network (GNN) model trained to predict the structural energy such as [potential file](example/1212tersoff_950each_model_new.pt) in the example should also be copied to the search folder. The file named `ga_input.yaml` can be modified so the correct settings applied. The function of the flags is dicussed in the comment of the [`ga_input.yaml`](ga_scripts/ga_input.yaml) in ga_scripts folder. 

## Prerequisites
[ASE](https://wiki.fysik.dtu.dk/ase/)  
[DEAP](https://github.com/DEAP/deap)  
[Numpy](https://numpy.org/)  
[PyTorch](https://pytorch.org/)   
[NetWorkX](https://networkx.org/)  
[PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/index.html) 

## GNN Model
The code of the GNN model used for search is share in the following two repositories:  
https://github.com/JannarZ/gnn_bp_gb_tersoff  
https://github.com/JannarZ/gnn_bp_gb_dft

## Citation
*<div class="csl-entry">Zhang, J., Koneru, A., Sankaranarayanan, S. K. R. S., &#38; Lilley, C. M. (2023). Graph Neural Network Guided Evolutionary Search of Grain Boundaries in 2D Materials. <i>ACS Applied Materials &#38; Interfaces</i>. https://doi.org/10.1021/ACSAMI.3C01161</div>*  

*Zhang J, Srinivasan S, Sankaranarayanan SK, Lilley CM. Evolutionary inverse design of defects at graphene 2D lateral interfaces. Journal of Applied Physics. 2021 May 14;129(18):185302. (https://aip.scitation.org/doi/full/10.1063/5.0046469)*

