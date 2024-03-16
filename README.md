[![DOI](https://zenodo.org/badge/402479185.svg)](https://zenodo.org/badge/latestdoi/402479185)

# Evolutionary model of Variant Effects (EVE)

This is the official code repository for the paper "Disease variant prediction with deep generative models of evolutionary data" (https://www.nature.com/articles/s41586-021-04043-8). This project is a joint collaboration between the Marks lab (https://www.deboramarkslab.com/) and the OATML group (https://oatml.cs.ox.ac.uk/).

## Overview
EVE is a set of protein-specific models providing for any single amino acid mutation of interest a score reflecting the propensity of the resulting protein to be pathogenic. For each protein family, a Bayesian VAE learns a distribution over amino acid sequences from evolutionary data. It enables the computation of an evolutionary index for each mutant, which approximates the log-likelihood ratio of the mutant vs the wild type. A global-local mixture of Gaussian Mixture Models separates variants into benign and pathogenic clusters based on that index. The EVE scores reflect probabilistic assignments to the pathogenic cluster.

## Usage
The end to end process to compute EVE scores consists of three consecutive steps:
1. Train the Bayesian VAE on a re-weighted multiple sequence alignment (MSA) for the protein of interest => train_VAE.py
2. Compute the evolutionary indices for all single amino acid mutations => compute_evol_indices.py
3. Train a GMM to cluster variants on the basis of the qevol indices then output scores and uncertainties on the class assignments => train_GMM_and_compute_EVE_scores.py
We also provide all EVE scores for all single amino acid mutations for thousands of proteins at the following address: http://evemodel.org/.

## Example scripts
The "examples" folder contains sample bash scripts to obtain EVE scores for a protein of interest (using PTEN as an example).
MSAs and ClinVar labels are provided for 4 proteins (P53, PTEN, RASH and SCN5A) in the data folder. 

## Data requirements
The only data required to train EVE models and obtain EVE scores from scratch are the multiple sequence alignments (MSAs) for the corresponding proteins. 

### MSA creation

We build multiple sequence alignments for each protein family by performing five search iterations of the profile HMM homology search tool Jackhmmer against the UniRef100 database of non-redundant protein sequences (downloaded on April 20th 2020). We retrieve sequences that align to at least 50% of the target protein sequence, and columns with at least 70% residue occupancy. This is done using [EVcouplings](https://github.com/debbiemarkslab/EVcouplings/tree/80d30b3d2568ae3327f973346be73cdcd41f678b). 

We explore a range of bit score thresholds, using 0.3 bits per residue as a reference, and select the best possible multiple sequence alignment based on the criteria of maximal coverage of the target protein sequence and sufficient, but not excessive, number of sequences in the alignment (the latter implying an alignment that is too lenient). Specifically, we prioritize alignments with coverage L<sub>cov</sub> ≥ 0.8L, where L is the length of the target protein sequence, and with a total number of sequences N such that 100,000 ≥ N ≥ 10L. If these requirements cannot be met, we sequentially relax them down to L<sub>cov</sub> ≥ 0.7L and N ≤ 200,000. These criteria are met for 97% of alignments. For the remaining 3%, we drop the coverage constraint entirely. Following this procedure, we have so far obtained a set of 3,219 clinically relevant proteins with corresponding evolutionary training data. While we expect the performance of our model to depend on the quality of the multiple sequence alignments, we do not find strong correlation between performance and alignment depth N/L<sub>cov</sub>.

Our github repo provides the MSAs for 4 proteins: P53, PTEN, RASH & SCN5A (see data/MSA). MSAs for all proteins may be accessed on our website (https://evemodel.org/).

### MSA pre-processing
The EVE codebase provides basic functionalities to pre-process MSAs for modelling (see the MSA_processing class in utils/data_utils.py). By default, sequences with 50% or more gaps in the alignment and/or positions with less than 70% residue occupancy will be removed. These parameters may be adjusted as needed by the end user.

### ClinVar labels
The script "train_GMM_and_compute_EVE_scores.py" provides functionalities to compare EVE scores with reference labels (e.g., ClinVar). Our github repo provides labels for 4 proteins: P53, PTEN, RASH & SCN5A (see data/labels). ClinVar labels for all proteins may be accessed on our website (https://evemodel.org/).

## Software requirements
The entire codebase is written in python. Package requirements are as follows:
  - python=3.7
  - pytorch=1.7
  - cudatoolkit=11.0
  - scikit-learn=0.24.1
  - numpy=1.20.1
  - pandas=1.2.4
  - scipy=1.6.2
  - tqdm
  - matplotlib
  - seaborn
  - numba

The corresponding environment may be created via conda and the provided protein_env.yml file as follows:
```
  conda env create -f protein_env.yml
  conda activate protein_env
```

## License
This project is available under the MIT license.

## Reference
If you use this code, please cite the following paper:
```bibtex
@article{Frazer2021DiseaseVP,
  title={Disease variant prediction with deep generative models of evolutionary data.},
  author={Jonathan Frazer and Pascal Notin and Mafalda Dias and Aidan Gomez and Joseph K Min and Kelly P. Brock and Yarin Gal and Debora S. Marks},
  journal={Nature},
  year={2021}
}
```

Links: 
- Paper: https://www.nature.com/articles/s41586-021-04043-8
- Pre-print: https://www.biorxiv.org/content/10.1101/2020.12.21.423785v1
