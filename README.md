# SFM-Net

Author: Liu Chunting
Affiliation: Department of Intelligence Science and Technology, Graduate School of Informatics, Kyoto University
E-mail: liuchunting@kuicr.kyoto-u.ac.jp


## Data
* PDB files are downloaded from https://www.rcsb.org/
* The division of the 10 folds in the ten-fold cross-validation experiment follows the same division as MpbPPI (refering to https://github.com/arantir123/MpbPPI).

## Installation
(1) Install protT5
The language model protT5 (referring to https://github.com/agemagician/ProtTrans) can be installed via
```
pip install transformers
pip install sentencepiece
```

(2) Install DSSP, PSIBLAST, and FoldX
* Install BLAST+ for extracting PSSM (position-specific scoring matrix) profiles:
<br>To download and install the BLAST+ package (https://ftp.ncbi.nlm.nih.gov/blast/executables/LATEST/) and BLAST database, please refer to BLAST® Help (https://www.ncbi.nlm.nih.gov/books/NBK52640/). 
* Install DSSP for extracting SS (Secondary structure) profiles:
<br>To download and install the DSSP, please refer to https://swift.cmbi.umcn.nl/gv/dssp/
* Install FoldX for the PDB files of mutant complexes:
<br>To download and install the FoldX, please refer to https://foldxsuite.crg.eu/
