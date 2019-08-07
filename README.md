## Kinformation_MD
Classification of Protein Kinase Conformations in MD trajectory to yield _C-helix-in/DFG-in (CIDI)_, _C-helix-out/DFG-in (CODI)_, _C-helix-in/DFG-out (CIDO)_, _C-helix-out/DFG-out (CODO)_, and _C-helix-in|out/DFG-intermediate (wCD)_.

#####################################################################################

**Generate SciKit Learn Machine-Learning Classification Models for Kinformation_MD**

This script generates various classifers using one of the following methods: _RandomForest (rf)_, _Support-Vector Machine (svm)_, _Neural Network (nn)_, _K-nearest Neighbors (kn)_, _Decision Tree (dt)_, _Gaussian Process (gp)_, and _Gradient Boosting (gb)_, via Jupyter/iPython interface.

- For the initial data of 3,611 processed kinase structures (dataset: 17.10.09), 325 structures have been manually determined for their DFG and overall conformational states and used as the training set.
- For the training set, missing data is imputed and all data is normalized ( (var-mean)/max(abs(var-mean)) )
- Training set is splited at 80/20 (280 and 65 of 325 structures) for training and validation, respectively.

All of these classifiers are able to distinguish the 5 different kinase conformations with error < 6%, with _nn, kn, gb_ classifiers doing the best (if not overfitted due to small validation size: 65 samples), followed by _rf, dt_.

_C-helix/DFG out-of bag error_:

_nn, kn, gb_:   1.538 %

_rf, dt_:       3.077 %

_svm, gp_:  6.154 %


```
  > 0_kinfo_SK_modelgen.py
      (this code use Jupyter/iPython to generate SKLearn ML Model sequentially)
```


####################################################################################

**Classification the conformation of protein kinase in the MD simulation trajectory**

This script classifies the conformation the MD trajectory of a protein kinase. User supplies the template PDB structure (or topology file) and the corresponding residue numbers of the 3 key residues - _b3k=conserved Lys on beta-3 of small-lobe_, _dfg=Asp of DFG-motif_, _glu=conserved Glu on aC-helix_ - corresponding to those found in PKA's 1ATP - _b3k=59_, _dfg=171_, _glu=78_.

The template structure of the kinase should be pre-superposed to the large-lobe of PKA (1ATP reference residues: 'resid 122-138+162-183', pymol-selection [syntax](https://pymol.org/dokuwiki/?id=selection), which includes the beta-sheet and E-,F-helices. 

For the trajectory, it should be pre-processed to remove water and salts for faster read-in, while the frames should be superposed to the template kinase structure on the most stable portion of the large-lobe (exclude DFG-motif, activation-loop until AP-motif, G-helix, and any non-canonical structure). If the trajectory is raw and hasn't been superposed to a common reference frame, then supply the residues for superposition using the _-superp_ flag (input is VMD-atomselection [syntax](https://www.ks.uiuc.edu/Research/vmd/vmd-1.2/ug/vmdug_node137.html): '_resid_ 100 _to_ 150 _and_ 200 _to_ 300')

During the first-pass of the run, relevant residue positions are read and stored in compressed pickle (\*.pkl.bz2) in case a different classification model is used. This compressed pickle can be used (_-pkl_) in place of the trajectory input (_-traj_) for faster processing.

For the classification model, two options are available: R-based and Sci-Kit Learn-based. 
- R-based:  _-use_r_rf_ flag is an on/off option, and uses a classifier generated by R::randomForest package
- SK-based: _-use_sk_ flag together with which classification model. There are 7 models available, all tested to have error < 5%. Among them _SVM_ seems to be marginally better

User should point the library path to this repository's database (_-lib xxx/Kinformation_MD/z_database_) unless you hard-code the path by modifying the scripts.

```
  > 1_kinfo_traj_MD.py
      -templ    [Template PDB structure (exact match to Topology Atom List and pre-aligned to Ref Structure 1ATP)]
      -traj     [Trajectory file, or an ordered list of traj filenames (format: dcd, nc, crd, xtc)]
      -b3k      [(beta-3 Lys)  Residue Number in Template Structure]
      -dfg      [(DFG Asp)     Residue Number in Template Structure]
      -glu      [(C-helix Glu) Residue Number in Template Structure]
      -out      [Output prefix]
      
    Optional:
      -pkl      [Use pre-pickled trajectory data generated from previous run, in pkl.bz2 format (def: False)]
      -superp   [VMD-like selection string to perform traj superposition to template PDB (def: False)]
      -use_r_rf [Use R::randomForest instead of SKLearn RFClassifier (def: False)]
      -use_sk   [Use SKLearn ML model: rf|svm|nn|kn|dt|gp|gb (def: rf)]
      -lib      [Kinformation_MD Repository database path (unless hard-coded)]
      
e.g.>  1_kinfo_traj_MD.py
          -templ strada_cido.prot.1atp.pdb -traj strada_cidi.2.200ps.dcd -pkl cidi.pkl.bz2
          -b3k 39 -dfg 152 -glu 57 -out test
          -superp 'resid 100 to 200 and 250 to 300'
          -use_sk svm
          -lib '/Users/xxx/scripts/Kinformation_MD/z_database'
```

* All relevant kinase residues are referenced to Protein Kinase A (cAMP-dependent protein kinase catalytic domain) crystal structure __1ATP__

* Required packages:
```
    python      # stable: 3.6.8+
    numpy       # stable: 1.16.2
    pandas      # stable: 0.24.2
    mdtraj      # stable: 1.9.3
    sklearn     # stable: 0.20.3
    rpy2        # stable: 2.9.4
    tqdm        # stable: 4.31.1
    pathos      # stable: 0.2.3
    argparse    # stable: 1.1 
```

__Citation 1:__ [\*Ung PMU, \*Rahman R, Schlessinger A. Redefining the Protein Kinase Conformational Space with Machine Learning. Cell Chemical Biology (2018) 25(7), 916-924.](https://doi.org/10.1016/j.chembiol.2018.05.002)

