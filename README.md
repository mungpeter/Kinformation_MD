## Kinformation_MD
Classification of Protein Kinase Conformations in MD trajectory to yield _C-helix-in/DFG-in (CIDI)_, _C-helix-out/DFG-in (CODI)_, _C-helix-in/DFG-out (CIDO)_, _C-helix-out/DFG-out (CODO)_, and _C-helix-in|out/DFG-intermediate (wCD)_.

###############################################################################################

**Generate SciKit Learn Machine-Learning Models for Kinformation_MD**
This script generates various classifers using one of the following methods: _RandomForest (rf)_, _Support-Vector Machine (svm)_, _Neural Network (nn)_, _K-nearest Neighbors (kn)_, _Decision Tree (dt)_, _Gaussian Process (gp)_, and _Gradient Boosting (gb)_, via Jupyter/iPython interface.

All of these classifiers are able to distinguish the 5 different kinase conformations with error < 5%, with SVM gives slightly better result than others.

```
  > 0_kinfo_SK_modelgen.py
      (this code use Jupyter/iPython to generate SKLearn ML Model sequentially)
```


###############################################################################################

```
  > 1_kinfo_traj_MD.py
      -templ    [Template PDB structure (exact match to Topology Atom List and pre-aligned to Ref Structure 1ATP)]
      -traj     [Trajectory file, or an ordered list of traj filenames (format: dcd)]
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
  python   # stable: 3.6.8+
  numpy    # stable: 1.16.2
  pandas   # stable: 0.24.2
  mdtraj   # stable: 1.9.3
  sklearn  # stable: 0.20.3
  rpy2     # stable: 2.9.4
  tqdm     # stable: 4.31.1
  pathos   # stable: 0.2.3
  argparse # stable: 1.1 
```

__Citation 1:__ [\*Ung PMU, \*Rahman R, Schlessinger A. Redefining the Protein Kinase Conformational Space with Machine Learning. Cell Chemical Biology (2018) 25(7), 916-924.](https://doi.org/10.1016/j.chembiol.2018.05.002)

