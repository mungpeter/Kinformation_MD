# Kinformation_MD
Classification of Protein Kinase Conformations in MD trajectory to yield _C-helix-in/DFG-in (CIDI)_, _C-helix-out/DFG-in (CODI)_, _C-helix-in/DFG-out (CIDO)_, _C-helix-out/DFG-out (CODO)_, and _C-helix-in|out/DFG-intermediate (wCD)_.

```
  > 1_kinfo_traj_MD.py
      -templ    [Template PDB structure (exact match to Topology Atom List and aligned to Ref Structure 1ATP)]
      -traj     [Trajectory file, or an ordered list of traj filenames (format: dcd)]
      -b3k      [(beta-3 Lys)  Residue Number in Template Structure]
      -dfg      [(DFG Asp)     Residue Number in Template Structure]
      -glu      [(C-helix Glu) Residue Number in Template Structure]
      -out      [Output prefix]
      
    Optional:
      -pkl      [Use pre-pickled trajectory data generated from previous run (def: False)]
      -superp   [VMD-like selection string to perform superposition (def: False)]
      -use_r_rf [Use R::randomForest instead of SKLearn RFClassifier (def: False)]
      -use_sk   [Use SKLearn ML model: rf|svm|nn|kn|dt|gp|gb (def: rf)]
      -lib      [Kinformation_MD Repository database path (unless hard-coded)]
      
e.g.>  1_kinfo_traj_MD.py
          -templ strada_cido.prot.1atp.pdb -traj strada_cidi.2.200ps.dcd 
          -b3k 39 -dfg 152 -glu 57 -out test
          -superp 'resid 20 to 50 and 100 to 200'
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

__Citation 2:__ [\*Rahman R, \*Ung PMU, Schlessinger A. KinaMetrix: a web resource to investigate kinase conformations and inhibitor space. 
Nucleic Acids Research (2019) 47(D1), D361–D366.](https://doi.org/10.1093/nar/gky916)
