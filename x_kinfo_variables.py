
## SKLearn machine learning models for DFG-motif conformation classification
def SKLearnDFGModelFiles():
  sk_dfg_model = {
    'rf': 'SK_rf_model_dfg.pkl.bz2', 'svm': 'SK_svm_rbf_model_dfg.pkl.bz2', 
    'nn': 'SK_nn_model_dfg.pkl.bz2', 'kn':  'SK_kn_model_dfg.pkl.bz2', 
    'gb': 'SK_gb_model_dfg.pkl.bz2', 'gp':  'SK_gp_model_dfg.pkl.bz2', 
    'dt': 'SK_dt_model_dfg.pkl.bz2'  }
  return sk_dfg_model

## SKLearn machine learning models for C-helix/DFG conformation classification
def SKLearnKinfoModelFiles():
  sk_chx_model = {
    'rf': 'SK_rf_model_full.pkl.bz2', 'svm': 'SK_svm_lin_model_full.pkl.bz2', 
    'nn': 'SK_nn_model_full.pkl.bz2', 'kn':  'SK_kn_model_full.pkl.bz2', 
    'gb': 'SK_gb_model_full.pkl.bz2', 'gp':  'SK_gp_model_full.pkl.bz2', 
    'dt': 'SK_dt_model_full.pkl.bz2'  }
  return sk_chx_model

## Hard-coded variables and ML models
def KinfoVariables():
  Vars = {
    'lib_dir':          './z_database',
    'home_dir':         './',

    'ref_pdb':          '1ATP.mdtraj.pdb',
    'ref_pkl':          '1ATP.mdtraj.pkl.bz2',
    'ref_dfg':          171,
    'ref_b3k':          59,
    'ref_c_glu':        78,

    'R_dfg_model':      'R_rf_model_dfg.190527.rda',
    'R_chx_model':      'R_rf_model_full.190527.rda',

    'kinfo_rf_data':    'kinfo_rf_data_pre_normal.190527.csv.gz',
    'kinfo_rf_data_norm':'kinfo_rf_data_normalized.190527.csv.gz',

    'kinfo_norm_param': 'kinfo_data_normalize_param.pkl',
    'kinfo_data':       'stdy_kinase.param.171009.csv',
  }
  return Vars