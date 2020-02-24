#!/usr/bin/env python3

import sys,os
#sys.path.append('/Users/xxx/scripts/Kinformation_MD')

import re
import bz2
import time
import pickle
import numpy as np
import pandas as pd

from collections import Counter

from x_kinfo_R_classify import R_RunRandomForest

from x_kinfo_traj_functions import Normalization
from x_kinfo_traj_functions import state_dfg, state_kinfo

Ref_Test_Cols = [ 'p1p1x','p2p2x','r3r3x','h_cgvc','ang_NHs','ang_CHs',
                  'dist_NH','dist_CH']

Ref_Final_Cols= [ 'Class',
                  'cidi_prob','cido_prob','codi_prob','codo_prob','wcd_prob',
                  'dfg_conf','dfg_prob','p1p1x','p2p2x','r3r3x','h_cgvc',
                  'ang_NHs','ang_CHs','dist_NH','dist_CH']

norm_cols = ['ang_NHs','ang_CHs','dist_NH','dist_CH']

dfg_train_cols = ['p1p1x','p2p2x','r3r3x','dist_NH','dist_CH']
full_train_cols= ['h_cgvc','ang_NHs','ang_CHs','dist_NH','dist_CH','dfg_conf']

sk_dfg_model = {
  'rf': 'SK_rf_model_dfg.pkl.bz2', 'svm': 'SK_svm_rbf_model_dfg.pkl.bz2', 
  'nn': 'SK_nn_model_dfg.pkl.bz2', 'kn':  'SK_kn_model_dfg.pkl.bz2', 
  'gb': 'SK_gb_model_dfg.pkl.bz2', 'gp':  'SK_gp_model_dfg.pkl.bz2', 
  'dt': 'SK_dt_model_dfg.pkl.bz2'  }
  
sk_chx_model = {
  'rf': 'SK_rf_model_full.pkl.bz2', 'svm': 'SK_svm_lin_model_full.pkl.bz2', 
  'nn': 'SK_nn_model_full.pkl.bz2', 'kn':  'SK_kn_model_full.pkl.bz2', 
  'gb': 'SK_gb_model_full.pkl.bz2', 'gp':  'SK_gp_model_full.pkl.bz2', 
  'dt': 'SK_dt_model_full.pkl.bz2'  }

kinfo_rf_data      = 'kinfo_rf_data_pre_normal.190527.csv.gz'
kinfo_norm_param   = 'kinfo_data_normalize_param.pkl'

##########################################################################
## Process the collected structural data by Normalization against a known
## kinase PDB derived Normalization parameter (mean and max), then run thru
## the RandomForest model

def KinfoClassify( traj_df, lib_dir, outpref, use_r_rf, use_sk='rf' ):

  ## make sure the trajectory dataframe has same columns as RF models
  if traj_df.columns.isin(Ref_Test_Cols).sum() != len(Ref_Test_Cols):
    print('  ERROR: Column in Trajectory not matching required: ')
    print(traj_df.columns)
    print(Ref_Test_Cols)
    sys.exit()

  ## Load parameters for data normalization, the Normalize ang_/dist_ data
  with open(lib_dir+kinfo_norm_param, 'rb') as fi:
    norm_param = pickle.load(fi)
  traj_df[norm_cols] = Normalization(traj_df[norm_cols], norm_param)

#############
  ## use R-generated RandomForest model for classification
  if use_r_rf:
    print('## INFO: Loading R RandomForest models...')
    result_df = R_RunRandomForest(traj_df, lib_dir, models='')
    result_df.to_csv(outpref+'.R_rf_kinfo_classify.csv', sep=',')
    return None

##############
  ## Use SK-generated RandomForest model for classification
  if use_sk:
    ## Load SK ML models
    print('## INFO: Loading trained SK ML models...')
    with bz2.open(lib_dir+sk_dfg_model[use_sk], 'rb') as fd:
      rfc_dfg = pickle.load(fd)
    with bz2.open(lib_dir+sk_chx_model[use_sk], 'rb') as fc:
      rfc = pickle.load(fc)

    result_df = SK_RunML(traj_df, use_sk, models=[rfc_dfg, rfc])
    print(Counter(result_df.Class))
    result_df.to_csv(outpref+'.SK_{}_kinfo_classify.csv'.format(use_sk), sep=',')
    return None



##########################################################################
##########################################################################
def SK_RunML( df, ml_alg, models='' ):

  rfc_dfg, rfc = models
  ##### classify DFG conformation of trajectory frames #####
  start = time.perf_counter()
  traj_dfg_pred = rfc_dfg.predict(df[dfg_train_cols])
  traj_dfg_prob = rfc_dfg.predict_proba(df[dfg_train_cols])

  # append 'dfg_conf' and probability data to traj frame data
  df['dfg_conf'] = traj_dfg_pred
  df['dfg_prob'] = np.max(traj_dfg_prob, axis=1)
  print('SK_RF Classify DFG:   {:.6f} s'.format((time.perf_counter()-start)))

  ##### classify Chelix/DFG conformation of traj frames #####
  start = time.perf_counter()
  traj_full_pred = rfc.predict(df[full_train_cols])
  traj_full_prob = rfc.predict_proba(df[full_train_cols])
  print('SK_RF Classify Kinfo: {:.6f} s'.format((time.perf_counter()-start)))

  ## append 'Class' and probability to traj frame data 
  start = time.perf_counter()
  df['dfg_conf']    = state_dfg(pd.DataFrame(traj_dfg_pred))
  df['Class']       = state_kinfo(pd.DataFrame(traj_full_pred))
  df['cidi_prob'] = traj_full_prob[:,0]
  df['cido_prob'] = traj_full_prob[:,1]
  df['codi_prob'] = traj_full_prob[:,2]
  df['codo_prob'] = traj_full_prob[:,3]
  df['wcd_prob']  = traj_full_prob[:,4]
  print('Append Kinfo data:    {:.6f} s'.format((time.perf_counter()-start)))

  df.index.name = 'frame'
  return df[Ref_Final_Cols]

################################################################
