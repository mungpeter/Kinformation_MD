#!/usr/bin/env python3

import sys,os
import re
import bz2
import time
import pickle
import numpy as np
import pandas as pd

from collections import Counter

from x_kinfo_variables import KinfoVariables
from x_kinfo_variables import SKLearnDFGModelFiles
from x_kinfo_variables import SKLearnKinfoModelFiles

#from x_kinfo_R_classify import R_RunRandomForest  # rpy2 isn't well maintained

from x_kinfo_traj_functions import Normalization
from x_kinfo_traj_functions import state_dfg, state_kinfo

####################

Vars = KinfoVariables()
kinfo_norm_param = Vars['kinfo_norm_param']

###################

Ref_Test_Cols = [ 'p1p1x','p2p2x','r3r3x','h_cgvc','ang_NHs','ang_CHs',
                  'dist_NH','dist_CH']

Ref_Final_Cols= [ 'Class',
                  'cidi_prob','cido_prob','codi_prob','codo_prob','wcd_prob',
                  'dfg_conf','dfg_prob','p1p1x','p2p2x','r3r3x','h_cgvc',
                  'ang_NHs','ang_CHs','dist_NH','dist_CH']

norm_cols = ['ang_NHs','ang_CHs','dist_NH','dist_CH']

dfg_train_cols = ['p1p1x','p2p2x','r3r3x','dist_NH','dist_CH']
full_train_cols= ['h_cgvc','ang_NHs','ang_CHs','dist_NH','dist_CH','dfg_conf']

##########################################################################
## Process the collected structural data by Normalization against a known
## kinase PDB derived Normalization parameter (mean and max), then run thru
## the RandomForest model

def KinfoClassify( traj_df, lib_dir, outpref, use_sk='rf' ):

  ## make sure the trajectory dataframe has same columns as RF models
  if traj_df.columns.isin(Ref_Test_Cols).sum() != len(Ref_Test_Cols):
    print('  \033[31mERROR: Column in Trajectory not matching required: \033[0m')
    print(traj_df.columns)
    print(Ref_Test_Cols)
    sys.exit()

  ## Load factors for data normalization, then Normalize ang_/dist_ data
  norm_param = pd.read_csv(lib_dir+kinfo_norm_param ,sep=',', comment='#')
  traj_df[norm_cols] = Normalization(traj_df[norm_cols], norm_param=norm_param)


#############
  ## use R-generated RandomForest model for classification
#  if use_r_rf:
#    print('\033/34m## Loading R RandomForest models...\033[0m')
#    result_df = R_RunRandomForest(traj_df, lib_dir, models=models)
#    result_df.to_csv(outpref+'.R_rf_kinfo_classify.csv', sep=',')
#    print(' \033[34mWrite to:\033[0m {0}{1}'.format(outpref+'.R_rf_kinfo_classify.csv'))
#    return None

##############
  ## Use SK-generated RandomForest model for classification
  if use_sk:

    sk_dfg_models = SKLearnDFGModelFiles()
    sk_chx_models = SKLearnKinfoModelFiles()

    ## Load SK ML models
    print('\033[34m## Loading trained SK ML models... \033[0m'+use_sk)
    rfc_dfg  = pickle.load(bz2.open(lib_dir+sk_dfg_models[use_sk], 'rb'))
    rfc_full = pickle.load(bz2.open(lib_dir+sk_chx_models[use_sk], 'rb'))

    result_df = SK_RunML(traj_df, use_sk, models=[rfc_dfg, rfc_full])
#    print(Counter(result_df.Class))
    print('\n\033[34mConformation   Counts\033[0m')
    for conf, num in Counter(result_df.Class).most_common():
      print('\033[35m     {0}\t\033[31m{1}\033[0m'.format(conf, num))

    result_df.to_csv(outpref+'.SK_{0}_kinfo_classify.csv'.format(use_sk), sep=',')
    print(' \033[34mWrite to:\033[0m {0}.SK_{1}_kinfo_classify.csv'.format(outpref, use_sk))
    return None



##########################################################################
##########################################################################
def SK_RunML( df, ml_alg, models='' ):

  rfc_dfg, rfc_full = models
  ##### classify DFG conformation of trajectory frames #####
  start = time.perf_counter()
  traj_dfg_pred = rfc_dfg.predict(df[dfg_train_cols])
  traj_dfg_prob = rfc_dfg.predict_proba(df[dfg_train_cols])

  # append 'dfg_conf' and probability data to traj frame data
  df['dfg_conf'] = traj_dfg_pred
  df['dfg_prob'] = np.max(traj_dfg_prob, axis=1)
  print(' \033[34mSK_RF Classify DFG:\033[0m   {:.6f} s'.format((time.perf_counter()-start)))

  ##### classify Chelix/DFG conformation of traj frames #####
  start = time.perf_counter()
  traj_full_pred = rfc_full.predict(df[full_train_cols])
  traj_full_prob = rfc_full.predict_proba(df[full_train_cols])
  print(' \033[34mSK_RF Classify Kinfo:\033[0m {:.6f} s'.format((time.perf_counter()-start)))

  ## append 'Class' and probability to traj frame data 
  start = time.perf_counter()
  df['dfg_conf']  = state_dfg(pd.DataFrame(traj_dfg_pred))
  df['Class']     = state_kinfo(pd.DataFrame(traj_full_pred))
  df['cidi_prob'] = traj_full_prob[:,0]
  df['cido_prob'] = traj_full_prob[:,1]
  df['codi_prob'] = traj_full_prob[:,2]
  df['codo_prob'] = traj_full_prob[:,3]
  df['wcd_prob']  = traj_full_prob[:,4]
  print(' \033[34mAppend Kinfo data:\033[0m    {:.6f} s'.format((time.perf_counter()-start)))

  df.index.name = 'frame'
  return df[Ref_Final_Cols]

################################################################
