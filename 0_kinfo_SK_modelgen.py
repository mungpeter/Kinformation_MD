#!/usr/bin/env python3

## this code use Jupyter/iPython to generate the initial
## SKLearn ML Model sequentially, which is then used by 
## 1_kinfo_traj_MD.py and other scripts. This model generation
## step is a one-time thing unless more the training set of the
## kinase conformations is changed 

import sys,os
import bz2
import time
import pickle
import numpy as np
import pandas as pd

from numpy import random

from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.ensemble import AdaBoostClassifier   # pretty bad result
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier

from x_kinfo_R_classify import R_Impute
from x_kinfo_traj_functions import Normalization
from x_kinfo_traj_functions import state_dfg,dfg_state
from x_kinfo_traj_functions import state_kinfo,kinfo_state

from x_kinfo_variables import KinfoVariables
from x_kinfo_variables import SKLearnDFGModelFiles
from x_kinfo_variables import SKLearnKinfoModelFiles

#############################################################################
def SK_GenerateMLModels():

  ## Parameters used for ML model generation
  Ref_Matrx_cols= [ 'Group','training','p1p1x','p2p2x','r3r3x','h_cgvc',
                    'ang_NHs','ang_CHs','dist_NH','dist_CH']

  Ref_Train_Cols= [ 'Group','training',
                    'p1p1x','p2p2x','r3r3x','h_cgvc','ang_NHs','ang_CHs',
                    'dist_NH','dist_CH',]

  Ref_Test_Cols = [ 'p1p1x','p2p2x','r3r3x','h_cgvc','ang_NHs','ang_CHs',
                    'dist_NH','dist_CH']

  Ref_Final_Cols= [ 'Class','training',
                    'cidi_prob','cido_prob','codi_prob','codo_prob','wcd_prob',
                    'dfg_conf','dfg_prob','p1p1x','p2p2x','r3r3x','h_cgvc',
                    'ang_NHs','ang_CHs','dist_NH','dist_CH']

  norm_cols = ['ang_NHs','ang_CHs','dist_NH','dist_CH']

  dfg_train_cols = ['p1p1x','p2p2x','r3r3x','dist_NH','dist_CH']
  full_train_cols= ['h_cgvc','ang_NHs','ang_CHs','dist_NH','dist_CH','dfg_conf']

############
  sk_dfg_models = SKLearnDFGModelFiles()
  sk_chx_models = SKLearnKinfoModelFiles()

  Vars = KinfoVariables()
  lib_dir            = Vars['lib_dir']
  home_dir           = Vars['home_dir']
  kinfo_data         = Vars['kinfo_data']
  kinfo_rf_data      = Vars['kinfo_rf_data']
  kinfo_rf_data_norm = Vars['kinfo_rf_data_norm']
  kinfo_norm_param   = Vars['kinfo_norm_param']


##########################################################################
  os.chdir(home_dir)
  print('\033[34m# Directory:\033[0m]\n {0}'.format(os.getcwd()))

############## Run this to generate SKL RandomForest Models ##############

  ## Read in the starting kinformation kinase-structural data file
  ## Usually there is a duplication of '1ATP_E' and '1atp' as internal check
  data_df  = pd.read_csv('{0}/{1}'.format(lib_dir,kinfo_data), sep=',', index_col=0)
  matrx_df = data_df[Ref_Matrx_cols].drop(['1atp_E'], axis=0) # internal check
  data_df[:5]
  matrx_df[:5]

##################################
  ## Training set, usually the top 325 lines of strutures with manually annotated conformations
  train_x  = PrepareTrainingSet( True, matrx_df, Ref_Train_Cols ); train_x[:5]
  test_x   = matrx_df[pd.isna(matrx_df.Group) & pd.isna(matrx_df.training)]
  test_x   = test_x.dropna(subset=Ref_Test_Cols)    # 3286 lines
  ## Complete dataset from 171009, dropped all NaN row, 3611 lines
  complete = pd.concat([train_x, test_x]); complete[:5]

  ## Save pre-normalized data for kinformation RF generation as backup
  complete.to_csv(kinfo_rf_data, compression='gzip', sep=',')
  complete = pd.read_csv('{0}/{1}'.format(lib_dir,kinfo_rf_data), sep=','); complete[:5]

  with open('{0}/{1}'.format(lib_dir, kinfo_norm_param), 'rb') as fi:
    norm_param = pickle.load(fi)

  norm_data = Normalization(complete[norm_cols], norm_param=norm_param)
  complete[norm_cols] = norm_data

  ## Save/load normalized data for kinformation RF generation
  complete.to_csv(kinfo_rf_data_norm, compression='gzip', sep=',')
  complete = pd.read_csv('{0}/{1}'.format(lib_dir,kinfo_rf_data_norm), sep=','); complete[:5]

  train_df = complete[ :len(train_x)]
  test_df  = complete[len(train_x): ]

  ## RandomForest: 'rf'; SVM: 'svm'; NeuralNet: 'nn'; DecisionTree: 'dt'
  ## GradientBoost: 'gb'; GaussianProcess: 'gp'; K-NearestNeighbors: 'kn'
  ml_alg    = 'rf'
  ml_models = SK_TrainML( train_df, lib_dir, ml_alg, save_model=False )
  result_df = SK_RunML( complete, lib_dir, ml_alg, models=ml_models )

  result_df[:10]
  result_df.to_csv('kinfo_{0}_conf_assign_result.csv'.format(ml_alg), sep=',')


##########################################################################

def SK_TrainML( df, lib_dir, ml_alg, save_model=False ):
#  df=train_df
  df.drop('training', axis=1)
  df['dfg_conf'] = dfg_state(df.Group.to_numpy())
  random.seed(0)    # set random number

  if ml_alg == 'rf':
    rfc_dfg = RandomForestClassifier( n_estimators=1000, bootstrap=True, random_state=0, n_jobs=-1  )
    rfc = RandomForestClassifier( n_estimators=1000, bootstrap=True, random_state=0,  n_jobs=-1 )
  if ml_alg == 'svm':
    rfc_dfg = SVC( kernel='rbf', decision_function_shape='ovo', probability=True, random_state=0 )
    rfc = SVC( kernel='linear', decision_function_shape='ovo', probability=True, random_state=0 )
  if ml_alg == 'nn':
    rfc_dfg = MLPClassifier( activation='relu', solver='adam', max_iter=500, random_state=0 )
    rfc = MLPClassifier( activation='relu', solver='adam', max_iter=500, random_state=0 )
  if ml_alg == 'dt':
    rfc_dfg = DecisionTreeClassifier( random_state=0 )
    rfc = DecisionTreeClassifier( random_state=0 )
  if ml_alg == 'gb':
    rfc_dfg = GradientBoostingClassifier( n_estimators=100, random_state=0 )
    rfc = GradientBoostingClassifier( n_estimators=100, random_state=0 )
  if ml_alg == 'gp':
    rfc_dfg = GaussianProcessClassifier(optimizer='fmin_l_bfgs_b',n_restarts_optimizer=3,random_state=0,n_jobs=-1)
    rfc = GaussianProcessClassifier(optimizer='fmin_l_bfgs_b',n_restarts_optimizer=3,random_state=0,n_jobs=-1)
  if ml_alg == 'kn':
    rfc_dfg = KNeighborsClassifier( n_neighbors=5, algorithm='auto' )
    rfc = KNeighborsClassifier( n_neighbors=15, algorithm='auto' )


  ## select the attribute columns to be used, and the label column to fit to
  df_train, df_test = train_test_split(df, train_size=0.8, random_state=1, shuffle=True)
  dfg_train_attri = df_train[dfg_train_cols]
  dfg_train_label = df_train.dfg_conf # as factors
  dfg_test_attri  = df_test[dfg_train_cols]
  dfg_test_label  = df_test.dfg_conf

  #### train DFG model on data to match 'dfg_conf' 3 states
  rfc_dfg.fit(dfg_train_attri, dfg_train_label)
  dfg_test_pred = rfc_dfg.predict(dfg_test_attri)
  EvaluatePerformance(rfc_dfg, dfg_test_label, dfg_test_pred, dfg_train_cols)


  ## train on the DFG/Chelix, add 'dfg_conf' according to Group, factor 'Group'
  chx_train_attri = df_train[full_train_cols]  # include dfg_conf ~from dfg predict
  chx_train_label = kinfo_state(df_train.Group.to_numpy())
  chx_test_attri  = df_test[full_train_cols]
  chx_test_label  = kinfo_state(df_test.Group.to_numpy())

  #### train DFG/Chelix model on data including 'dfg_conf' to match 'Group' 5 states
  rfc.fit(chx_train_attri, chx_train_label)
  chx_test_pred = rfc.predict(chx_test_attri)
  EvaluatePerformance(rfc, chx_test_label, chx_test_pred, full_train_cols)

  ## save the models into pickle
  if save_model:
    with bz2.open(sk_dfg_models[ml_alg], 'wb') as fd:
      pickle.dump(rfc_dfg, fd, protocol=pickle.HIGHEST_PROTOCOL)
    with bz2.open(sk_chx_models[ml_alg], 'wb') as fc:
      pickle.dump(rfc,     fc, protocol=pickle.HIGHEST_PROTOCOL)
  return [rfc_dfg, rfc]


##########################################################################

## get the training set with manual annotation
def PrepareTrainingSet( r_impute, matrx_df, Ref_Train_Cols ):
  print('\033[34m## Cleaning up Training Set...\033[0m')
  ## parse out training set and assign factorized 'dfg_conf' based on 'Group'
  df = matrx_df[ pd.notna(matrx_df.Group) & pd.notna(matrx_df.training) ]

  ## impute data by their subset ('Group')
  print('\033[34m## Imputing Training Set...\033[0m')
  ## skip if no NA is found in dataset
  if not df.isnull().values.any():
    print(' INFO: no missing data for imputing.')
    clean_df = df
  else:
    # Use R::randomForest::impute if True, otherwise use SKlearn impute
    if r_impute:
      df = df.drop(['training'], axis=1)
      clean_df = R_Impute( df, kinfo_state(df.Group.to_numpy()) )
    else:
      ## detect/remove rows in Group with Null; replace np.nan wtih 'NaN'
      ## Imputer need manual separation of groups for better imputing
      df = df.replace(np.nan, 'NaN').drop(['training'], axis=1)
      clean_df = pd.concat( [ SK_Impute(df[df.Group == conf], conf, 'Group', 1) 
                              for conf in set(df.Group) ] )

  ## Rearrange columns to have consistent order
  train_df = clean_df[Ref_Train_Cols]

  print('\033[34m## Completed Training Set Imputing. Check for remaining NULL...\033[0m')
  if sum(train_df.isnull().sum()):
    print('  \033[31mFATAL: Cannot complete Training Set Imputation, there are NULL: \033[0m')
    sys.exit(train_df.isnull().sum())
  else:
    print(' -- No NULL data --')
  return train_df


#######################################################################
## Unlike R-randomForest rfImput function, which takes "responses" as a
## factor to automatically subset the data to impute separately, sklean 
## Imputer does a straight impute and average ALL data in the column 
## regardless their "responses", i.e. cidi and cido's missing DFG vectors
## will be the averaged of both sets. Need to manually separate the subsets
## and impute them
def SK_Impute( df, conf, coln, coli ):
  imputer = SimpleImputer(missing_values='NaN', strategy='median', axis=0)
  Ip_List = imputer.fit_transform( df.iloc[:, coli:] )
  i_df    = pd.DataFrame(Ip_List, index=df.index, columns=df.columns[coli:])
  return pd.concat([df[coln], i_df], axis=1)


###########################################################################
## Perform matrics calculation to evaluate model performance
def EvaluatePerformance( model, test, predict, Cols ):

  print('\033[34m### Evaluate SKlearn ML Model Performance ###\033[0m')
  print(' # Confusion Matrix:')
  print(confusion_matrix(test, predict))
  print('\n# Mean Squared Error:')
  print(classification_report(test, predict))
  print('\n# Accuracy Score - Out-of-bag Error:')
  a_score = accuracy_score(test, predict)
  print('{0:.3f} %  -  {1:.3f} %\n'.format(a_score*100, (1-a_score)*100))
  try:
    features = model.feature_importances_
    print('\033[34m# Feature importance for RandomForest:\033[0m')
    for idx, importance in enumerate(features):
      print(' {0:10s} - {1:.2f}'.format(Cols[idx], importance*100))
  except AttributeError:
    return None


##########################################################################
def SK_RunML( traj, lib_dir, ml_alg, models='' ):

  ## load in RF models if it is not generated on the fly
  if not models:
    print('\033[34m## INFO: Loading trained SK ML models...\033[0m')
    with bz2.open(lib_dir+sk_dfg_models[ml_alg], 'rb') as fd:
      rfc_dfg = pickle.load(fd)
    with bz2.open(lib_dir+sk_chx_models[ml_alg], 'rb') as fc:
      rfc = pickle.load(fc)
  else:
    rfc_dfg, rfc = models

  training_df = traj.training
  traj.drop('training', axis=1)   # prevent interference

  ##### classify DFG conformation of trajectory frames #####
  start = time.perf_counter()
  traj_dfg_pred = rfc_dfg.predict(traj[dfg_train_cols])
  traj_dfg_prob = rfc_dfg.predict_proba(traj[dfg_train_cols])

  # append 'dfg_conf' and probability data to traj frame data
  traj['dfg_conf'] = traj_dfg_pred
  traj['dfg_prob'] = np.max(traj_dfg_prob, axis=1)
  print(' \033[32mdfg:\033[0m {:.6f} s'.format((time.perf_counter()-start)))

  ##### classify Chelix/DFG conformation of traj frames #####
  start = time.perf_counter()
  traj_full_pred = rfc.predict(traj[full_train_cols])
  traj_full_prob = rfc.predict_proba(traj[full_train_cols])
  print(' \033[32mfull:\033[0m {:.6f} s'.format((time.perf_counter()-start)))

  ## append 'Class' and probability to traj frame data 
  start = time.perf_counter()
  traj['dfg_conf']  = state_dfg(pd.DataFrame(traj_dfg_pred))
  traj['Class']     = state_kinfo(pd.DataFrame(traj_full_pred))
  traj['cidi_prob'] = traj_full_prob[:,0]
  traj['cido_prob'] = traj_full_prob[:,1]
  traj['codi_prob'] = traj_full_prob[:,2]
  traj['codo_prob'] = traj_full_prob[:,3]
  traj['wcd_prob']  = traj_full_prob[:,4]
  traj['training']  = training_df
  print(' \033[32madd:\033[0m {:.6f} s'.format((time.perf_counter()-start)))

  return traj[Ref_Final_Cols]


#######################################################################
## Load the trajectory structural data and reorder the columns
def PrepareTestSet( args, Ref_Test_Cols ):
  test_df = pd.read_csv(args.test, header=True, delimiter=',')

  ## make sure the arrangement of columns is consistent
  try:
    test_df = test_df[Ref_Test_Cols]
  except IndexError:
    print('\033[31m# ERROR: expected input column name and order:\033[0m')
    print(Ref_Test_Cols)
    print(test_df.columns)
    sys.exit('\033[31m# FATAL: Input dataset columns not matching\033[0m')
  
  return test_df
  

########################################################################
