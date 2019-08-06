#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 18:52:26 2019

@author: pmung
"""
import sys,os
import re
import bz2
import time
import pickle
import sklearn
import numpy as np
import pandas as pd
import mdtraj as md

from tqdm import tqdm
from pathos import multiprocessing
from argparse import ArgumentParser

from x_kinfo_traj_functions import ReadTraj
from x_kinfo_traj_functions import TrajCoord
from x_kinfo_traj_functions import ArrayCent
from x_kinfo_traj_functions import CollectCoords
from x_kinfo_traj_functions import ExtractCoords
from x_kinfo_traj_functions import CompareMetrics
from x_kinfo_traj_functions import CalculateMetrics
from x_kinfo_traj_functions import CalculateHelixAxis
from x_kinfo_traj_functions import CalculateDFGVectors
from x_kinfo_traj_functions import VecMag, VecGen, Distance, VectorAngle

from x_kinfo_SK_modelgen import Normal_Param
from x_kinfo_SK_classify import KinfoClassify

#print(np.__version__)       # stable: 1.16.2
#print(pd.__version__)       # stable: 0.24.2
#print(md.__version__)       # stable: 1.9.3
#print(sklearn.__version__)  # stable: 0.20.3
np.seterr(invalid='ignore')

lib_dir = '/Users/pmung/Dropbox (Schlessinger lab)/9_scripts/3_program/structures/4_Konformation/z_database/'
print('##### need to change the hard-coded library/repository directory #####')
print('current lib_dir: '+lib_dir+'\n')

sk_ml = ['rf','svm','nn','dt','kn','gb', 'gp']

##########################################################################
def main():
  args = UserInput()
  
  if args.use_sk not in sk_ml:   # default SK model: RandomForest
    args.use_sk = 'rf'

## These are hard-coded test cases
#  wrk_dir = '/Users/pmung/Dropbox (Schlessinger lab)/z_others/8_strad/'
#  args.tmpl_file = wrk_dir+'2_md/strada_cido.prot.1atp.pdb'
#  args.traj_file = wrk_dir+'2_md/strada_cidi.2.200ps.dcd'
#  args.outpref   = 'test'
#  args.b3k = 39
#  args.dfg = 152
#  args.c_glu = 57

  ## user-defined library path
  if args.lib_dir:
    lib_dir = args.lib_dir

  ## reference structure must start at resid 1. Modified ref is hardcoded here
  if not os.path.isfile(lib_dir+'1ATP.mdtraj.pdb'):
    sys.exit('\n    ERROR: Reference structure "1ATP.mdtraj.pdb" not found\n'+
             lib_dir+'1ATP.mdtraj.pdb')
  else:
    ref_file = lib_dir+'1ATP.mdtraj.pdb'
    ref_pkl  = lib_dir+'1ATP.mdtraj.pkl.bz2'
    ref_dfg  = 171
    ref_b3k  = 59
    ref_c_glu= 78

######################

  ## get reference PDB structure 1ATP.pdb coordinates dataframe
  print('# Reading in reference file: '+ref_file)
  if not ref_pkl or not os.path.isfile(ref_pkl):
    ref    = md.load_pdb(ref_file)
    ref_cd = ExtractCoords(dfg=ref_dfg, b3k=ref_b3k, c_glu=ref_c_glu, pkl=ref_pkl)
    ref_df = CalculateMetrics( ref_cd(ref) )
  ## skip calculation if data is already stored in pickle
  else:
    print('  ## INFO: Read structural residue coords from: {0}\n'.format(ref_pkl))
    with bz2.open(ref_pkl, 'rb') as fi:
      ref = pickle.load(fi)
    ref_df = CalculateMetrics( ref )

######################
  ## load trajectory file(s) with MDtraj, can be multiple traj files at once
  traj = []
  print('# Reading in trajectory file(s)...')
  start = time.perf_counter()
  if not args.pkl or not os.path.isfile(args.pkl):
    start2= time.perf_counter()
    TrjIn = ReadTraj(top=args.tmpl_file) 
    if re.search(r'dcd$|nc$|crd$|xtc$', args.traj_file):
      traj = TrjIn(args.traj_file)
    else:
      traj_list = filter(None, (l.rstrip() for l in open(args.traj_file, 'r')
                                if l is not re.search(r'^#', l)))
      mpi  = multiprocessing.Pool(processes = multiprocessing.cpu_count())
      traj = md.join( mpi.imap(TrjIn, traj_list,2) )
      mpi.close()
      mpi.join()
    end2 = time.perf_counter()
    print('  ## Time to load trajectory: {0:.1f} ms for {1} frames\n'.format(
             (end2-start2)*1000, len(traj)) )

  ## superpose all frames to template structure pre-superposed to ref 1ATP.pdb
    if args.superp:
      print('# Applying superposition to trajectory with: '+args.superp)
      tmpl = md.load_pdb(args.tmpl_file)
      traj = traj.superpose(tmpl, atom_indices=args.superp, parallel=True)

    ## get trajectory coordinates dataframe
    print('# Extracting structural matrics from trajectory...')
    start  = time.perf_counter()
    trj_cd = ExtractCoords(dfg=args.dfg, b3k=args.b3k, c_glu=args.c_glu, pkl=args.pkl)
    trj_df = CalculateMetrics( trj_cd(traj) )
         
  ## skip calculation if data is already stored in pickle
  else:
    print('  ## INFO: Read structural residue coords from: {0}\n'.format(args.pkl))
    with bz2.open(args.pkl, 'rb') as fi:
      trj_df = CalculateMetrics( pickle.load(fi) )

  end    = time.perf_counter()
  print('## Total time to get traj descriptors: {0:.1f} ms for {1} frames'.format(
        (end-start)*1000, len(trj_df)))
  del traj    # save memory
  print('\n#########################################\n')

######################
######################
  ## calculate structural metrics from coordinates, then print out raw output
  print('# Calculating structural matrics from coordinates...')
  start  = time.perf_counter()
  mat_df = CompareMetrics(trj_df, ref_df)

  mat_df.to_csv(args.outpref+'.csv', sep=',')
  end    = time.perf_counter()
  print('## Total time to compare descriptors: {0:.1f} ms for {1} frames'.format(
        (end-start)*1000, len(mat_df)))
  print('\n#########################################\n')


#####################
  ## use Kinformation Random Forest Classifier to assign conformation/confidence
  start = time.perf_counter()
  KinfoClassify(mat_df, lib_dir, args.outpref, args.use_r_rf, args.use_sk)
  end    = time.perf_counter()
  print('\n## Total time to SK {0} Classification: {1:.3f} ms for {2} frames'.format(
        args.use_sk, (end-start)*1000, len(mat_df)))

  print('\n#########################################\n')



##########################################################################
def UserInput():
  p = ArgumentParser(description='Command Line Arguments')

  p.add_argument('-templ', dest='tmpl_file', required=True,
                 help='Template PDB structure (exact match to Topology Atom List and aligned to Ref structure 1ATP)')
  p.add_argument('-traj', dest='traj_file', required=True,
                 help='Trajectory file, or an ordered list of traj filenames')
  p.add_argument('-out', dest='outpref', required=True,
                 help='Output prefix')
  p.add_argument('-b3k', dest='b3k', required=True,
                 help='(beta-3 Lys) Residue Number in Template Structure')
  p.add_argument('-dfg', dest='dfg', required=True,
                 help='(DFG Asp) Residue Number in Template Structure')
  p.add_argument('-glu', dest='c_glu', required=True,
                 help='(C-helix Glu) Residue Number in Template Structure')

  p.add_argument('-pkl', dest='pkl', required=False,
                 help='Use pre-pickled trajectory data generated from previous run (def: False)')

  p.add_argument('-superp', dest='superp', required=False,
                 help='*Optional: VMD-like selection string to perform superposition (default: False)')

  p.add_argument('-use_r_rf', action='store_true',
                 help='Use R::randomForest instead of SKLearn RFClassifier (def: None)')
  p.add_argument('-use_sk', dest='use_sk', required=False,
                 help='Use SKLearn ML model: rf|svm|nn|kn|dt|gp|gb (def: rf)')

  p.add_argument('-lib', dest='lib_dir', required=False, 
                 help='Kinformation_MD Repository database path (unless hard-coded)')

  args=p.parse_args()
  return args

##########################################################################
if __name__ == '__main__':
  main()

##########################################################################
#
# Peter M.U. Ung @ MSSM/Yale
#
# v1.0  19.05.06  finished reading and calculating traj coodinates
#                 optimized with DataFrame/Numpy vectorization and MPI
# v2.0  19.05.18  implement R RandomForest classifier to assign conformation
# v3.0  19.05.28  add RandomForest conformation classification and rearrange
#                 definitions to other scripts
#
# e.g.> x.py -templ strada_cido.prot.1atp.pdb -traj strada_cidi.2.200ps.dcd 
#            -b3k 39 -dfg 152 -glu 57 -out test
#            -superp 'resid 20 to 50 and 100 to 200'
#            -use_sk svm
#            -lib '/Users/xxx/scripts/Kinformation_MD/z_database'
