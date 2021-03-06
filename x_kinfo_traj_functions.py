#!/usr/bin/env python3

import sys,os
import re
import bz2
import time
import pickle

import numpy as np
import pandas as pd
import mdtraj as md

from tqdm import tqdm
from pathos import multiprocessing
chunk = 25

##########################################################################
#
# this script contains functions to handle kinase structures and conformations
# that are used by other scripts.
#
#
##########################################################################
## Calculate the structural metrics using the list of coordinates
## Instead of using MPI looping over the coordinates, here use Dataframe + Numpy
## to 'Vectorize' the simple calculations (add, subtract, multiply, divide) to
## gain ~ 1,000-1,250x speed boost in comparsion to single processor operation.
## This vectorization can only be done to simple maths and cannot be done over
## complicated ones, e.g. 2nd order regression used in Chelix center determination.
## For Numpy operations such as dot and cross products, have to break down the 
## calculation back to they basic (+|-|x|/) operations.
def CalculateMetrics( Crd ):
  frames = len(list(Crd.coord_b3k.ca))
  ## predefine coord dataframe index/column (frame/atom) to maximize efficiency
  Cols = ['p1', 'p2', 'v3', 'cg_vec', 'ang_NHs', 'ang_CHs',
          'dist_NC', 'dist_NH', 'dist_CH', 'temp']
  m_df = pd.DataFrame(index=range(frames), columns=Cols)

  # to successfully put object of 'list of np.arrays' into rows of Dataframe cell,
  # convert the object into 'list' first, then into 'dictionary', then Dataframe
  d_df = pd.DataFrame({ 
    'dfg_d_ca': list(Crd.coord_dfg_d.ca), 'dfg_d_cb': list(Crd.coord_dfg_d.cb), 
    'dfg_d_cg': list(Crd.coord_dfg_d.cg), 'dfg_f_ca': list(Crd.coord_dfg_f.ca), 
    'dfg_f_cb': list(Crd.coord_dfg_f.cb), 'dfg_f_cg': list(Crd.coord_dfg_f.cg),
    'b3k_ca':   list(Crd.coord_b3k.ca),   'b3k_cb':   list(Crd.coord_b3k.cb), 
    'b3k_cg':   list(Crd.coord_b3k.cg),
    'c_glu_ca': list(Crd.coord_c_glu.ca), 'c_glu_cb': list(Crd.coord_c_glu.cb), 
    'c_glu_cg': list(Crd.coord_c_glu.cg), 'hlx_cent': list(Crd.coord_hlx.cn),
    }, index=range(frames) )

####################
  ## Using Dataframe vectorization is ~ 250x faster than MPIx4, ~ 1000x looping
  print('# Calculate C-Glu vector (cg_vec)...')
  start = time.perf_counter()
  m_df.temp = VecGen( list(d_df.c_glu_cg.to_numpy()), list(d_df.hlx_cent.to_numpy()) )
  m_df.cg_vec = m_df['temp'].div( VecMag( list(m_df['temp']) ) )
  end   = time.perf_counter()
  print(' {0:.1f} ms for {1} frames\n'.format((end-start)*1000, frames))

##################
  start = time.perf_counter()
  print('# Calculate DFG vectors (p1, p2, v3)...')
  Tmp = CalculateDFGVectors( 
                [ d_df.dfg_d_cg.to_numpy(), d_df.dfg_d_ca.to_numpy(), 
                  d_df.dfg_f_ca.to_numpy(), d_df.dfg_f_cg.to_numpy() ] )
  m_df.p1 = Tmp.p1
  m_df.p2 = Tmp.p2
  m_df.v3 = Tmp.v3

#####################
  print('# Calculate N-domain/C-helix angle (ang_NHs)...')
  m_df.ang_NHs = VectorAngle( [ d_df.hlx_cent.to_numpy(), d_df.b3k_ca.to_numpy(),
                                d_df.hlx_cent.to_numpy(), d_df.c_glu_cg.to_numpy() ])

#####################
  print('# Calculate C-domain/C-helix angle (ang_CHs)...')
  m_df.ang_CHs = VectorAngle( [ d_df.hlx_cent.to_numpy(), d_df.dfg_d_ca.to_numpy(),
                                d_df.hlx_cent.to_numpy(), d_df.c_glu_cg.to_numpy() ])

####################
  print('# Calculate N-/C-domain distance (dist_NC)...')
  m_df.dist_NC = Distance( d_df.b3k_ca.to_numpy(), d_df.dfg_d_ca.to_numpy() )

####################
  print('# Calculate N-domain/C-helix distance (dist_NH)...')
  m_df.dist_NH = Distance( d_df.b3k_cg.to_numpy(), d_df.c_glu_cg.to_numpy() )

####################
  print('# Calculate C-domain/C-helix distance (dist_CH)...')
  m_df.dist_CH = Distance( d_df.dfg_d_ca.to_numpy(), d_df.c_glu_ca.to_numpy() )
  end = time.perf_counter()
  print(' {0:.1f} ms for {1} frames\n'.format((end-start)*1000, frames))

  return m_df


##########################################################################
## read in trajectory using the supplied topology file (template pdb)
class ReadTraj(object):
  def __init__( self, top='' ):
    self.top = top

  def __call__( self, traj_file ):
    return md.load( traj_file, top=self.top )

#########
## each elements in this object will contain a list of coordinates
class TrajCoord(object):
  def __init__( self, ca=[], cb=[], cg=[], cn=[] ):
    self.ca = ca
    self.cb = cb
    self.cg = cg
    self.cn = cn

##########################################################################
## return an array of user-defined residue atoms, also check if residue is Gly,
## use average of HA2 and HA3 coords as CB and CG.
## selection with MDtraj, similar to VMD selection syntax
## For the collected coordinates, somehow MDTraj is missing by 1 decimal place
## and need to multiple the actual coords by a factor of 10
def SelectAtom( traj, resid, around=0, pkl=False ):

  Coord = TrajCoord()    # create object to hold atom coords

  ## if true, include extra residues before and after "resid"
  if around:
    resid = '{0} to {1}'.format(resid-around, resid+around)
  else:
    # md.Trajectory.xyz[frames, selection, atom-coordinates]
    select_ca = traj.top.select('resi {0} and name {1}'.format(resid, 'CA'))
    Coord.ca  = np.array(list(zip(*(10*traj.xyz[:, select_ca, :])))[0] )
#    print('ca', select_ca,'n',Coord.ca)

  ## check if residue is Glycine, single residue, or several residues
  if CheckResidue(traj.topology, resid) == 'helix':
    ## input is a set of 7-9 residues of C-helix centering on conserved Glu
    ## to calculate the helix axis curve, use center-most atoms as coordinates
    select_bb = traj.top.select('(resi {0}) and (name CA C N)'.format(resid))
    mid_atom  = ArrayCent(len(select_bb)) # center-most atom of helix atoms
#    print('bb', select_bb)
    Frames    = 10*np.array(traj.xyz[:, select_bb, :], dtype=np.float64)
    frames = len(Frames)

#################
    ## 2nd-order regression of C-helix, take mid-point as helix center coords

    ## ** Using chunks of 'apply' + MPI, it is no different than MPI alone
    ##    but faster than 'loops' and 'apply' alone
    ## Use map will improve performance over imap by ~8-10% due to mpi overhead?
    ## but with manual setting a fixed value of chunk size, imap perform ~ map
    print('# Calculate 2nd-order regression on C-helix for axis mid-point...')
    mpi       = multiprocessing.Pool()
    Reg2_list = [x for x in tqdm(mpi.imap(CalculateHelixAxis,Frames,chunk),total=frames)]

#    def CalculateHelixAxis_InParallel( chunk ):
#      chunk_df = pd.DataFrame( {'frame': list(chunk)} )
#      Reg2_list = chunk_df.apply(lambda row: CalculateHelixAxis(row['frame']), axis=1 )
#      return Reg2_list

#    ## Test case using 'apply+MPI' - multi CPU
#    start = time.perf_counter()
#    # determine chunk-size by number of CPU
#    f_chunks = np.array_split(Frames, multiprocessing.cpu_count())
#    print('# Calculate 2nd-order regression on C-helix for axis mid-point...')
#    Chunk_list = [x for x in tqdm(mpi.imap(CalculateHelixAxis_InParallel, f_chunks),total=len(f_chunks))]
#    Reg2_list = [item for sublist in Chunk_list for item in sublist]
#    end = time.perf_counter()
#    print(' {0:.1f} ms for {1} frames with MPI+Apply\n'.format((end-start)*1000, frames))

#    ## Test case using 'apply' only - single CPU
#    start = time.perf_counter()
#    df['reg2_a'] = df.apply(lambda row: CalculateHelixAxis(row['frame']), axis=1 )
#    Reg2_list = df['reg2_a'].to_list()
#    end = time.perf_counter()
#    print(' {0:.1f} ms for {1} frames with Apply\n'.format((end-start)*1000, frames))

#################     
    Coord.cn  = np.asarray([Reg2[mid_atom] for Reg2 in Reg2_list])
    mpi.close()
    mpi.join()

  ## most residues have 'CB' and 'CG', but some have branched 'CG*' and 'CD*'
  ## Branched: Val/Ile have 'CG*', Cys has 'SG', Thr has 'OG*|CG*'
  ## Met has 'SD', Asp/Asn have 'OD*|ND*', His/Phe/Tyr/Trp have 'CD*|ND*'
  ## if CD not available, use CG; if CG not available, use CB
  elif CheckResidue(traj.topology, resid) != 'GLY':
    select_cb = traj.top.select('resi {0} and name {1}'.format(resid,'CB'))
#    print('cb', select_cb)
    Coord.cb  = np.array(list(zip(*(10*traj.xyz[:, select_cb, :])))[0], dtype=np.float64)

    select_cg = traj.top.select('resi {0} and (name =~ "{1}")'.format(
                                resid, 'CG|OG|SG'))
    if len(select_cg) == 1:
      Coord.cg = np.array(list(zip(*(10*traj.xyz[:, select_cg, :])))[0], dtype=np.float64)
    elif len(select_cg) > 1:
      Frames   = 10*np.array(traj.xyz[:, select_cg, :], dtype=np.float64)
      Coord.cg = np.asarray(list(zip(*[np.mean(frame, axis=0) for frame in Frames]))[0])
    else:
      Coord.cg = Coord.cb
#    print('cg', select_cg, '\n', Coord.cg)

  else:   # if it is Glycine, use HA2 HA3 average as substitute of CB and CG
    topx, bonds = traj.topology.to_dataframe()
    print( ( topx[topx['resSeq']==resid+1] ) )
    select_h = traj.top.select('resi {0} and (name =~ "{1}")'.format(resid,'HA'))
    Frames   = 10*np.array(traj.xyz[:, select_h, :], dtype=np.float64)
    AvgTraj  = np.array([ list(np.mean(frame, axis=0)) for frame in Frames ])
    Coord.cb = AvgTraj
    Coord.cg = AvgTraj
    print(' * GLY{0} found, use HA* as CB/CG coords'.format(resid+1))

  return Coord


##########################################################################
## the numbers in topology record is same as user input (1-based), but actual
## mdtraj internal record (0-based) is 1 fewer than user input
def CheckResidue( top, resid ):

  ## if input is a set of residue, skip
  if re.search('to', str(resid)):
    print(resid, 'in GLU')
    return 'helix'

  top_df, bonds = top.to_dataframe()
  print(list(top_df[top_df['resSeq'] == resid+1]['resName'])[0], resid+1)

  if list(top_df[top_df['resSeq'] == resid+1]['resName'])[0] == 'GLY':
    return 'GLY'
  else:
    return 'other'


##########################################################################
## Object of a collection of key structural residues of each kinase frame
class CollectCoords(object):
  def __init__( self, coord_dfg_d=[], coord_dfg_f=[], coord_b3k=[],
                      coord_c_glu=[], coord_hlx=[] ):
    self.coord_dfg_d = coord_dfg_d
    self.coord_dfg_f = coord_dfg_f
    self.coord_b3k   = coord_b3k
    self.coord_c_glu = coord_c_glu
    self.coord_hlx   = coord_hlx

##########################################################################
## Extract coordinates of key structural residues of kinase and save as pickle
class ExtractCoords(object):
  def __init__( self, dfg='', b3k='', c_glu='', pkl=False ):
    self.dfg   = dfg
    self.b3k   = b3k
    self.c_glu = c_glu
    self.pkl   = pkl

##################
  def __call__( self, inp ):
    return self._extract_coords( inp )
    
  def _extract_coords( self, inp ):
    Crd = CollectCoords()

    ## extract traj-coordinates of each kinase residues, mdtraj uses 0-based ID
    Crd.coord_dfg_d = SelectAtom(inp, int(self.dfg)-1,   around=0)
    Crd.coord_dfg_f = SelectAtom(inp, int(self.dfg)+0,   around=0)
    Crd.coord_b3k   = SelectAtom(inp, int(self.b3k)-1,   around=0)
    Crd.coord_c_glu = SelectAtom(inp, int(self.c_glu)-1, around=0)
    Crd.coord_hlx   = SelectAtom(inp, int(self.c_glu)-1, around=4)

    if self.pkl and not os.path.isfile(self.pkl):
      with bz2.open(self.pkl, 'wb') as fo:
        pickle.dump(Crd, fo, protocol=pickle.HIGHEST_PROTOCOL)
        print('  ## INFO: Write structural coords to: {0}\n'.format(self.pkl))

    return Crd

##########################################################################
## Calculate the helix axis using coordinates supplied, calculate 2nd-order
## regression curves to represent helix axis.
def CalculateHelixAxis( Coords ):
#  print('Coords\n', Coords)
  count = len(Coords)
#  print(count)

  # Linear regression on Cartesian Coordinates to estimate helix axis coords.
  # Use moving sets of points on both end to average out regression error
  # iterate range(3) to calculate x,y,z coordinates separately.
  # Use minimium of 7 residues (> 21 points),
  if count >= 15:
    posit  = 6                # 3 atoms (N,C,CA) = 1 residue
  else:
    posit  = 1
  xcount = count - posit    # reduced number of points to do LSQ

  Fn2Pts = []
  for m in range(0,posit):
#    Fn2 = [LsqFit_1d(range(xcount), Coords[m:m-posit, x] ,2) for x in range(3)]
    Fn2 = LsqFit_nd(range(xcount), Coords[m:m-posit], 2)
    Fn2Pts.append( [np.asarray([f(x) for f in Fn2]) for x in range(count) ])
  Reg2 = np.mean(Fn2Pts, axis=0)
  return Reg2


##########################################################################
## Take in coordinates, calculate vectors among the coords, generate
## Cross-Products of the pairs
## Asp-CG (r1), Asp-CA (r2), Phe-CA (r3), Phe-CG (r4)
def CalculateDFGVectors( inp ):
  r1, r2, r3, r4 = inp
  x = len(r1)

  ## need to convert the *list* of vectors into dict before into Dataframe. 
  ## Cannot be np.array(vectors) but list(vectors) for Pandas to take the input
  vec = {
  'r21': r1 - r2,  # (AspCG - AspCA)      D (r1)
  'r23': r3 - r2,  # (AspCA - PheCA)       \______ (r3)
  'r32': r2 - r3,  # (PheCA - AspCA)      (r2)    \
  'r34': r4 - r3,  # (PheCG - PheCA)               F (r4)
  'temp1':  np.zeros(x), 
  'temp2':  np.zeros(x) }

  t_df = pd.DataFrame(vec, index=range(x))

  ur21 = t_df.r21.div( VecMag( list(t_df.r21) ) )   # univector of r21
  ur23 = t_df.r23.div( VecMag( list(t_df.r23) ) )
  ur32 = t_df.r32.div( VecMag( list(t_df.r32) ) )
  ur34 = t_df.r34.div( VecMag( list(t_df.r34) ) )

  ## To enable pandas vectorized calculation, transpose the Mx3 to 3xM format
  ## for calculation, which is later transposed back to Mx3 format when done
  ## Use of np.array(list(ur)).T is important; Transpose behaves differently with
  ## np.array(ur).T and ur.T
  # np.array(list(ur1))     -> array of list of lists of xyz   = array( 1 x M x 3 )
  # np.array(list(ur1)).T   -> array of list of xyz from lists = array( 1 x 3 x M )
  # np.array(ur1)           -> array of list of array of xyz   = array( 1 x M x 3 )
  # np.array(ur1).T         -> ** same as np.array(ur1)
  # ur1 / ur1.T             -> pd.Series of xyz                = ( M x 3 )
  t1 = {'temp1': list(VecCross( np.array(list(ur21)).T, np.array(list(ur23)).T ).T) }
  t2 = {'temp2': list(VecCross( np.array(list(ur34)).T, np.array(list(ur32)).T ).T) }
  t_df.temp1 = pd.DataFrame(t1)
  t_df.temp2 = pd.DataFrame(t2)

  y = { 'p1': t_df.temp1.div( VecMag(list(t_df.temp1)) ),  # univector for p1
        'p2': t_df.temp2.div( VecMag(list(t_df.temp2)) ),
        'v3': ur23 }                               # already univector for v3
  u_df = pd.DataFrame(y)

  return u_df


##########################################################################
## calculate structural vectors, and recycle the angular/distance metrics
def CompareMetrics( trj_df, ref_df_orig ):
  Cols = ['p1p1x', 'p2p2x', 'r3r3x', 'dfg_st', 'h_cgvc',
          'ang_NHs', 'ang_CHs', 'dist_NC', 'dist_NH', 'dist_CH']
  c_df = pd.DataFrame(index=range(len(trj_df)), columns=Cols)

  # for vectorization, make ref_df and trj_df same dimension
  ref_df = pd.DataFrame(np.repeat(ref_df_orig.to_numpy(), len(trj_df), axis=0))
  ref_df.columns = ref_df_orig.columns

  # transpose the input data for vectorization, but no need to transpose back
  # when exit - already in 1-D array format
  c_df.p1p1x  = VecDot( np.array(list(trj_df['p1'].to_numpy())).T, 
                        np.array(list(ref_df['p1'].to_numpy())).T )
  c_df.p2p2x  = VecDot( np.array(list(trj_df['p2'].to_numpy())).T, 
                        np.array(list(ref_df['p2'].to_numpy())).T )
  c_df.r3r3x  = VecDot( np.array(list(trj_df['v3'].to_numpy())).T, 
                        np.array(list(ref_df['v3'].to_numpy())).T )
  c_df.h_cgvc = VecDot( np.array(list(trj_df['cg_vec'].to_numpy())).T, 
                        np.array(list(ref_df['cg_vec'].to_numpy())).T )
  c_df.dfg_st = DFGState( c_df.p1p1x.to_numpy(), c_df.p2p2x.to_numpy() )

  c_df.ang_NHs = trj_df.ang_NHs
  c_df.ang_CHs = trj_df.ang_CHs
  c_df.dist_NC = trj_df.dist_NC
  c_df.dist_NH = trj_df.dist_NH
  c_df.dist_CH = trj_df.dist_CH

  return c_df

###############################################################################
def DFGState( x, y ):
  ## Model PDB has same DFG- config as template DFG-in:     'in'
  ## Model PDB has opposite DFG- config as template DFG-in: 'out'
  ## Model PDB has undefined DFG- config:                   'random'

  ## predefine a table of value for Dataframe
  ## newer numpy vectorized method, faster than _conditions by 125-1000x
  dfg_in  = (x > 0.005) & (y > 0.050)
  dfg_out = (x <-0.125) & (y <-0.125)
  dfg = pd.DataFrame({ '0': ['other']*len(x) })
  dfg[dfg_in[0]  == True] = 'in'
  dfg[dfg_out[0] == True] = 'out'
  return dfg['0'].to_numpy()

#############
## old, original non-vectorized conditions method
def _conditions( df ):
    if df.p1p1x is None or df.p2p2x is None:
      return 'missing DFG'
    elif df.p1p1x > 0.005 and df.p2p2x > 0.05:
      return 'in'
    elif df.p1p1x < -0.125 and df.p2p2x < -0.125:
      return 'out'
    else:
      return 'weird'

#  df.dfg_st = df.apply(_conditions, axis=1)    # slower by 100x


##########################################################################
## Magnitude of a vector, equivalent to np.linalg.norm( v )
## this algorithm is much faster than np.linalg.norm function
def VecMag( v ):
  return np.sqrt((np.asarray(v)**2).sum(-1))

#################
## Distance between 2 points for pandas vectorization
def Distance( a, b ):
  return VecMag( list( VecGen( a, b ) ) )

#################
# ## Generate a unit vector from 2 coordinates for pandas vectorization
def VecGen( a, b ):
  return ( np.array(b) - np.array(a) )

#################
## Cross product in the most basic form for pandas vectorization
def VecCross( a, b ):
  c = np.array( [ a[1]*b[2] - a[2]*b[1],
                  a[2]*b[0] - a[0]*b[2],
                  a[0]*b[1] - a[1]*b[0]  ] )
  return c

#################
## Dot product in the most basic form for pandas vectorization
def VecDot( a, b ):
#  print('a\n', a, '\n', 'b', '\n', b)
  return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

#################
## Angle between vectors for pandas vectorization
def VectorAngle( inp ):
  r1, r2, r3, r4 = inp
  
  o_df = pd.DataFrame(index=range(len(r1)), columns=['v1', 'v2', 'ang'])

  o_df.v1 = ( VecGen( r1, r2 ) )   # df of v1 vector
  o_df.v2 = ( VecGen( r3, r4 ) )

  uv1 = o_df.v1.div( VecMag( list(o_df.v1)) )   # df of v1 univector
  uv2 = o_df.v2.div( VecMag( list(o_df.v2)) )
  
  ## To enable pandas vectorized calculation, transpose the Mx3 to 3xM format
  ## for calculation, which is later transposed back to Mx3 format when done
  dot = VecDot( np.array(list(uv1)).T, np.array(list(uv2)).T ).transpose()
  o_df.ang = np.arccos( dot ) * 180/np.pi
  return o_df.ang

#################
## n-order Least-square fit of 2 arrays, x and y, return an object/def as
## a nst-order polynomial function f(x) = mx+c, or f(x) = mx^2+nx+c, etc
## i.e.                            f(1) = 15       f(5) = 25
## polyfit of single nd-array seems slightly faster than polyfit of multiple
## 1d-array, probably due to numpy's vectorization
def LsqFit_1d( X, Y, order=1 ):
  return np.poly1d( np.polyfit( X, Y, order, full=False)) # for single polyfit

def LsqFit_nd( X, Y, order=1 ):
  Fits = list( zip(*np.polyfit( X, Y, order, full=False )) )
  return [ np.poly1d(coeff) for coeff in Fits ]           # for multiple polyfit

#################
# Find the center element in an array. The result is 1 less than correct
# answer -- Python numbering starts from 0. i.e. 9-element array is [0,...,8]
# center is 4, not 5
def ArrayCent( count ):
  if count % 2 == 0:
    center = (count/2)-1
  else:
    center = (count-1)/2
  return int(center)


########################################################################
## Normalize ang_ and dist_ data with vectorization, same result as R's
## clusterSim data.Normalization(input, type='n5', normalization='column')
## If no normalized_factor is supplied, this will assume to calculate the
## normalization factor and write it to a csv file, place it in z_database
def Normalization( data, norm_file='', norm_param='' ):

  if not len(norm_param):
    cb_mean = data.to_numpy().mean(axis=0)
    cb_vars = data.to_numpy() - cb_mean
    cb_max  = np.max(np.abs(cb_vars), axis=0) 

    ## write normalization factor to z_database
    df = pd.DataFrame( [cb_mean,cb_max] ).T
    df.columns = ['mean','max']
    df.to_csv(norm_file, index=None, sep=',')

  else:
    cb_vars = data.to_numpy() - norm_param['mean'].to_numpy()
    cb_max  = norm_param['max'].to_numpy()

  return cb_vars/cb_max  # (var-mean)/max(abs(var-mean))


########################################################################
## set DFG conformation type based on DFG in/out with vectorized T/F
## Pandas based and use half with numpy vectorization to give ~ 10x speedup
## old half-vectorized ~ 1.2s for 3800 items, full-vectorized ~5ms, ~240x speedup
def dfg_state( conf ):
  conf_di = (conf == 'cidi') | (conf == 'codi')
  conf_do = (conf == 'cido') | (conf == 'codo')
  state = pd.DataFrame({ '0': [2]*len(conf) }) # 'interm' has '2'
  state[conf_di == True] = 0  # any DI is '0'
  state[conf_do == True] = 1  # any DO is '1'
  return state['0'].to_numpy()

# current and older way to define DFG state. Older way is much slower
def state_dfg( state ):
  conf_di = (state == 0)
  conf_do = (state == 1)
  conf = pd.DataFrame({ '0': ['other']*len(state) })
  conf[conf_di[0].to_numpy() == True] = 'di' # any DI is '0'
  conf[conf_do[0].to_numpy() == True] = 'do' # any DO is '1'
  return conf['0'].to_numpy()

def state_dfg_old( state ):
  conf_di = (state == 0)
  conf_do = (state == 1)
  conf = ['other']*(len(state))   # other has '2'
  for i in range(len(state)):
    if conf_di.iloc[i][0]: conf[i] = 'di'  # any DI has '0'
    if conf_do.iloc[i][0]: conf[i] = 'do'  # any DO has '1'
  return conf

#################

def kinfo_state( conf ):
  conf_cidi = (conf == 'cidi')
  conf_cido = (conf == 'cido')
  conf_codi = (conf == 'codi')
  conf_codo = (conf == 'codo')
  state = pd.DataFrame({ '0': [4]*len(conf) }) # 'wcd' has '4'
  state[conf_cidi == True] = 0  # any DI is '0'
  state[conf_cido == True] = 1  # any DO is '1'
  state[conf_codi == True] = 2  # any DI is '0'
  state[conf_codo == True] = 3  # any DO is '1'
  return state['0'].to_numpy()

################
def state_kinfo( state ):
  conf_cidi = (state == 0)
  conf_cido = (state == 1)
  conf_codi = (state == 2)
  conf_codo = (state == 3)
  conf = pd.DataFrame({ '0': ['wcd']*len(state) })
  conf[conf_cidi[0].to_numpy() == True] = 'cidi'  # ci DI is '0'
  conf[conf_cido[0].to_numpy() == True] = 'cido'  # ci DO is '1'
  conf[conf_codi[0].to_numpy() == True] = 'codi'  # co DI is '2'
  conf[conf_codo[0].to_numpy() == True] = 'codo'  # co DO is '3'
  return conf['0'].to_numpy()


##########################################################################
