
# because -traj would require a large .dcd/.nc file, here use -pkl instead

../1_kinfo_traj_MD.py              \
  -templ strada_cido.prot.1atp.pdb \
  -pkl   traj_cido.pkl.bz2         \
  -b3k 39 -dfg 152 -glu 57         \
  -out test.200228                 \
  -superp 'resid 100 to 200 and 250 to 300' \
  -use_sk nn                       \
  -lib   /home/pmung/Dropbox/9_scripts/3_program/structures/4_Kinformation_MD/z_database/
