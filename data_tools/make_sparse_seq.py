import glob, csv, numpy as np


for seq in glob.glob("/ssd_disk/gay/scenegraph/seqs/*.seq"):
  seq_dict={}
  for (fnum,fname,olabel,shapelabel,vglabel,oid,oc,x1,y1,x2,y2)  in list(csv.reader(open(seq), delimiter= ' ')):
    if int(fnum) not in seq_dict:
      seq_dict[int(fnum)]=[]
    seq_dict[int(fnum)].append((fnum,fname,olabel,shapelabel,vglabel,oid,oc,x1,y1,x2,y2))
  ks=sorted(seq_dict.keys())
  if len(ks)>10:
    idx=np.round(np.linspace(0,len(ks)-1,10)).astype(int)
    ks=np.array(ks)[idx]
  fn=seq.split('/')[-1]
  f=open('/ssd_disk/gay/scenegraph/sparse_seqs/'+fn,'w')
  for k in ks:
    fields_list = seq_dict[k]
    for fields in fields_list:
      f.write(' '.join(fields)+'\n')  
  f.close()
