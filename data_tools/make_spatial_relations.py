import numpy as np, csv, glob, argparse, utils_prep

parser = argparse.ArgumentParser(description="Make the geometrical relations from the 3D bounding boxes")
parser.add_argument('-d','--datadir', default="/media/scannet/scannet/SensReader/")
parser.add_argument('-l','--seq_dir', help="file containing on each line the id of a scene, it should also correspond to the name of its directory")
parser.add_argument('--rel_file')
parser.add_argument('-o','--obj_file')
args = parser.parse_args()

(seq2im, im2roi) = utils_prep.read_seq(args.seq_dir, args.datadir)
objs_all = utils_prep.read_orientated_obj(args.obj_file)
f=open(args.rel_file,'w')
for ((seq_name,im_f),objs_seq) in im2roi.items():
  scene = utils_prep.seq2scene(seq_name)
  #cam_f = im_f.replace('color.jpg','pose.txt')
  #world2cam = utils_prep.get_pose(cam_f)
  #if world2cam is None:
  #  continue
  objs_frame={}
  for (fnum,fname,olabel,shapelabel,vglabel,o1,oc,x1,y1,x2,y2) in objs_seq:
    for (fnum,fname,olabel,shapelabel,vglabel,o2,oc,x1,y1,x2,y2) in objs_seq:
      if o1==o2:
        continue
      bbx1 = objs_all[(scene,fname,o1)]
      bbx2 = objs_all[(scene,fname,o2)]
      centre1 = bbx1[24:27] 
      centre2 = bbx2[24:27]
      if centre1[0] < centre2[0]:
        f.write(' '.join((seq_name,fname,o1,o2,'left'))+'\n')
        f.write(' '.join((seq_name,fname,o2,o1,'not_left'))+'\n')
      if centre1[1] < centre2[1]:
        f.write(' '.join((seq_name,fname,o1,o2,'below'))+'\n')
        f.write(' '.join((seq_name,fname,o2,o1,'not_below'))+'\n')
      if centre1[2] < centre2[2]:
        f.write(' '.join((seq_name,fname,o1,o2,'in_front_of'))+'\n')
        f.write(' '.join((seq_name,fname,o2,o1,'not_in_front_of'))+'\n')
f.close()

