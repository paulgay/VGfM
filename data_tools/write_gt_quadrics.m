clear all
load('home/gay/exp_icip_2018/pers_calibrated/psfmo_persp_c_constraints/pers_all_precond.mat');
addpath('/home/gay/exp_icip_2018/functions');
seqdir = '/ssd_disk/gay/scenegraph/seqs/';
fid = fopen('/ssd_disk/gay/scenegraph/sparse_seqs/gt_bbx.txt','w');
for ii =1:length(seqs)
  trackfile=[seqdir '/' seqs(ii).name];
  trackfile
  seq_name = strrep( seqs(ii).name,'.seq','');
  tmp=strsplit(seq_name,'_');
  scene=strcat(tmp{1},'_',tmp{2});
  base_dir = strcat('/ssd_disk/datasets/scannet/data/',scene,'/');
  [Imm, frVw, bbx, labels,ojs] = read_scannet_scene2(base_dir,trackfile);
  no = length(ojs);
  nf = length(Imm);
  GT = allresults{ii}.GT;
  ell=quadric2ellipsoide(GT);
  bbx3d=ellipsoid2bbx(ell);
  centres = zeros(3,no);
  for o = 1:no
    centres(:,o) = ell{o}.C;
  end
  for f = 1:nf
    for o = 1:no
      or = [ reshape(bbx3d(o,:),3,8) centres(:,o) ];
      bbx3d_or = [Imm{f}.R Imm{f}.t] * [ or ; ones(1, size(or,2)) ]; 
      bbx3d_or = reshape(bbx3d_or(1:3,:),1,27);
      fprintf(fid,'%s %s %d',seq_name,Imm{f}.fname{1},ojs(o));
      for i=1:length(bbx3d_or)
        fprintf(fid,' %f',bbx3d_or(i));
      end      
      fprintf(fid,'\n');
    end
  end
end
fclose(fid)
