# Which data do I need?

Essentially you need 4 files. 
* `imdb_1296.h5`: contains among others the following fields: 
	* `im_paths`: the absolute paths of the images.
	* `images`: the image data
* `proposals.h5`: contains the proposal quadrics and 2d bounding boxes (in our case, the 2D bbx are GT).
	* `im_scales`: in case size is changed (not in our case)
	* `seq_names`: the sequence names
	* `seq_to_im_idx`: seq_to_im_idx[s] gives the idx of the first image for the sequence seq_names[s]
	* `num_ims`: num_ims[s] gives the number of images in the sequence seq_names[s]
	* `im_to_roi_idx`: im_to_roi_idx[i] gives the index of the first proposal bounding box for the image i.
	* `num_rois`: num_rois[i] gives the number of rois contained in the image i.
	* `num_rois`: num_rois[i] gives the number of roi present in the image i.
	* `quadric_rois` and `rpn_rois` contain the 3D and 2D bounding boxes. `rpn_rois` is in format (left, top, right, bottom)
	* `im_to_imdb_idx` : Link between the indices of the files `proposals.h5` and `imdb_1296.h5`. For example, the roi im_to_roi_idx[i] should be plotted with the image loaded from the path im_paths[im_to_imdb_idx[i]]
* `scannet-SGG.h5`: contains the GT. The array are aligned with the proposal: `rpn_rois[o]` and `boxes_1296[o]` correspond to the same object.
	* `img_to_first_box`, `img_to_last_box`: (img_to_first_box[i], img_to_last_box[i]) indices of the first and last roi for the image index i. 
	* `img_to_first_rel`, `img_to_last_rel`: same as above but for relations.
	* `boxes_1296`: contains the bounding boxes, in format (centre_x, centre_y, width, height).
	* `labels`: contain the object labels. labels[o] is the object label of the object with bounding box boxes_1296[o].
	* `relationships`: array of object couples.
	* `predicates`, `predicates_all`: predicate label for the relationships containsd in `relationships`. In the former, you have only one label. In the latter you have a binary vector where predicates_all[p] indicates if predicate p is occurring for this relation.
	* `roi_idx_to_scannet_oid`: the scannet object id (from the scannet .aggregation.json files).
	* `split`: whether this 
* `scannet-SGG-dicts.json`: dictionnary which maps the label ids with the actual words.

# Preparing the dataset

To generate this data, the python script `prepare.py` has been used.
In particular, it takes as input: 
* `quadric_file`: this file contains the 3D bbox coordinates
* `seqdir`: this directory contains the csv sequence file. Each sequence file contains the 2D detections for the corresponding sequence.
* `relations_file` : this file contains the annotated relations on the ScanNet dataset in a csv format
* The path where the ScanNet data is located (the folder containing the scenexxxx subfolders)

To create these files you need to 

1. Extract the sequences from the ScanNet dataset
2. Extract the quadrics from these sequences

These tools to create these steps are described in the [LfDC](https://gitlab.iit.it/pgay/lfd_lfdc_plfd/tree/master) github. 

After having generated the sequences and computed LfDC quadrics, you need these additional steps: 

1. Generate the 3D bounding boxes from the quadrics (see matlab script `write_pers_cons_quadrics.m`). This uses the lfdc output.

2. Make shorter sequences if you want to run the VGfM fusion model.
```
python make_sparse_seq.py # set the folder paths inside the scripts
```

