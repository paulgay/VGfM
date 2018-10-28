# Visual Graphs from Motion (VGfM): Scene understanding with object geometry reasoning

This repo contains the code to generate scene graphs given an image sequence and bounding boxe proposal. It also contains the relation annotations added on the ScanNet dataset. This material has been used in the following paper:

```
@article{gay2018visual,
  title={Visual Graphs from Motion (VGfM): Scene understanding with object geometry reasoning},
  author={Gay, Paul and James, Stuart and Del Bue, Alessio},
  booktitle={Asian conference on computer vision (ACCV)},
  year={2018}
}
```
# Installation 

This code is heavily build from the work of Xu et al. 
```
@inproceedings{xu2017scenegraph,
  title={Scene Graph Generation by Iterative Message Passing},
  author={Xu, Danfei and Zhu, Yuke and Choy, Christopher and Fei-Fei, Li},
  booktitle={Computer Vision and Pattern Recognition (CVPR)},
  year={2017}
 }
```

To install the framework, you must follow the instructions from their [repo](https://github.com/danfeiX/scene-graph-TF-release).

Eventually, check the [data_tools](https://github.com/paulgay/VGfM/tree/master/data_tools) folder for details on how to pre-process the ScanNet data.

# inference on a model
From the root directory:
```
model=dual_graph_vrd_3d
model_file=checkpoints/weights_200999.ckpt # there place the path the model you want to test 
out_file=my_out_file # will save the GT and the softmax for output for each state. This can be used for later evaluation. It uses the pickle python module
datadir=/ssd_disk/datasets/scannet/material_for_scene_graph/lfdc_quadrics/
gpu_id=1
split=2 # 0 for training, 1 for validation, 2 for testing

bash experiments/scripts/test_scannet.sh ${model_file} $model $gpu_id ${my_out_file}  $datadir 0 $split
```
The different models are coded in the file `lib/networks/models.py` as different classes:

* `dual_graph_vrd_final`: the initial model of Xu et al. 
* `dual_graph_vrd_3d`: VGfM approach
* `dual_graph_vrd_2d`: VGfM-2D: using 2D bounding boxes instead of quadrics

To train the VGfM-fusion, you need to use a different script:
```
model=dual_graph_vrd_fus

bash experiments/scripts/test_scannet_fus.sh ${model_file} $model 0 ${my_out_dir}/weights_${iter}_${split}.pc  $datadir 0 $split
```

The .pc pickle file contains the predictions for every object and relations as well as the corresponding GT.

# training a model

```
model=dual_graph_vrd_3d 
niter=2 # number of iterations of message passing
checkpoint_dir=checkpoints/this_exp/ # the weights will be saved in this directory
gpu_id=1
bash experiments/scripts/train.sh $model $niter $checkpoint_dir $gpu_id $datadir 0
```
or if you want to train the VGfM-fusion model:
```
bash experiments/scripts/train_fus.sh $model $niter $checkpoint_dir $gpu_id $datadir 0
```

# Evaluation
The Evaluation is usually done for all the .pc files stored in the `my_out_dir` directory obtained from the model inference runs. Typically, it contains the results at different steps of the training

The following script computes the accuracies:
```
python tools/comp_accuracy.py --result_dir ${my_out_dir}  --dict_file ${datadir}/scannet-SGG-dicts.json --output accuracy_result.res
```
where accuracy_result.res is a file with a python dictionnary which contains the accuracy for each model found in the directory given by the option result_dir.

You can then use this script to plot the curves
```
python tools/plot_curves.py --path accuracy_result.res
```

# Citation
If you use this dataset, please cite this paper:



