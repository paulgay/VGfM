# VGfM
This repository contains annotations made over the first version of the ScanNet dataset [1]. A set of predicates are given for 8762 pairs of objects in the file `annotations.json`. 


We manually annotated 4 predicates: 
* part of: A portion or division of a whole that is separate or distinct, e.g. shelf is part-of a bookcase.
* support: To bear or hold up. 
* same set: Belonging to a group with similar properties of function. The objects could define a region, e.g. in the illustration below, the table, chair and plate belong to the same set whereas the shoes on the floor are separated.
* same plane: Belonging to the same vertical plane in regards to the ground normal.


![adding image](https://github.com/paulgay/VGfM/blob/master/images/illustration_relations_github.png)



We are currently working to propagate these annotations to the second version of ScanNet, in which the 3D segmentation of the objects have been considerably improved.

If you use this dataset, please cite this paper:
```
@article{gay2018visual,
  title={Visual Graphs from Motion (VGfM): Scene understanding with object geometry reasoning},
  author={Gay, Paul and James, Stuart and Del Bue, Alessio},
  journal={arXiv preprint arXiv:1807.05933},
  year={2018}
}

```





[1]: ScanNet: Richly-annotated 3D Reconstructions of Indoor Scenes, Dai, Angela et al, CVPR 2017


