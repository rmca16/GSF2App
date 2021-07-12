# GSF2App
Implementation of the Global and Semantic Feature Fusion Approach ([GSF2App](https://ieeexplore.ieee.org/abstract/document/9096068)) for Indoor Scene Classification using the PyTorch framework.

<p align="center"><img src="assets/GSF2App.png" width="720"\></p>

## Performing
GSF2App was evaluated on the [NYU V2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) and [SUN RGB-D](https://rgbd.cs.princeton.edu) datasets.

<p align="center"><img src="assets/GSF2App_NYU_results.png" width="350"/> <img src="assets/GSF2App_SUN_results.png" width="350"/> 
  
### Requirements

    Python >= 3.6
    PyTorch >= 1.0.1
    torchvision
    cv2
    tqmd
    
### Training & Evaluate
Global and Semantic features are combined in a two-step learning.
<p align="center"><img src="assets/GSF2App_training.png" width="400"\></p>


YOLOv3:

  You need to download a COCO's dataset model and add it on YOLOv3\weights folder. You can do it [here](https://drive.google.com/file/d/1u5gyZZnUA-8MetKhW2U-8g29WOzltIV0/view?usp=sharing).
  For more details you can check this YOLOv3's [PyTorch implementation](https://github.com/eriklindernoren/PyTorch-YOLOv3), or you can check the [original](https://pjreddie.com/darknet/yolo/) implementation.
  

To train:

    $ python3 GSF2App_train.py --stage_1_n_epochs 75 --batch_size 32 (and so on)(see the options available on the training file)
  
or you can edit the options available directly on the file and:

    $ python3 GSF2App_train.py

To Evaluate:

The same options as aforementioned are available...

    $ python3 GSF2App_eval

## Citation

```
@InProceedings{gsf2app_2020,
  author={R. {Pereira} and N. {Gonçalves} and L. {Garrote} and T. {Barros} and A. {Lopes} and U. J. {Nunes}},
  booktitle={IEEE International Conference on Autonomous Robot Systems and Competitions (ICARSC)}, 
  title={{Deep-Learning based Global and Semantic Feature Fusion for Indoor Scene Classification}}, 
  year={2020}}
```

```
@InProceedings{yolov3,
	author={Redmon, Joseph and Farhadi, Ali},
	title={{YOLOv3: An Incremental Improvement}},
	booktitle = {arXiv},
	year={2018}}
```


## Contacts
ricardo.pereira@isr.uc.pt
