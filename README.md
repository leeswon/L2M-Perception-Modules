# L2M Perception Modules

This is for integration of perception (especially classifier) modules of L2M project.

## Instruction prior to usage
1. Install [Habitat API](https://github.com/facebookresearch/habitat-api) and [Habitat Sim](https://github.com/facebookresearch/habitat-sim).
1. Download MatterPort3D data as follows (check [here](https://github.com/facebookresearch/habitat-api#data)):
    1. Download 'Point goal navigation (MatterPort3D)' file in the table of 'Task datasets'.
    1. Download MatterPort3D scene datasets by using the download script and command
    `python download_mp.py --task habitat -o data/scene_datasets/mp3d/`
1. Unzip the downloaded data at `habitat-api/data`
    1. Point goal navigation : `habitat-api/data/datasets/pointnav/mp3d`
    1. Scene datasets : `habitat-api/data/scene_datasets/mp3d`
    
## How to run testing code
1. Modify the path to Habitat-API at line 80 of `testMain.py`
1. To visualize the data from Habitat and data generator, change the variable `_debug1` and `_debug2` in `testMain.py` from False to True.
1. To incorporate classifier, load the classifier at line 103 of `testMain.py`, and modify other parts of code if necessary.