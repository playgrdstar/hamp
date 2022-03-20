## Learning User Interface Semantics from Heterogeneous Networks with Multimodal and Positional Attributes

### Overview:
This repository contains a Pytorch implementation of the HAMP model proposed in the paper Learning User Interface Semantics from Heterogeneous Networks with Multimodal and Positional Attributes.

### Requirements:

The key libraries required are Pytorch, Numpy, Pandas, Scipy, Networkx, DGL. See requirements.txt.

### Datasets

Due to Github's size constraints (as the data is >100MB), please email me (at playgrdstar(at)gmail(dot)com) for the data files that should be placed in this folder. Data should be placed in the ``data/rico_n`` folder. 

### Repository Organization
- ``data/rico_n`` this should contain the pre-processed datasets. 
- ``models.py`` contains the models
- ``train.py`` is the notebook that can be used to load pre-trained model or train the model 

### Running the code
Run command below for the UI screen genre classification task. Change to task=='element_comp_class' for the UI element component type classification task.
```
python train.py --task='screen_genre_class' --n_epochs=3000
```

If you have issues with the tables or pytables library, which we will need for loading hdf files (particularly on Windows machines), then run the following.
```
conda remove -n py pytables
conda config --add channels conda-forge
conda install pytables
```

## Citation

If you use this repository, e.g., the code and the datasets, in your research, please cite the following paper:
```
@inproceedings{hamp_iui2021,
    title={Learning User Interface Semantics from Heterogeneous Networks with Multimodal and Positional Attributes},
    author={Ang, Gary and Lim, Ee-Peng},
    booktitle={27th ACM International Conference on Intelligent User Interfaces (IUI 2022)},
    year={2022}
}
```

## Credits
[DGL](https://github.com/dmlc/dgl/tree/master/examples/pytorch/hgt), [ND](https://github.com/trokas/neural_decomposition)

