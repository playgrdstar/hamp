## Learning User Interface Semantics from Heterogeneous Networks with Multimodal and Positional Attributes

### Overview:
This repository contains a Pytorch implementation of the HAMP model proposed in the paper Learning User Interface Semantics from Heterogeneous Networks with Multimodal and Positional Attributes.

### Requirements:

The key libraries required are Pytorch, Numpy, Pandas, Scipy, Networkx, DGL. See requirements.txt.

### Datasets

The RICO data set is publicly available at http://interactionmining.org/. Details on processing the data are provided in the paper.

### Repository Organization
- ``models.py`` contains the models
- ``train.py`` trains the model 
- ``data/rico_n`` this should contain the pre-processed datasets. 

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

