# Robust Distant Supervision Relation Extraction via Deep Reinforcement Learning

PyTorch implementation of the Reinforcement Learning for Distant Supervision RE model described in our ACL 2018 paper [Robust Distant Supervision Relation Extraction via Deep Reinforcement Learning](https://arxiv.org/abs/1805.09927). In this work, we try to use reinforcement learning method to detect and remove noise instances for each relation type; moreover, this process is independent to the traning of relation extraction system.

## Steps to run the experiments

### Requirements
* ``Python 2.7.12 ``
* ``PyTorch 0.4.1``
* ``panda 0.19.1``

### Datasets and word embeddings
* [Dataset and Pretrained word embeddings](https://pan.baidu.com/s/1RT6bBtqzsJg4WfBCBvqw4g) are from [OpenNRE](https://github.com/thunlp/OpenNRE). Please download and put it into this directory. 
* We include two versions of training dataset; they have different size, ``522611`` sentences and ``570088`` sentences repectively. This two options are included in ``args.py``. Compared with ``570088`` version, ``522611`` version removes entity pairs that are repetitive with test dataset. ``522611`` is the default options in ``args.py``.

### Training
* python train.py

### Output
* The cleaned dataset is outputed to the directory ``./cleaned_data``. 

### Test
* In order to validate the performance, we run [thunlp/NRE](https://github.com/thunlp/NRE) on the cleaned dataset. For convenience, we have put their code in to the directory ``./NRE-master``. 
* Taking CNN-ONE model as an example, run the code by
* make
* ./train
* The Precision-Recall file is outputed to ``./NRE-master/CNN-ONE/out``. Good Precision-Recal curves can be obtained from pr11.txt to pr14.txt.

### Plot
* plot_PR_curve.ipynb

### Reference
```
@article{qin2018robust,
  title={Robust Distant Supervision Relation Extraction via Deep Reinforcement Learning},
  author={Qin, Pengda and Xu, Weiran and Wang, William Yang},
  journal={arXiv preprint arXiv:1805.09927},
  year={2018}
}
```
