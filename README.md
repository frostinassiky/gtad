# G-TAD
The official implementation of G-TAD: Sub-Graph Localization for Temporal Action Detection

## Update
30 Mar 2020: THUMOS14 feature is available! Gooogle Drive [Link](https://drive.google.com/drive/folders/10PGPMJ9JaTZ18uakPgl58nu7yuKo8M_k?usp=sharing).

## Overview
Temporal action detection is a fundamental yet challenging task in video understanding. Video context is a critical cue to effectively detect actions, but current works mainly focus on temporal context, while neglecting semantic context as well as other important context properties. In this work, we propose a graph convolutional network (GCN) model to adaptively incorporate  multi-level semantic context into video features and cast temporal action detection as a sub-graph localization problem. Specifically, we formulate video snippets as graph nodes, snippet-snippet correlations as edges, and actions associated with context as target sub-graphs. With graph convolution as the basic operation, we design a GCN block called GCNeXt, which learns the features of each node by aggregating its context and dynamically updates the edges in the graph. To localize each sub-graph, we also design a SGAlign layer to embed each sub-graph into the Euclidean space. Extensive experiments show that G-TAD is capable of finding effective video context without extra supervision and achieves state-of-the-art performance on two detection benchmarks. On ActityNet-1.3, we obtain an average mAP of 34.09%; on THUMOS14, we obtain 40.16% in mAP@0.5, beating all the other one-stage methods.

[Detail](https://sites.google.com/kaust.edu.sa/g-tad), [Video](https://www.youtube.com/watch?v=BlPxnDcykUo), [Arxiv](https://arxiv.org/abs/1911.11462).

## Dependencies 
* Python == 3.7
* Pytorch==1.1.0
* CUDA==10.0.130
* CUDNN==7.5.1_0

## Installation
1. Create conda environment
    ```shell script
    conda create -f env.yml
    ```
2. Install `Align1D2.2.0` 
    ```shell script
    cd lib
    python setup.py install
    ```
3. Test `Align1D2.2.0`
    ```shell script
    python align.py
    ```
## Code Architecture

    gtad                        # this repo
    ├── data                    # feature and label
    ├── evaluation              # evaluation code from offical API
    ├── gtad_lib                # gtad library
    └── ...

## Train and evaluation
```shell script
python gtad.py --mode train
python gtad.py --mode inference
python gtad.py --mode detect
```
or
```shell script
bash gtad.sh | tee log.txt
```
## Bibtex
```text
@misc{xu2019gtad,
    title={G-TAD: Sub-Graph Localization for Temporal Action Detection},
    author={Mengmeng Xu and Chen Zhao and David S. Rojas and Ali Thabet and Bernard Ghanem},
    year={2019},
    eprint={1911.11462},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
## Contact
mengmeng.xu[at]kaust.edu.sa
