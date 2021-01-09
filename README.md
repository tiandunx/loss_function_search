# (ICML2020) Loss Function Search for Face Recognition
Xiaobo Wang*, Shuo Wang*, Cheng Chi, Shifeng Zhang, Tao Mei

This is the official implementation of our loss function search for face recognition.
It's accepted by ICML 2020.

## Introduction
In face recognition, designing margin-based (e.g., angular, additive, additive angular margins) softmax loss functions plays an important role in learning discriminative features.
However, these hand-crafted heuristic methods are sub-optimal because they require much effort to explore the large design space.
We first analyze that the key to enhance the feature discrimination is actually how to <font color=#FF0000>**reduce the softmax probability** </font>. We then design a unified formulation for the current margin-based softmax losses. Accordingly, we define a novel search space and develop a reward-guided search method to automatically obtain the best candidate.
Experimental results on a variety of face recognition benchmarks have demonstrated the effectiveness of our method over the state-of-the-art alternatives.
## Results
![image](https://github.com/tiandunx/loss_function_search/blob/master/resource/result.png)
## Our Search Space
![image](https://github.com/tiandunx/loss_function_search/blob/master/resource/search_space.png)
To validate the effectiveness of our search space, one can simply choose random-softmax. In train.sh, you can set do_search=1.
![image](https://github.com/tiandunx/loss_function_search/blob/master/resource/random_softmax.png)
## How to train
### Prerequisite
Pytorch 1.1 or higher are required.
### Data preparation
In current implementation, we use lmdb to pack our training images. The format of our lmdb mainly comes from Caffe. And
you could write your own caffe.proto file as follows:
```buildoutcfg
syntax = "proto2";
message Datum {
    //the acutal image data, in bytes.
   optional bytes data=1;
}
```
Aside from the lmdb, there should exist a text file describing the lmdb. Each line of the text file
contains 2 fields which is separated by a space. The line in the text file is as follows:
```buildoutcfg
lmdb_key label
```
### Train
```shell
./train.sh
```
You could either use ./train.sh. NOTE that before you execute train.sh, you should provide your own train_source_lmdb 
and train_source_file. For more usage, please 
```python
python main.py -h
```


