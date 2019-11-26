# LGSR
This is the code for our paper. We have implemented our methods in Tensorflow

Here are two datasets we used in our paper. After downloaded the datasets, you can put them in the folder `datasets/`:
- YOOCHOOSE: <http://2015.recsyschallenge.com/challenge.html>

- DIGINETICA: <http://cikm2016.cs.iupui.edu/cikm-cup>

## Usage
You need to run the file  `datasets/preprocess.py` first to preprocess the data.

For example: `cd datasets; python preprocess.py --dataset=diginetica`

## Requirements
python 3.6

tensorflow 1.12.0

## Implementation Reference
https://github.com/eliorc/node2vec

https://github.com/CRIPAC-DIG/SR-GNN
