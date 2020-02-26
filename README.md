# LGSR
This is the code for our paper: Joint Modeling of Local and Global Behavior Dynamics for Session-based Recommendation. We have implemented our methods in Tensorflow

Here are two datasets we used in our paper. After downloaded the datasets, you can put them in the folder `datasets/`:
- YOOCHOOSE: <http://2015.recsyschallenge.com/challenge.html>

- DIGINETICA: <http://cikm2016.cs.iupui.edu/cikm-cup>

## Usage
You need to run the file  `datasets/preprocess.py` first to preprocess the data.

For example: `cd datasets; python preprocess.py --dataset=diginetica`

Then you can run the file `main.py` to train the model

```bash
  -h, --help            show this help message and exit
  --dataset DATASET     dataset diginetica/yoochoose1_4/yoochoose1_64
  --method METHOD       recommendation module method ha/sr_gnn
  --validation          validation
  --epoch EPOCH         number of epochs
  --batch_size BATCH_SIZE
                        input batch size
  --hidden_size HIDDEN_SIZE
                        hidden state size
  --emb_size EMB_SIZE   hidden state size
  --l2 L2               l2 penalty
  --lr LR               learning rate
  --dropout DROPOUT     dropout rate
  --max_len MAX_LEN     sequence max length
  --cide CIDE           the train frequency of cross-session item dependency
                        encoder (cide)
  --cide_batch_size CIDE_BATCH_SIZE
                        the batch size of cide
  --num_length NUM_LENGTH
                        the number of length in random walk
  --num_walks NUM_WALKS
                        the number of walk in random walk
  --skip_window SKIP_WINDOW
                        the window size in skip-gram
  --n_sample N_SAMPLE   the negative sample size in skip-gram
  --rand_seed RAND_SEED
  --log_file LOG_FILE
```


## Requirements
python 3.6

tensorflow 1.12.0

networkx 1.11

## Implementation Reference
https://github.com/eliorc/node2vec

https://github.com/CRIPAC-DIG/SR-GNN
