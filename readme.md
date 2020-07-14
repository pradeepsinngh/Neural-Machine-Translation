# Neural Machine Translation in Tensorflow 2.0

This repository host my code base for Neural Machine Translation projects. I have implemented 3 different types of models for machine translation:
 - [MT using Seq2Seq model]() (No attention mechanism)
 - [MT using Seq2Seq model](https://github.com/pradeepsinngh/Neural-Machine-Translation/tree/master/NMT_attention) with attention
 - [MT using Transformer](https://github.com/pradeepsinngh/Neural-Machine-Translation/tree/master/NMT_Transformer)
 
 Dataset:
 
 ### Repository Structure: 
```
├─ NMT_xyz                      # replace xyz with attention, transformer, seq2seq
│  ├─ checkpoints               # stores encoder and decoder weights in .h5
│  │  ├─ decoder
│  │  │  └─ *.h5
│  │  └─ encoder
│  │  │  └─ *.h5
│  ├─ data.                     # data for training models
│  │  ├─ fra-eng
│  │  │  ├─ _about.txt
│  │  │  └─ fra.txt
│  │  └─ fra-eng.zip             # data in zip file
│  ├─ data.py                    # script for loading and creating dataset
│  ├─ model.py                   # file for encoder and decoder models
│  ├─ preprocessing.py           # for pre-processing data
│  ├─ run.py                     # driver file for the project - set MODE == 'TRAIN' or 'TEST'
│  └─ utils.py                   # utility function 
└─ readme.md            
```
 
### How to train/ test:

To Train:
  - Set MODE = 'Train' in ```run.py``` file
  - run ```python run.py```
  
To TEST/ PREDICT:
  - Set MODE = 'TEST' in ```run.py``` file
  - run ```python run.py```
