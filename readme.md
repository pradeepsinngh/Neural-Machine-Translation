# Neural Machine Translation using Tensorflow 2.0

This reporsiroty host my code base for Neural Machine Translation projects. I have implemented 3 different types of models for machine translation:
 - MT using Seq2Seq model (No attention mechanism)
 - MT using Seq2Seq model with attention
 - MT using Transformer 
 
 
 ### Structure: 
```
├─ NMT_xyz             
│  ├─ checkpoints      
│  │  ├─ decoder
│  │  │  └─ *.h5
│  │  └─ encoder
│  │  │  └─ *.h5
│  ├─ data.             
│  │  ├─ fra-eng
│  │  │  ├─ _about.txt
│  │  │  └─ fra.txt
│  │  └─ fra-eng.zip   
│  ├─ data.py           
│  ├─ model.py         
│  ├─ preprocessing.py
│  ├─ run.py            
│  └─ utils.py          
└─ readme.md            
```
 
 # replace xyz with attention, transformer, seq2seq
# stores encoder and decoder weights in .h5
# data for training models
# data in zip file
# script for loading and creating dataset
# file for encoder and decoder models
# for pre-processing data
# driver file for the project - set MODE == 'TRAIN' or 'TEST'
# utility function 
