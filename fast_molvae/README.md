# Accelerated Training of Junction Tree VAE
Suppose the repository is downloaded at `$PREFIX/icml18-jtnn` directory. First set up environment variables:
```
export PYTHONPATH=$PREFIX/icml18-jtnn
```
The MOSES dataset is in `icml18-jtnn/data/moses` (copied from https://github.com/molecularsets/moses).

## Deriving Vocabulary 
If you are running our code on a new dataset, you need to compute the vocabulary from your dataset.
To perform tree decomposition over a set of molecules, run
```
python ../fast_jtnn/mol_tree.py < ../data/moses/train.txt
```
This gives you the vocabulary of cluster labels over the dataset `train.txt`. 

## Training
Step 1: Preprocess the data:
```
python preprocess.py --train ../data/moses/train.txt --split 100 --jobs 16
mkdir moses-processed
mv tensor* moses-processed
```
This script will preprocess the training data (subgraph enumeration & tree decomposition), and save results into a list of files. We suggest you to use small value for `--split` if you are working with smaller datasets.

Step 2: Train VAE model with KL annealing. 
```
mkdir vae_model/
python vae_train.py --train moses-processed --vocab ../data/vocab.txt --save_dir vae_model/
```
Default Options:

`--beta 0` means to set KL regularization weight (beta) initially to be zero.

`--warmup 40000` means that beta will not increase within first 40000 training steps. It is recommended because using large KL regularization (large beta) in the beginning of training is harmful for model performance.

`--step_beta 0.002 --kl_anneal_iter 1000` means beta will increase by 0.002 every 1000 training steps (batch updates). You should observe that the KL will decrease as beta increases.

`--max_beta 1.0 ` sets the maximum value of beta to be 1.0. 

`--save_dir vae_model`: the model will be saved in vae_model/

Please note that this is not necessarily the best annealing strategy. You are welcomed to adjust these parameters.

## Testing
To sample new molecules with trained models, simply run
```
python sample.py --nsample 30000 --vocab ../data/moses/vocab.txt --hidden 450 --model moses-h450z56/model.iter-700000 > mol_samples.txt
```
This script prints in each line the SMILES string of each molecule. `model.iter-700000` is a model trained with 700K steps with the default hyperparameters. This should give you the same samples as in [moses-h450z56/sample.txt](moses-h450z56/sample.txt). The result is as follows:
```
valid = 1.0
unique@1000 = 1.0
unique@10000 = 0.9992
FCD/Test = 0.42235413520261034
SNN/Test = 0.5560595345050097
Frag/Test = 0.996223352989786
Scaf/Test = 0.8924981494347503
FCD/TestSF = 0.9962165008703465
SNN/TestSF = 0.5272934146558245
Frag/TestSF = 0.9947901514732745
Scaf/TestSF = 0.10049873444911761
IntDiv = 0.8511712225340441
IntDiv2 = 0.8453088593783662
Filters = 0.9778
logP = 0.0054694810121243
SA = 0.015992957588069068
QED = 1.15692473423544e-05
NP = 0.021087573878091237
weight = 0.5403194879856983
```
