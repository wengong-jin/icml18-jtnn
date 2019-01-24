# Molecule Generation
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
python pretrain.py --train moses-processed --vocab ../data/vocab.txt --hidden 450 --save_dir vae_model/
```
Default Options:

`--beta 0` means to set KL regularization weight (beta) initially to be zero.

`--warmup 20000` means that beta will not increase within first 20000 training steps. It is recommended because using large KL regularization (large beta) in the beginning of training is harmful for model performance.

`--step_beta 0.01 --kl_anneal_iter 5000` means beta will increase by 0.01 every 5000 training steps (batch updates). You should observe that the KL will decrease as beta increases.

`--max_beta 1.0 ` sets the maximum value of beta to be 1.0. 

`--save_dir vae_model`: the model will be saved in vae_model/

Please note that this is not necessarily the best annealing strategy. You are welcomed to adjust these parameters.

## Testing
To sample new molecules with trained models, simply run
```
python sample.py --nsample 30000 --vocab ../data/moses/vocab.txt --hidden 450 --model moses-h450z56/model.iter-400000 > mol_samples.txt
```
This script prints in each line the SMILES string of each molecule. `model.iter-400000` is a model trained with 400K steps with the default hyperparameters. This should give you the same files as in `moses-h450z56/sample.400000`.

