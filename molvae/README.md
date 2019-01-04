# Molecule Generation
Suppose the repository is downloaded at `$PREFIX/icml18-jtnn` directory. First set up environment variables:
```
export PYTHONPATH=$PREFIX/icml18-jtnn
```
Our ZINC dataset is in `icml18-jtnn/data` (copied from https://github.com/mkusner/grammarVAE). 
We follow the same train/dev/test split as previous work. 

## Deriving Vocabulary 
If you are running our code on a new dataset, you need to compute the vocabulary from your dataset.
To perform tree decomposition over a set of molecules, run
```
python ../jtnn/mol_tree.py < ../data/all.txt
```
This gives you the vocabulary of cluster labels over the dataset `all.txt`. 

## Training
Step 1: Preprocess the data:
```
python preprocess.py --train ../data/train.txt --split 30 --jobs 8 
mkdir zinc-processed
mv tensor* zinc-processed
```
This script will preprocess the training data (subgraph enumeration & tree decomposition), and save results into a list of files. We suggest you to use larger value for `--split` if you are working with larger datasets.

Step 2: Train VAE model with KL annealing strategy. 
```
mkdir vae_model/
CUDA_VISIBLE_DEVICES=0 python pretrain.py --train zinc-processed --vocab ../data/vocab.txt \
--hidden 450 --warmup 20000 --beta 0 --step_beta 0.001 --max_beta 0.01 --kl_anneal_iter 5000 \
--epoch 10 --save_dir vae_model/
```

`--beta 0` means to set KL regularization weight (beta) initially to be zero.

`--warmup 20000` means that beta will not increase within first 20000 training steps. It is recommended because using large KL regularization (large beta) in the beginning of training is harmful for model performance.

`--step_beta 0.001 --kl_anneal_iter 10000` means beta will increase by 0.001 every 5000 training steps (batches). You should observe that the KL will decrease as beta increases

`--max_beta 0.01 ` sets the maximum value of beta to be 0.01. 

`--epoch 10 `: the model will be trained with 10 epochs

`--save_dir vae_model`: the model will be saved in vae_model/

Please note that this is not necessarily the best annealing strategy. You are welcomed to adjust these parameters.

## Testing
To sample new molecules with pretrained models, simply run
```
python sample.py --nsample 1000 --vocab ../data/vocab.txt \
--hidden 450 --model vae_model/model.iter-60000 > mol_samples.txt
```
This script prints in each line the SMILES string of each molecule. 

