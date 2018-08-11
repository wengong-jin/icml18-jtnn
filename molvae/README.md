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
This gives you the vocabulary of cluster labels over the dataset `all.txt`. Note that it will give you warnings when it encounters a molecule with high tree-width. It is recommended to remove them from the dataset, as training JT-VAE on high tree-width molecules will cause out-of-memory error. 

## Training
We trained VAE model in two phases:
1. We train our model for three epochs without KL regularization term (So we are essentially training an autoencoder).
Pretrain our model as follows (with hidden state dimension=450, latent code dimension=56, graph message passing depth=3):
```
mkdir pre_model/
CUDA_VISIBLE_DEVICES=0 python pretrain.py --train ../data/train.txt --vocab ../data/vocab.txt \
--hidden 450 --depth 3 --latent 56 --batch 40 \
--save_dir pre_model/
```
PyTorch by default uses all GPUs, setting flag `CUDA_VISIBLE_DEVICES=0` forces PyTorch to use the first GPU (1 for second GPU and so on).

The final model is saved at pre_model/model.2

2. Train out model with KL regularization, with constant regularization weight $beta$. 
We found setting beta > 0.01 greatly damages reconstruction accuracy.
```
mkdir vae_model/
CUDA_VISIBLE_DEVICES=0 python vaetrain.py --train ../data/train.txt --vocab ../data/vocab.txt \
--hidden 450 --depth 3 --latent 56 --batch 40 --lr 0.0007 --beta 0.005 \
--model pre_model/model.2 --save_dir vae_model/
```

## Testing
To sample new molecules with pretrained models, simply run
```
python sample.py --nsample 100 --vocab ../data/vocab.txt \
--hidden 450 --depth 3 --latent 56 \
--model MPNVAE-h450-L56-d3-beta0.005/model.4
```
This script prints each line the SMILES string of each molecule. `prior_mols.txt` contains these SMILES strings.

For molecule reconstruction, run  
```
python reconstruct.py --test ../data/test.txt --vocab ../data/vocab.txt \
--hidden 450 --depth 3 --latent 56 \
--model MPNVAE-h450-L56-d3-beta0.005/model.4
```
Replace `test.txt` with `valid.txt` to test the validation accuracy (for hyperparameter tuning).

