# Bayesian Optimization

For Bayesian optimization, we used the scripts from https://github.com/mkusner/grammarVAE

This requires you to install their customized Theano library. 
Please see https://github.com/mkusner/grammarVAE#bayesian-optimization for installation.

## Usage
First generate the latent representation of all training molecules:
```
python gen_latent.py --data ../data/train.txt --vocab ../data/vocab.txt \
--hidden 450 --depth 3 --latent 56 \
--model ../molvae/MPNVAE-h450-L56-d3-beta0.005/model.4
```
This generates `latent_features.txt` for latent vectors and other files for logP, synthetic accessability scores.

To run Bayesian optimization:
```
SEED=1
mkdir results$SEED
python run_bo.py --vocab ../data/vocab.txt --save_dir results$SEED \
--hidden 450 --depth 3 --latent 56 --seed $SEED \
--save_dir ../molvae/MPNVAE-h450-L56-d3-beta0.005/model.4
```
It performs five iterations of Bayesian optimization with EI heuristics, and saves discovered molecules in `results$SEED/` 
Following previous work, we tried `$SEED` from 1 to 10.

To summarize results accross 10 runs:
```
python print_result.py
```
