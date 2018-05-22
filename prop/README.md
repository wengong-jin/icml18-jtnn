# Property Prediction
Here we provide graph convolutional networks (or message passing network) for molecular property prediction. 
To tryout our model, please run the training script on the provided training/development dataset.

## Training
```
mkdir model
python nntrain.py --train malaria.train --valid malaria.valid --save_dir model --epoch 20
```
This script will train the network 20 epochs, and save the best model in `model/model.best`

When input molecules are huge, adding attention mechanism would sometimes improve the performance.
```
mkdir model
python nntrain.py --train malaria.train --valid malaria.valid --save_dir model --epoch 20 --attention 1
```
You can also change the hidden layer dimension and depth of graph convolution by setting `--hidden` and `--depth` options.

## Testing
```
python nntest.py --test malaria.valid --model model/model.best
```
This will print out predicted property value for each input molecule.
