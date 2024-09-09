# Official implementation of Cross-Scan Attention Transformer (CSAT)

## Prepare Dataset
Download pickle.zip from [link](https://drive.google.com/file/d/1GSaanysnf2dYD6pqomuUOGOHfxyfO54W/view?usp=sharing)
To prepare your own OCT dataset for SCR detection
```shell
cd csat/
python make_pickle_data.py
```
Each pickle file is a dictionary of tensors with the following entries
```
tensor = {'img':image, 'box':bounding_boxes, 'label':SCR_label, 'name':filename}
```
Generate positive and negative pairs for pre-training
```shell
python util/util.py
```
The directory structure should be similar to
```
- csat
  - data
    - positive_<fold>.txt
    - negative_<fold>.txt
    - scr.yaml
  - pickle
  - eval
  - loss
  - ...
```
Download pre-trained model from [link](https://drive.google.com/file/d/1psdJRRyuMKzhAv8R24gnXDe0McHIYo6y/view?usp=sharing)

## Train the pre-training model
```shell
# to run with default arguments
python pretrain.py

# to modify arguments
python pretrain.py --root <str> --world_size <int> --resume <bool> --resume_weight <str> --train_folder <str> --val_folder <str> --epochs <int> --folds <int> --cf <int> --batch_size <int>
```

## Train the detection model
```shell
# to run with default arguments
python train.py

# to modify arguments
python train.py --root <str> --dataroot <str> --world_size <int> --resume <bool> --resume_weight <str> --pretrain <bool> --pretrain_weights <str> --epochs <int> --nc <int> --r <int> --space <int> --train_batch <int> --val_batch <int>
```

## Run validation directly on the detection model
```shell
# to run with default arguments
python validate.py

# to modify arguments
python validate.py --root <str> --dataroot <str> --world_size <int> --weights <str> --nc <int> --r <int> --space <int> --batch <int>
```
