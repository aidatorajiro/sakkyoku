# midi notes trainer & generator in pytorch

## How to run

1. make `dataset` directory and put training midi files in it
2. run `train.py`. Every 100 epochs, weight file will be generated.
3. change `const.py` setting (`gen_load` and `optim_load` variable) according to generated weight files
4. run `gen.py`. `out.mid` will be generated.

## network

linear -> relu -> linear -> relu -> linear -> relu -> softmax

Each note will be serialized as 2 * 128 tensor

Input: last 16 notes

Output: probability distribution of note melody and note length

