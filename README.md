# DENet training code 

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Author 
- [Arthur Zucker](https://github.com/ArthurZucker)

## Installation 
Clone this repository 

## Model 
The model is available at the original github repository : git@github.com:MiviaLab/DENet.git

## Information 
The original DENET code is in tensorflow, but I need to use both YOLOR and DENET, and given that YOLOR is in pytorch, I have to convert the code to pytorch. That's also a good exercice, thus I will do it. 

Moreover, I am using my template implementation which allows to easly switch between model and data type, integrating this also requires a bit of work. 
## TODO 

- [x] Create dataset
- [ ] Add `get_model`, `get_loss`, `get_optimizer` from the names
- [x] Create dataloaders
- [ ] Convert both [__init__.py](./__init__.py) and [layers.py](./layers.py) to pytorch (since its now in tensorflow)
- [ ] Create model 
- [ ] Create trainig agent
- [ ] Add utils and choose metric 
- [ ] Sample training on subset 
- [ ] **Use fp16 and mixed precision with apex!**
- [ ] Create Docker 
- [ ] Details on the readme 
- [ ] Add to my website 
