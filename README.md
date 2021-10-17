# DENet training code 

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Author 
- [Arthur Zucker](https://github.com/ArthurZucker)

## Installation 
Clone this repository 

## Model 
The model is available at the original github repository : git@github.com:MiviaLab/DENet.git

## Information 
The original DENET code is in tensorflow, but I need to use both YOLOR and DENET, and given that YOLOR is in pytorch, I have to convert the code to pytorch. That's also a good exercise, thus I will do it. 

Moreover, I am using my template implementation which allows to easily switch between model and data type, integrating this also requires a bit of work. 
## TODO 

- [x] Create dataset
- [x] Add `get_model`, `get_loss`, `get_optimizer` from the names
- [x] Create dataloaders
- [ ] Convert both [__init__.py](./__init__.py) and [layers.py](./layers.py) to pytorch (since its now in tensorflow)
- [ ] Create model 
- [ ] Create training agent
- [ ] Add utils and choose metric 
- [ ] Sample training on subset 
- [ ] **Use fp16 and mixed precision with apex!**
- [ ] Create Docker 
- [ ] Details on the readme 
- [ ] Add to my website 

## Issue : 
output of sincnet and denet is the label at each window of 50ms. Thus the labels that I use should also be of the same format. Which means that when I select an audio file to use, randomly select a part of it? then extract the label from the sample. 
Sample available labels : {500->550},{...},{...} a list of ranges of samples where we know that a flying foxe shouted! 

DENET's input : 1 sequence of 10 frames of 50ms audio. Labels? Probably a sequence of labels.  
SincNet's input : 1 chunk of 200ms, 10ms overlap, 128 batch size. Labels are for each wav file, a single label, which is a bit strange. 


### Annotations 

Realized that the annotations were a bit... Strange? 
In the `FileInfo.csv` file, each call's start sample and end sample is labeled. For example, for the following recording (visualized using sonic Visualiser), 2 calls are visible, the label is the following: 

![example immage](assets/images/example_call.png)

| File name              | Voice start sample (1) | Voice end sample (1) | Voice start sample (2) | Voice end sample (2) |
|------------------------|------------------------|----------------------|------------------------|----------------------|
| 120601011047196133.WAV | 47553                  | 95094                | 298537                 | 349426               |

But in the `annotation.csv`, the corresponding labels are the following: 

| FileID | Context | Start sample | End sample |
|--------|---------|--------------|------------|
| 30     | 1       | 298537       | 607056     |

Not only is a single call labeled, but the tail of it is also marked as `context = 1`. This makes it a bit strange. Though we can actually see that there is some noise/signal at the end of the recording, which annotation should we trust. 

### Dataloader

This has a huge impact on the way I will chose to label the data from a given random begin and end sample. 
I don't have a lot of option. 

1. Use call annotations, without processing it. That's the simplest solution but not very intelligent. Though given that the GRU uses previous samples, it should be able to automatically know tht this or this is not a call anymore (tail of a call). 
2. Look into unsupervised learning, wince some of the calls are labeled, others are not. In this case I should use the per-label calls. Then, process them *using the phoneme annotation* to at least remove the silence parts. Since the GRU uses previous samples, I guess it makes sense to say "this is still part of the audio event" and process it as a whole. 

## More ideas:
Add FF call to background sounds and use it as training. Hashizume's FF already have Background noises. 

### Improving the dataset  

Our dataset only has positive instances -- bat calls are almost always present in recordings -- thus we need to augment it with more recordings, were other events happen. We have a huge enough dataset, which means that we can subsample it, and combine it with random noises from famous benchmark dataset? The audio conditions will be different :confused: