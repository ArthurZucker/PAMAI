from dataclasses import dataclass
from simple_parsing.helpers import list_field
from typing import List

@dataclass
class hparams:
    """Hyperparameters of MyModel"""
    # self.config.visualization_path
    visualization_path: str =  "/home/arthur/Work/FlyingFoxes/sources/flying_foxes_study/AudioEventDetection/DENet/assets/sound_vis/example_call.png"
    # validatin size
    valid_size: float = 0.2
    # validation frequency
    validate_every: int = 1
    # Learning rate of the Adam optimizer.
    lr: float = 0.05
    # learning rate scheduler, policy to decrease
    lr_schedule: str = "poly2"
    # Momentum of the optimizer.
    momentum: float = 0.01
    # Use cuda for training
    cuda: bool = True
    # gpu_device
    gpu_device: int = 0 
    # Architecture to choose, available are "denet (to come)", "sincnet (to come)", "leaf (to come)", "yolor (to come)"
    arch: str = "denet"
    #Agent to use, the agent has his own trining loop
    agent: str = "DenetAgent"
    # Dataset used for training
    dataset: str = "egyptian"
    # path to images in dataset
    img_dir: str = "/home/arthur/Work/FlyingFoxes/database/EgyptianFruitBats"
    # annotation path 
    annotation_file: str = "/home/arthur/Work/FlyingFoxes/sources/flying_foxes_study/AudioEventDetection/DENet/assets/subsampled_datset.csv"
    # Number of workers used for the dataloader
    num_workers: int = 16
    # number of classes to consider
    num_classes: int = 2
    # Sampling rate of the raw audio
    sr: int = 25000  # in HZ
    # Probability of dropout
    dropout: float = 0.3
    # Use of deterministic training, setting constant random seed
    deterministic: bool = True
    # Rank in the cluster
    global_rank: int = 0
    # Number of epochs
    epochs: int = 5
    # optimizer 
    optimizer: str = "Rmsprop"
    # decay : 
    weight_decay: float = 0.001
    # loss
    loss: str = "NNL"
    # optimization
    batch_size: int = 32
    N_epochs: int = 2900
    N_batches: int = 100
    N_eval_epoch: int = 50
    reg_factor: int = 10000
    fact_amp: float = 0.2
    seed: int = 1234
    # checkpoint dir
    checkpoint_dir: str = "weights"
    # checkpoint file 
    checkpoint_file: str = ""
    # mode
    mode: str = "train"
    # Toggle testing mode, which only runs a few epochs and val
    test_mode: bool = True
    # max epoch tu run 
    max_epoch: int = 5
    # async_loading
    async_loading: bool = True
    # activation function
    activation: str = "relu"
    # frequency sample
    fs: int = 8000
    # length of the input in ms
    cw_len: int = 500
    # overlap in ms between the samples taken from the input
    cw_shift: int = 10
    # input dimension in terms of samples, actual input size of the NN here, 1600
    input_dim: int = int(fs*cw_len/1000.00)
    # Regex to process lists
    #  =([^ ].*?),(.*)\)
    # : List[] = list_field($1,

    # cnn
    cnn_N_filt: List[int] = list_field(80,60,60)
    cnn_len_filt: List[int] = list_field(251,5,5)
    cnn_max_pool_len: List[int] = list_field(3,3,3)
    cnn_use_laynorm_inp: bool =True
    cnn_use_batchnorm_inp: bool =False
    cnn_use_laynorm: List[bool] = list_field(True,True,True)
    cnn_use_batchnorm: List[bool] = list_field(False,False,False)
    cnn_act: List[str] = list_field("relu","relu","relu")
    cnn_drop: List[float] = list_field(0.0, 0.0, 0.0)

    # dnn
    fc_lay: List[int] = list_field(2048,2048,2048,1024,256,2)
    fc_drop: List[float] = list_field(0.0,0.0,0.0,0.0,0.0,0.1)
    fc_use_laynorm_inp: bool = True
    fc_use_batchnorm_inp: bool = False
    fc_use_batchnorm: List[bool] = list_field(True,True,True,True,True,True)
    fc_use_laynorm: List[bool] = list_field(False,False,False,False,False,False)
    fc_act: List[str] = list_field("leaky_relu","linear","leaky_relu","leaky_relu","leaky_relu","softmax")

    # Denet parameters
    # Denet dropout 
    DenetDropout: float = 0.3
    # number of classes 
    nb_classes: int = 2
    # apply attention before pooling
    DenetBeforePooling: bool = True
    # Sum over channels
    SumChannel: bool = True

     # amsgrad
    amsgrad: float = 0