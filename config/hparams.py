from dataclasses import dataclass
from simple_parsing.helpers import list_field
from typing import List

@dataclass
class hparams:
    """Hyperparameters of MyModel"""
    # validatin size
    valid_size: float = 0.2
    # Learning rate of the Adam optimizer.
    lr: float = 0.05
    # Momentum of the optimizer.
    momentum: float = 0.01
    # Architecture to choose, available are "denet (to come)", "sincnet (to come)", "leaf (to come)", "yolor (to come)"
    arch: str = "sincnet"
    #Agent to use, the agent has his own trining loop
    agent: str = "DenetAgent"
    # Dataset used for training
    dataset: str = "egyptian"
    # path to images in dataset
    img_dir: str = "/home/arthur/Work/FlyingFoxes/database/EgyptianFruitBats"
    # annotation path 
    annotation_file: str = "/home/arthur/Work/FlyingFoxes/database/EgyptianFruitBats/processed_labeled.csv"
    # Number of workers used for the dataloader
    num_workers: int = 8
    # Sampling rate of the raw audio
    sr: int = 24000  # in HZ
    # Probability of dropout
    dropout: float = 0.3
    # Use of deterministic training, setting constant random seed
    deterministic: bool = True
    # Toggle testing mode, which only runs a few epochs and val
    test_mode: bool = True
    # Rank in the cluster
    global_rank: int = 0
    # Number of epochs
    epochs: int = 10
    # optimizer 
    optimizer: str = "RMSprop"
    # optimization
    batch_size: int =128
    N_epochs=2900
    N_batches=100
    N_eval_epoch=50
    reg_factor=10000
    fact_amp: float =0.2
    seed=1234


@dataclass
class sincnet:
    """Hyperparameters of the sincnetmodel"""

    # activation function
    activation: str = "relu"
    # windowing
    fs=8000
    cw_len=375
    cw_shift=10

    # Regex to process lists
    #  =([^ ].*?),(.*)\)
    # : List[] = list_field($1,

    # cnn
    cnn_N_filt: List[int] = list_field(80,60,60)
    cnn_len_filt: List[int] = list_field(251,5,5)
    cnn_max_pool_len: List[int] = list_field(3,3,3)
    cnn_use_laynorm_inp=True
    cnn_use_batchnorm_inp=False
    cnn_use_laynorm: List[bool] = list_field(True,True,True)
    cnn_use_batchnorm: List[bool] = list_field(False,False,False)
    cnn_act: List[str] = list_field("relu","relu","relu")
    cnn_drop: List[float] = list_field(0.0, 0.0)

    # dnn
    fc_lay: List[int] = list_field(2048,2048,2048)
    fc_drop: List[float] = list_field(0.0,0.0)
    fc_use_laynorm_inp=True
    fc_use_batchnorm_inp=False
    fc_use_batchnorm: List[bool] = list_field(True,True,True)
    fc_use_laynorm: List[bool] = list_field(False,False,False)
    fc_act: List[str] = list_field("leaky_relu","linear","leaky_relu")

    # class
    class_lay= 2484
    class_drop: List[float] = list_field(0.0,0.0)
    class_use_laynorm_inp=True
    class_use_batchnorm_inp=False
    class_use_batchnorm=False
    class_use_laynorm=False
    class_act="softmax"


    
    