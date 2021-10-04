from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset

from runx.logx import logx


from sklearn.model_selection import train_test_split
from raw_audio_dataset import raw_audio_dataset

class raw_audio_Dataloader():
    """
    Creates a dataloader for train and val splits
    """
    def __init__(self,args):
        
        
        # self.logger     = logging.getLogger("Cifar10DataLoader") # should i use this to debug?
        self.transform = None
        
        
        logx.msg(f"BraTS_mean_Dataloader, data_mode : {args.data_mode}, path : {self.config.img_dir}")
        dataset = raw_audio_dataset(self.config.img_dir,self.config.annotation_file,transform,self.config.build,split="train")

        train_indices, valid_indices = train_test_split(range(len(dataset)),test_size=self.config.valid_size,train_size=1-self.config.valid_size,
                                                        shuffle=False)

        train_dataset = Subset(dataset, train_indices)
        valid_dataset = Subset(dataset, valid_indices)

        self.len_train_data = len(train_dataset)
        self.len_valid_data = len(valid_dataset)

        self.train_iterations = (self.len_train_data + self.args.batch_size - 1) // self.config.batch_size
        self.valid_iterations = (self.len_valid_data + self.args.batch_size - 1) // self.config.batch_size


        self.train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True)
        self.valid_loader = DataLoader(valid_dataset, batch_size=self.args.batch_size, shuffle=False) 
        