import numpy as np
import pandas as pd
import torchaudio
import torch

def sub_sample_dataset(filename1 : str,filename2:str,res_path:str = "/home/arthur/Work/FlyingFoxes/sources/flying_foxes_study/AudioEventDetection/DENet/assets/subsampled_datset.csv"):
    """Extract emitter and context labeled data form the filename dataset  

    Args: 
        filename: path and name of the dataset in csv format
    
    Return:
        Nothing. It will write the subsampled dataset at res_path

    Depending on the type of classification that will be done, various filters can be applied.
    1. Take only the files which have a context, whether it is known or unknown
    2. Remove calls where the emitter is unknown
    3. Remove calls where the context is either unknown or general (which would clearly unbalance the dataset)
    """
    list_names = ["FileID","Treatment ID","File name","File folder","Recording channel","Recording time","Voice start sample (1)","Voice end sample (1)","Voice start sample (2)","Voice end sample (2)"]
    for i in range(3,48):
        list_names.append("Voice start sample ("+str(i)+")")
        list_names.append("Voice end sample ("+str(i)+")")

    data1 = pd.read_csv(filename1)
    data2 = pd.read_csv(filename2,names=list_names,skiprows=1) # The first row is corrupted and prevented the merge. 

    result = pd.merge(data1, data2,how="left",on="FileID") # here the merge is "left" which subsamples 1/3 of the dataset. 
    # remove useless headers. 
    result["File name"]= result["File folder"].str.cat(result["File name"], sep ="/")
    result.drop(["FileID","File folder","Treatment ID","Recording channel",\
                                                    "Recording time","Addressee","Emitter pre-vocalization action", \
                                                    "Addressee pre-vocalization action", "Emitter post-vocalization action",\
                                                    "Addressee post-vocalization action"],axis=1,inplace=True)
    col = result.pop("File name")
    result.insert(0, col.name, col)
    result.to_csv(res_path)
    return result


def show_info(data : pd.DataFrame):
    nb_known_emitter_calls = len(data.query('Emitter > 0'))
    nb_known_context_calls = len(data.query('Context > 0 & Context != 11 '))

    unknown_sex = [108,113,120,210,214,220,220,230]
    nb_known_call_genders = len(data.query('Emitter not in @unknown_sex & Emitter > 0'))

    print(f"Known emitter calls : {nb_known_emitter_calls}")
    print(f"Known context calls : {nb_known_context_calls}")
    print(f"Known calls from which the gender of the emitter is known: {nb_known_call_genders}")

from pamai.utils.signal_processing import extract_label_bat

def create_reference_lists_from_path(sample,label,ws,fs):
    """ Creates an annotation file for visualizing in sed_vis
    Args:
        samples : name of the audio files being processed
        label   : labels available for the audio files
        ws      : window size 
    Return :
        Nothing
    """
    res = []
    start = 0
    audio,sr = torchaudio.load(sample)
    for b in range(0,audio.shape[1],ws): 
        i = b//ws
        data = extract_label_bat(label,b,b+ws)
        if data == 1 and start == 0: start = (i*ws)/fs
        elif data == 0 and start!= 0 : 
            res.append([start,(i*ws)/fs])
            start = 0
    content = ""
    for r1,r2 in res:
        content += f"{r1} {r2} batcall\n"
    with open(f"/home/arthur/Work/FlyingFoxes/sources/flying_foxes_study/AudioEventDetection/DENet/assets/sound_vis/{sample.split('/')[-1][:-4]}.ann","w") as f:
        f.write(content)

    return 

def create_estimated_lists_from_output(path,audio,label,ws,fs):
    """ Creates an annotation file for visualizing in sed_vis
    Args:
        sample: name of the audio file being processed
        output : labels predicted by the NN 
    Return :
        Nothing
    """
    res = []
    start = 0
    for i in range(0,audio.shape[0]): 
        data = torch.argmax(audio[i])
        if data == 1 and start == 0: start = (i*ws)/fs
        elif data == 0 and start!= 0 : 
            res.append([start,(i*ws)/fs])
            start = 0
    content = ""
    for r1,r2 in res:
        content += f"{r1} {r2} batcall\n"
    with open(f"/home/arthur/Work/FlyingFoxes/sources/flying_foxes_study/AudioEventDetection/DENet/assets/sound_vis/{path.split('/')[-1][:-4]}_pred.ann","w") as f:
        f.write(content)

if __name__ == '__main__':
    path = "/home/arthur/Work/FlyingFoxes/database/EgyptianFruitBats"
    data = sub_sample_dataset(path+"/Annotations.csv",path + "/FileInfo.csv")
    show_info(data)