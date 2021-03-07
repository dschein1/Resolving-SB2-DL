import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import scipy as sp
from scipy import interpolate
from matplotlib import pyplot

class SpectraDataset(Dataset):
    def __init__(self, index_list,path = "/home/drors/project/sampled/target data"):
        self.tables = {}
        self.standard = {}
        os.chdir(path)
        self.index_list = index_list
        root = os.getcwd()
        for file_name in os.listdir():
            file_path = os.path.join(root,file_name)
            file_read = pd.read_csv(file_path)
            file_name_split = file_name.split('_')
            index = int(file_name_split[0])
            if index in index_list:
                if file_name_split[2] == 'ts':
                    file_type = file_name_split[1] + " " + file_name_split[2]
                    column_list = [column for column in file_read.columns if "s_" in column]
                    file_read = file_read.dropna(subset = column_list)
                else:
                    file_type = file_name_split[2]
                if index in self.tables:
                    self.tables[index][file_type] = file_read
                else:
                    self.tables[index] = {file_type : file_read}
        to_remove = [] #cant change the dictionary while iterating over it
        for index in self.tables:
            if len(self.tables[index]) < 3 and index_list != [1]:
                to_remove.append(index)
        for index in to_remove:
            self.tables.pop(index)
            self.index_list.remove(index)
        self.get_standard_vector()
        #self.normalize_vectors()
        
    def get_standard_vector(self):
        min_source = -1000000
        min_target = -1000000
        max_source = 1000000 
        max_target = 1000000
        length_source = 1000000
        length_target = 1000000
        for index in self.tables:
            columns = [column for column in self.tables[index]['0 ts'].columns if "w_" in column]
            mins_source = self.tables[index]['0 ts'][columns].min().max()
            maxs_source = self.tables[index]['0 ts'][columns].max().min()
            sources_length = len(self.tables[index]['0 ts'][columns].index)
            columns = [column for column in self.tables[index]['ORIG ts'].columns if "w_" in column]
            mins_target = self.tables[index]['ORIG ts'][columns].min().max()
            maxs_target = self.tables[index]['ORIG ts'][columns].max().min()
            targets_length = len(self.tables[index]['ORIG ts'][columns].index)

            if mins_source > min_source:
                min_source = mins_source
            if mins_target > min_target:
                min_target = mins_target
            if max_source > maxs_source:
                max_source = maxs_source
            if max_target > maxs_target:
                max_target = maxs_target
            if length_source > sources_length:
                length_source = sources_length
            if length_target > targets_length:
                length_target = targets_length
        self.standard["min source"] = min_source
        self.standard["max source"] = max_source
        self.standard["min target"] = min_target
        self.standard["max target"] = max_target
        self.standard["source len"] = length_source
        self.standard["target len"] = length_target
    
    def normalize_vectors(self):
        self.data = {}
        for index in self.tables:
            self.data[index] = {}
            columns_w = [column for column in self.tables[index]['0 ts'].columns if "w_" in column]
            columns_s = [column for column in self.tables[index]['0 ts'].columns if "s_" in column]
            self.data[index]['source'] = np.zeros((20,self.standard["source len"]))
            for i,(w,s)  in enumerate(zip(columns_w,columns_s)):
                x = self.tables[index]['0 ts'][w].to_numpy(dtype = np.float32)
                y = self.tables[index]['0 ts'][s].to_numpy(dtype = np.float32)
                z = interpolate.interp1d(x, y, kind='quadratic',fill_value="extrapolate")
                self.data[index]['source'][i,:] = z(self.standard["vector source"])
            columns_w = [column for column in self.tables[index]['ORIG ts'].columns if "w_" in column]
            columns_s = [column for column in self.tables[index]['ORIG ts'].columns if "s_" in column]
            self.data[index]['target'] = np.zeros((2,self.standard["target len"]))
            for i,(w,s)  in enumerate(zip(columns_w,columns_s)):
                x = self.tables[index]['ORIG ts'][w].to_numpy(dtype = np.float32)
                y = self.tables[index]['ORIG ts'][s].to_numpy(dtype = np.float32)
                z = interpolate.interp1d(x, y, kind='quadratic', fill_value="extrapolate")
                self.data[index]['target'][i,:] = (z(self.standard["vector target"])).astype(np.float32)
        
    def __len__(self):
        return len(self.index_list) * 20
    def __getitem__(self,idx):
        #idx = idx + 20
        table_idx = idx // 20
        table_idx = self.index_list[table_idx]
        #x = self.tables[table_idx]['0 ts']
        #x = x[f's_{idx % 20}']
        x = self.data[table_idx]['source'][idx % 20,:]
        #y = self.tables[table_idx]['ORIG ts']
        #y = y[['s_0','s_1']]
        y = self.data[table_idx]['target']
        return idx,x,y
    
    def getTimeVector(self,table_idx):
        base = self.tables[table_idx]['vrad']
        #column_list = [f'time_{i}' for i in range(20)]
        #print(base[column_list])
        return base.iloc[2,1:].to_numpy(dtype = np.float32)
    def getSpeedMatrix(self, table_idx):
        base = self.tables[table_idx]['vrad']
        return base.iloc[0:2,1:].to_numpy(dtype = np.float32)
    def additive_collate(self,batch):
        table_idx = batch[0][0] // 20
        table_idx = self.index_list[table_idx]
        times = self.getTimeVector(table_idx)
        readings = np.asarray([row[1] for row in batch], dtype=np.float32)
        real = (batch[1][2]).astype(np.float32)
        speeds = self.getSpeedMatrix(table_idx)
        return torch.from_numpy(readings),torch.from_numpy(real),table_idx,torch.from_numpy(times), torch.from_numpy(speeds)
    def set_standard(self,values):
        self.standard["min source"] = values["min source"]
        self.standard["max source"] = values["max source"]
        self.standard["min target"] = values["min target"]
        self.standard["max target"] = values["max target"]
        self.standard["vector source"] = values["vector source"]
        self.standard["vector target"] = values["vector target"]
        self.standard["source len"] = values["source len"]
        self.standard["target len"] = values["target len"]    
def getLoader(seed = None):
    if seed != 0:
        np.random.seed(seed)
    perm = np.random.permutation(np.arange(1,121))
    train = perm[0:90].tolist()
    test = perm[90:105].tolist()
    val = perm[105:121].tolist()
    train_set = SpectraDataset(train)
    test_set = SpectraDataset(test)
    val_set = SpectraDataset(val)
    standard_values = {}
    standard_values["min source"] = max(train_set.standard["min source"],val_set.standard["min source"],test_set.standard["min source"])
    standard_values["max source"] = min(train_set.standard["max source"],val_set.standard["max source"],test_set.standard["max source"])
    standard_values["min target"] = max(train_set.standard["min target"],val_set.standard["min target"],test_set.standard["min target"])
    standard_values["max target"] = min(train_set.standard["max target"],val_set.standard["max target"],test_set.standard["max target"])
    standard_values["source len"] = min(train_set.standard["source len"],val_set.standard["source len"],test_set.standard["source len"])
    standard_values["target len"] = min(train_set.standard["target len"],val_set.standard["target len"],test_set.standard["target len"])
    standard_values["source len"] = (standard_values["source len"] // 100) * 100
    standard_values["target len"] = (standard_values["target len"] // 100) * 100
    standard_values["vector source"] = np.linspace(standard_values["min source"],standard_values["max source"], num = standard_values["source len"])
    standard_values["vector target"] = np.linspace(standard_values["min target"],standard_values["max target"], num = standard_values["target len"])
    
    train_set.set_standard(standard_values)
    test_set.set_standard(standard_values)
    val_set.set_standard(standard_values)
    
    
    train_set.normalize_vectors()
    test_set.normalize_vectors()
    val_set.normalize_vectors()
    
    train_loader = DataLoader(train_set, batch_size = 20, collate_fn = train_set.additive_collate)
    test_loader = DataLoader(test_set, batch_size = 20, collate_fn = test_set.additive_collate)
    val_loader = DataLoader(val_set, batch_size = 20, collate_fn = val_set.additive_collate)
    return train_loader, test_loader, val_loader, standard_values

def get_real_loader(std_vector):
    dataset = SpectraDataset(index_list = [1], path = "/home/drors/project/real data sampled")
    x = torch.zeros((20,std_vector['source len']))
    y = torch.zeros((2,std_vector['target len']))
    dataset.set_standard(std_vector)
    dataset.normalize_vectors()
    for i,(_,source,target) in enumerate(dataset):
        x[i,:] = torch.from_numpy(source)
        y = torch.from_numpy(target)
    return x,y

#def getLoader(seed = None):
#    if seed != 0:
#        np.random.seed(seed)
#    perm = np.random.permutation(np.arange(1,121))
#    train = perm[0:90].tolist()
#    test = perm[90:105].tolist()
#    val = perm[105:121].tolist()
#    train_set = SpectraDataset(train)
#    test_set = SpectraDataset(test)
#    val_set = SpectraDataset(val)
#    train_loader = DataLoader(train_set, batch_size = 20, collate_fn = train_set.additive_collate)
#    test_loader = DataLoader(test_set, batch_size = 20, collate_fn = test_set.additive_collate)
#    val_loader = DataLoader(val_set, batch_size = 20, collate_fn = val_set.additive_collate)
#    return train_loader, test_loader, val_loader
# class SpectraDataset(Dataset):
#     def __init__(self, index_list,path = "/home/drors/project/sampled/target data"):
#         self.tables = {}
#         os.chdir(path)
#         self.index_list = index_list
#         root = os.getcwd()
#         for file_name in os.listdir():
#             file_path = os.path.join(root,file_name)
#             file_read = pd.read_csv(file_path)
#             file_name_split = file_name.split('_')
#             index = int(file_name_split[0])
#             if index in index_list:
#                 if file_name_split[2] == 'ts':
#                     file_type = file_name_split[1] + " " + file_name_split[2]
#                     column_list = [column for column in file_read.columns if "s_" in column]
#                     file_read = file_read.dropna(subset = column_list)
#                 else:
#                     file_type = file_name_split[2]
#                 if index in self.tables:
#                     self.tables[index][file_type] = file_read
#                 else:
#                     self.tables[index] = {file_type : file_read}
#         to_remove = [] #cant change the dictionary while iterating over it
#         for index in self.tables:
#             if len(self.tables[index]) < 3:
#                 to_remove.append(index)
#         for index in to_remove:
#             self.tables.pop(index)
#             self.index_list.remove(index)
#     def __len__(self):
#         return len(self.index_list) * 20
#     def __getitem__(self,idx):
#         #idx = idx + 20
#         table_idx = idx // 20
#         table_idx = self.index_list[table_idx]
#         x = self.tables[table_idx]['0 ts']
#         x = x[f's_{idx % 20}']
#         y = self.tables[table_idx]['ORIG ts']
#         y = y[['s_0','s_1']]
#         return idx,x.to_numpy(dtype = np.float32),y.to_numpy(dtype = np.float32)
    
#     def getTimeVector(self,table_idx):
#         base = self.tables[table_idx]['vrad']
#         #column_list = [f'time_{i}' for i in range(20)]
#         #print(base[column_list])
#         return base.iloc[2,1:].to_numpy(dtype = np.float32)
#     def getSpeedMatrix(self, table_idx):
#         base = self.tables[table_idx]['vrad']
#         return base.iloc[0:2,1:].to_numpy(dtype = np.float32)
#     def additive_collate(self,batch):
#         table_idx = batch[0][0] // 20
#         table_idx = self.index_list[table_idx]
#         times = self.getTimeVector(table_idx)
#         readings = np.asarray([row[1] for row in batch])
#         real = batch[1][2]
#         speeds = self.getSpeedMatrix(table_idx)
#         return torch.from_numpy(readings),torch.from_numpy(real),table_idx,torch.from_numpy(times), torch.from_numpy(speeds)
# def getLoader(seed = None):
#     if seed != 0:
#         np.random.seed(seed)
#     perm = np.random.permutation(np.arange(1,121))
#     train = perm[0:90].tolist()
#     test = perm[90:105].tolist()
#     val = perm[105:121].tolist()
#     train_set = SpectraDataset(train)
#     test_set = SpectraDataset(test)
#     val_set = SpectraDataset(val)
#     train_loader = DataLoader(train_set, batch_size = 20, collate_fn = train_set.additive_collate)
#     test_loader = DataLoader(test_set, batch_size = 20, collate_fn = test_set.additive_collate)
#     val_loader = DataLoader(val_set, batch_size = 20, collate_fn = val_set.additive_collate)
#     return train_loader, test_loader, val_loader