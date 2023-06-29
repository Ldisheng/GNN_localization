import torch
from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.data import Data
import numpy

#这里给出大家注释方便理解
class Route(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
    #返回数据集源文件名
    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2', ...]
    #返回process方法所需的保存文件名。你之后保存的数据集名字和列表里的一致
    @property
    def processed_file_names(self):
        return ['data.pt']
    # #用于从网上下载数据集
    # def download(self):
    #     # Download to `self.raw_dir`.
    #     download_url(url, self.raw_dir)
        ...
    #生成数据集所用的方法
    def process(self):
        # Read data into huge `Data` list.
        # Read data into huge `Data` list.
        # 这里用于构建data
        Edge_index = torch.tensor([[0, 1, 0, 2, 0, 3, 0, 4, 1, 2, 1, 3, 1, 4, 2, 3, 2, 4, 3, 4],
                                   [1, 0, 2, 0, 3, 0, 4, 0, 2, 1, 3, 1, 4, 1, 3, 2, 4, 2, 4, 3]], dtype=torch.long)

        Edge_weight=torch.tensor([[0.7876],[0.7876],[2.2541],[2.2541],[2.8206],[2.8206],[1.4521],[1.4521],[0.8540],[0.8540],[0.6844],[0.6844],[0.6815],[0.6815],
                                [2.9325],[2.9325],[2.9173],[2.9173],[2.5500],[2.5500]],dtype=torch.float32)




        with open("Dataset/Classfication/route144680.txt", mode="rt", encoding="utf-8") as f:
            #res0 = f.readlines()
            string = f.read()  # string是一行字符串，该字符串包含文件所有内容
            f.close()
            row_list = string.splitlines()  # splitlines默认参数是‘\n’
            data_list = [[float(i) for i in row.strip().split("\t")] for row in row_list]
            res0=data_list
        with open("Dataset/Classfication/route734737.txt", mode="rt", encoding="utf-8") as f:
            #res1 = f.readlines()
            string = f.read()  # string是一行字符串，该字符串包含文件所有内容
            f.close()
            row_list = string.splitlines()  # splitlines默认参数是‘\n’
            data_list = [[float(i) for i in row.strip().split("\t")] for row in row_list]
            res1=data_list
        with open("Dataset/Classfication/route734817.txt", mode="rt", encoding="utf-8") as f:
            #res2 = f.readlines()
            string = f.read()  # string是一行字符串，该字符串包含文件所有内容
            f.close()
            row_list = string.splitlines()  # splitlines默认参数是‘\n’
            data_list = [[float(i) for i in row.strip().split("\t")] for row in row_list]
            res2=data_list
        with open("Dataset/Classfication/route748823.txt", mode="rt", encoding="utf-8") as f:
            #res3 = f.readlines()
            string = f.read()  # string是一行字符串，该字符串包含文件所有内容
            f.close()
            row_list = string.splitlines()  # splitlines默认参数是‘\n’
            data_list = [[float(i) for i in row.strip().split("\t")] for row in row_list]
            res3=data_list
        with open("Dataset/Classfication/route749197.txt", mode="rt", encoding="utf-8") as f:
            #res4 = f.readlines()
            string = f.read()  # string是一行字符串，该字符串包含文件所有内容
            f.close()
            row_list = string.splitlines()  # splitlines默认参数是‘\n’
            data_list = [[float(i) for i in row.strip().split("\t")] for row in row_list]
            res4=data_list
        with open("Dataset/Classfication/label_route.txt", mode="rt", encoding="utf-8") as f:
            #label = f.readlines()
            string = f.read()  # string是一行字符串，该字符串包含文件所有内容
            f.close()
            row_list = string.splitlines()  # splitlines默认参数是‘\n’
            label = [[float(i) for i in row.strip().split("\t")] for row in row_list]
            data_list=[]
        for i in range(0, 39100):
            x0=res0[i]
            x1=res1[i]
            x2=res2[i]
            x3=res3[i]
            x4=res4[i]
            y=label[i]
            # 每个节点的特征：从0号节点开始。。
            X = torch.from_numpy(numpy.array([x0, x1, x2, x3, x4])).to(torch.float32)
            # 每个节点的标签：从0号节点开始-两类0，1
            Y = torch.from_numpy(numpy.array(y)).to(torch.int)
            data = Data(x=X, edge_index=Edge_index, y=Y,edge_weight=Edge_weight)
            # 放入datalist
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class Coordinate(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
    #返回数据集源文件名
    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2', ...]
    #返回process方法所需的保存文件名。你之后保存的数据集名字和列表里的一致
    @property
    def processed_file_names(self):
        return ['data.pt']
    # #用于从网上下载数据集
    # def download(self):
    #     # Download to `self.raw_dir`.
    #     download_url(url, self.raw_dir)
        ...
    #生成数据集所用的方法
    def process(self):
        # Read data into huge `Data` list.
        # Read data into huge `Data` list.
        # 这里用于构建data
        Edge_index = torch.tensor([[0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 1, 2, 1, 3, 1, 4, 1, 5, 1, 6, 2, 3, 2, 4, 2, 5, 2, 6, 3, 4, 3, 5, 3, 6, 4, 5, 4, 6, 5, 6],
                                   [1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 2, 1, 3, 1, 4, 1, 5, 1, 6, 1, 3, 2, 4, 2, 5, 2, 6, 2, 4, 3, 5, 3, 6, 3, 5, 4, 6, 4, 6, 5]], dtype=torch.long)

        Edge_weight=torch.tensor([[3.4902],[3.4902],[2.0954],[2.0954],[1.8219],[1.8219],[14.8677],[14.8677],[2.4568],[2.4568],[2.6131],[2.6131],[2.6665],[2.6665],
                                [1.4153],[1.4153],[4.5548],[4.5548],[8.0225],[8.0225],[3.1624],[3.1624],[1.9396],[1.9396],[2.2541],[2.2541],[2.8206],[2.8206]
                                  ,[1.4521],[1.4521],[1.7188],[1.7188],[1.2995],[1.2995],[1.0816],[1.0816],[2.9325],[2.9325],[2.9173],[2.9173],[2.5500],[2.5500]],dtype=torch.float32)




        with open("Dataset/Regression/cor137410.txt", mode="rt", encoding="utf-8") as f:
            #res0 = f.readlines()
            string = f.read()  # string是一行字符串，该字符串包含文件所有内容
            f.close()
            row_list = string.splitlines()  # splitlines默认参数是‘\n’
            data_list = [[float(i) for i in row.strip().split("\t")] for row in row_list]
            res0=data_list
        with open("Dataset/Regression/cor143850.txt", mode="rt", encoding="utf-8") as f:
            #res1 = f.readlines()
            string = f.read()  # string是一行字符串，该字符串包含文件所有内容
            f.close()
            row_list = string.splitlines()  # splitlines默认参数是‘\n’
            data_list = [[float(i) for i in row.strip().split("\t")] for row in row_list]
            res1=data_list
        with open("Dataset/Regression/cor144680.txt", mode="rt", encoding="utf-8") as f:
            #res2 = f.readlines()
            string = f.read()  # string是一行字符串，该字符串包含文件所有内容
            f.close()
            row_list = string.splitlines()  # splitlines默认参数是‘\n’
            data_list = [[float(i) for i in row.strip().split("\t")] for row in row_list]
            res2=data_list
        with open("Dataset/Regression/cor734777.txt", mode="rt", encoding="utf-8") as f:
            #res3 = f.readlines()
            string = f.read()  # string是一行字符串，该字符串包含文件所有内容
            f.close()
            row_list = string.splitlines()  # splitlines默认参数是‘\n’
            data_list = [[float(i) for i in row.strip().split("\t")] for row in row_list]
            res3=data_list
        with open("Dataset/Regression/cor734817.txt", mode="rt", encoding="utf-8") as f:
            #res4 = f.readlines()
            string = f.read()  # string是一行字符串，该字符串包含文件所有内容
            f.close()
            row_list = string.splitlines()  # splitlines默认参数是‘\n’
            data_list = [[float(i) for i in row.strip().split("\t")] for row in row_list]
            res4=data_list
        with open("Dataset/Regression/cor748823.txt", mode="rt", encoding="utf-8") as f:
            #res4 = f.readlines()
            string = f.read()  # string是一行字符串，该字符串包含文件所有内容
            f.close()
            row_list = string.splitlines()  # splitlines默认参数是‘\n’
            data_list = [[float(i) for i in row.strip().split("\t")] for row in row_list]
            res5=data_list
        with open("Dataset/Regression/cor749197.txt", mode="rt", encoding="utf-8") as f:
            #res4 = f.readlines()
            string = f.read()  # string是一行字符串，该字符串包含文件所有内容
            f.close()
            row_list = string.splitlines()  # splitlines默认参数是‘\n’
            data_list = [[float(i) for i in row.strip().split("\t")] for row in row_list]
            res6=data_list
        with open("Dataset/Regression/label_cor.txt", mode="rt", encoding="utf-8") as f:
            #res4 = f.readlines()
            string = f.read()  # string是一行字符串，该字符串包含文件所有内容
            f.close()
            row_list = string.splitlines()  # splitlines默认参数是‘\n’
            data_list = [[float(i) for i in row.strip().split("\t")] for row in row_list]
            label=data_list
        # with open("label_route.txt", mode="rt", encoding="utf-8") as f:
        #     #label = f.readlines()
        #     string = f.read()  # string是一行字符串，该字符串包含文件所有内容
        #     f.close()
        #     row_list = string.splitlines()  # splitlines默认参数是‘\n’
        #     label = [[float(i) for i in row.strip().split("\t")] for row in row_list]
            data_list=[]
        for i in range(0, 27700):
            x0=res0[i]
            x1=res1[i]
            x2=res2[i]
            x3=res3[i]
            x4=res4[i]
            x5=res5[i]
            x6=res6[i]
            y=label[i]
            # 每个节点的特征：从0号节点开始。。
            X = torch.from_numpy(numpy.array([x0, x1, x2, x3, x4,x5,x6])).to(torch.float32)
            # 每个节点的标签：从0号节点开始-两类0，1
            Y = torch.from_numpy(numpy.array([y])).to(torch.float32)
            data = Data(x=X, edge_index=Edge_index, y=Y,edge_weight=Edge_weight)
            # 放入datalist
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


b = Route(root="./route/processed").shuffle()
a = Coordinate(root="./coordinate/processed").shuffle()