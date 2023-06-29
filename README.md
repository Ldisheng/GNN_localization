# GNN_localization
Folder "Dataset" includes two raytracing CSI datasets based on the open NTU 3D map and real BS positions, simulated by Wireless Insite. Currently, these datasets are augmented by 40dB Gaussian noise.

Dataset.py build undirected graphs based on these CSI data, taking BS as the vertices and the reciporcal of distance between BS as edge weight.

Main.py implemented three different GNNs (GCN,GIN and GraphSAGE) and MLP on above datasets for classification and regression, respectively.
