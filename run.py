import os
from tools.load_data import load_data
from tools import Form_data
import torch
import numpy as np
from train import train_opt
from pretrain import pretrain_opt
from graph_embedding.spectralnet import SpectralNet

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


setup_seed(0)
import time

if __name__ == '__main__':
    raw_data, label, c = load_data()
    K = 3

    dataname = 'Chen_high'

    if not os.path.exists('Intermediate_data/' + dataname):
        os.makedirs('Intermediate_data/' + dataname)
    if not os.path.exists('result/' + dataname):
        os.makedirs('result/' + dataname)

    for t in range(1):
        setup_seed(t)

        start = time.perf_counter()

        # pretrain
        pretrain_opt(raw_data, label, c, dataname)

        train_graph = True
        if train_graph:
            spectralNet = SpectralNet(n_clusters=c, should_use_ae=True, should_use_siamese=True, ae_hiddens=[512, c],
                                    siamese_hiddens=[256, c], spectral_hiddens=[256, c])
            data_cat = torch.cat(raw_data, dim=1)
            spectralNet.fit(data_cat)
            spectralNet_dist = spectralNet.predict(data_cat)  # Get the final assignments to clusters
            
            torch.save(spectralNet_dist, 'Intermediate_data/' + dataname + '/SpectralNet_D.pkl')
        else:
            spectralNet_dist = torch.load('Intermediate_data/' + dataname + '/SpectralNet_D.pkl')

        neighbors = Form_data.cal_neighbors_D(spectralNet_dist, K)
        data = Form_data.form_data(raw_data, label, neighbors)

        acc, NMI, ARI, pur, FMI = train_opt(data, label, dataname)

        end = time.perf_counter()
        runTime = end - start
        print('| ACC = {:6f} NMI = {:6f} ARI = {:6f} Purity = {:6f} FMI = {:6f} TIME = {:6f}'.format(acc, NMI, ARI, pur, FMI, runTime))
