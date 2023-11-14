import tqdm
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import kl_divergence
from torch.distributions.multivariate_normal import MultivariateNormal as Normal
import math
import os 
import pickle

font = {'family' : 'serif',
        'size'   : 20,
        'weight': 'bold'}

plt.rc('text', usetex=True)
plt.rc('font', **font)
plt.rc('text.latex', preamble=r'\usepackage{bm}')

color_palette_1 = ['#335c67','#fff3b0','#e09f3e','#9e2a2b','#540b0e']
color_palette_2 = ['#177e89','#084c61','#db3a34','#ef8354','#323031']
color_palette_3 = ['#bce784','#5dd39e','#348aa7','#525274','#513b56']
color_palette_4 = ['#002642','#840032','#e59500','#e5dada','#02040e']
color_palette_5 = ['#202c39','#283845','#b8b08d','#f2d449','#f29559']

palette_red = ["#03071e","#370617","#6a040f","#9d0208","#d00000","#dc2f02","#e85d04","#f48c06","#faa307","#ffba08"]
palette_blue = ["#012a4a","#013a63","#01497c","#014f86","#2a6f97","#2c7da0","#468faf","#61a5c2","#89c2d9","#a9d6e5"]
palette_green = ['#99e2b4','#88d4ab','#78c6a3','#67b99a','#56ab91','#469d89','#358f80','#248277','#14746f','#036666']
palette_pink = ["#ea698b","#d55d92","#c05299","#ac46a1","#973aa8","#822faf","#6d23b6","#6411ad","#571089","#47126b"]
palette_super_red = ["#641220","#6e1423","#85182a","#a11d33","#a71e34","#b21e35","#bd1f36","#c71f37","#da1e37","#e01e37"]

palette = color_palette_2
meta_color = 'r'


if __name__ == "__main__":
    
#     loss_mnist1 = [(2.3167), (0.0735), (0.0702), (0.0695), (0.0693), (0.0692), (0.0692), (0.0692), (0.0692), (0.0691), (0.0691), (0.0691), (0.0691), (0.0691), (0.0691), (0.0691), (0.0691), (0.0691), (0.0691), (0.0690), (0.0690), (0.0690), (0.0690), (0.0690), (0.0690), (0.0690), (0.0690), (0.0690), (0.0690), (0.0690)]
#     loss_mnist2 = [(2.3158), (0.0715), (0.0695), (0.0691), (0.0690), (0.0690), (0.0690), (0.0689), (0.0689), (0.0689), (0.0689), (0.0689), (0.0689), (0.0689), (0.0689), (0.0689), (0.0689), (0.0689), (0.0689), (0.0689), (0.0689), (0.0689), (0.0689), (0.0689), (0.0689), (0.0689), (0.0689), (0.0689), (0.0689), (0.0689)]
#     loss_mnist3 = [(2.3118), (0.0735), (0.0706), (0.0699), (0.0697), (0.0695), (0.0694), (0.0693), (0.0693), (0.0692), (0.0692), (0.0692), (0.0691), (0.0691), (0.0691), (0.0691), (0.0690), (0.0690), (0.0690), (0.0690), (0.0690), (0.0690), (0.0690), (0.0690), (0.0690), (0.0690), (0.0690), (0.0690), (0.0690), (0.0690)]
   
#     l = [loss_mnist1, loss_mnist2, loss_mnist3]

    loss_mnist = [2.3158, 0.1203, 0.0811, 0.0724, 0.0689, 0.0671, 0.0660, 0.0653, 0.0648, 0.0644, 0.0642, 0.0640, 0.0638, 0.0637, 0.0636, 0.0635, 0.0634, 0.0634, 0.0633, 0.0633, 0.0632, 0.0632, 0.0632, 0.0632, 0.0632, 0.0631, 0.0631, 0.0631, 0.0631, 0.0631, 0.0631, 0.0631, 0.0631, 0.0631, 0.0631, 0.0631, 0.0631, 0.0631, 0.0631, 0.0630, 0.0630, 0.0630, 0.0630, 0.0630, 0.0630, 0.0630, 0.0630, 0.0630, 0.0630, 0.0630, 0.0630]

    plt.figure(figsize=(8, 6))
    plt.plot(torch.arange(len(loss_mnist)), [0.07197153568267822]*(len(loss_mnist)),'--', lw=2.0, color=palette_green[3], alpha=0.9, label="Isotropic")
    plt.plot(torch.arange(len(loss_mnist)), [0.06297246366739273]*(len(loss_mnist)),'--', lw=2.0, color=palette_red[0], alpha=0.9, label="Fisher")
    plt.plot(loss_mnist, lw=3, alpha=0.6, color=palette_red[3+3], label="SSF")
    # plt.plot(loss_mnist4, lw=4.5, alpha=0.4, color=palette_blue[3+4])
    plt.xlim(0, len(loss_mnist)-20)

    plt.ylim(0,max(loss_mnist))
    num_points = len(loss_mnist)
    print(num_points)
    x_ticks = np.arange(0, num_points, num_points // 2)
    x_labels = [str(x * 100) for x in x_ticks]
    plt.xticks(x_ticks, x_labels)
    plt.xlabel(r'Epochs')
    plt.ylabel(r'Test loss')
    plt.title(r'MNIST')
    plt.legend(loc='upper right') 
    plt.show()
