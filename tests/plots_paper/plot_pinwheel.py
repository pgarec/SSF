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
        'size'   : 20}

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
    
    # loss_pinwheel = [(1.5502), (0.0943), (0.0935), (0.0934), (0.0933), (0.0932), (0.0932), (0.0932), (0.0932), (0.0932), (0.0932), (0.0932), (0.0931), (0.0931), (0.0931), (0.0931), (0.0931), (0.0931), (0.0931), (0.0931), (0.0931), (0.0931), (0.0931), (0.0931), (0.0931), (0.0931), (0.0931), (0.0931), (0.0931), (0.0931)]
    # seed 450, 3 models, 5000 steps, 20 perms
    # Istropic model loss: 0.40588057041168213, Fisher model loss: 0.2160848379135131
    loss_pinwheel = [1.5786, 0.4309, 0.3173, 0.2773, 0.2576, 0.2459, 0.2385, 0.2335, 0.2299, 0.2273, 0.2254, 0.2239, 0.2227, 0.2217, 0.2209, 0.2202, 0.2196, 0.2191, 0.2186, 0.2183, 0.2179, 0.2176, 0.2173, 0.2171, 0.2168, 0.2166, 0.2164, 0.2163, 0.2161, 0.2160, 0.2158, 0.2157, 0.2156, 0.2154, 0.2153, 0.2152, 0.2151, 0.2150, 0.2150, 0.2149, 0.2148, 0.2147, 0.2147, 0.2146, 0.2145, 0.2145, 0.2144, 0.2144, 0.2143, 0.2143]
    n = len(loss_pinwheel)

    plt.figure(figsize=(8, 6))
    plt.plot(torch.arange(len(loss_pinwheel)), [0.4059]*(len(loss_pinwheel)),'--', lw=2.0, color=palette_green[3], alpha=0.9, label="Isotropic")
    plt.plot(torch.arange(len(loss_pinwheel)), [0.2160]*(len(loss_pinwheel)),'--', lw=2.0, color=palette_red[0], alpha=0.9, label="Fisher")
    plt.plot(loss_pinwheel, lw=3, alpha=0.6, color=palette_red[3+3], label='SSF')

    plt.ylim(0,max(loss_pinwheel))
    num_points = len(loss_pinwheel)
    x_ticks = np.arange(0, num_points, num_points // 3)
    x_labels = [str(x * 100) for x in x_ticks]
    plt.xticks(x_ticks, x_labels)
    plt.xlim(0,len(loss_pinwheel)-1)
    plt.xlabel(r'Steps')
    plt.ylabel(r'Test loss')
    plt.title(r'Pinwheel')
    plt.legend(loc='upper right') 
    plt.show()
