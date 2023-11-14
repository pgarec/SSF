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
    
    loss_fmnist = [2.3085, 0.7197, 0.5493, 0.4994, 0.4795, 0.4697, 0.4642, 0.4607, 0.4584, 0.4567, 0.4554, 0.4545, 0.4537, 0.4531, 0.4526, 0.4521, 0.4517, 0.4514, 0.4511, 0.4509, 0.4506, 0.4504, 0.4503, 0.4501, 0.4499, 0.4498, 0.4496, 0.4495, 0.4494, 0.4493, 0.4492, 0.4491, 0.4490, 0.4489, 0.4488, 0.4488, 0.4487, 0.4486, 0.4486, 0.4485, 0.4484, 0.4484, 0.4483, 0.4483, 0.4482, 0.4482, 0.4481, 0.4481, 0.4480, 0.4480, 0.44794]

    plt.figure(figsize=(8, 6))
    plt.plot(torch.arange(len(loss_fmnist)), [0.4497138261795044]*(len(loss_fmnist)),'--', lw=2.0, color=palette_green[3], alpha=0.9, label="Isotropic")
    plt.plot(torch.arange(len(loss_fmnist)), [0.4461183249950409]*(len(loss_fmnist)),'--', lw=2.0, color=palette_red[0], alpha=0.9, label="Fisher")
    plt.plot(loss_fmnist, lw=3, alpha=0.6, color=palette_red[3+3], label="SSF")
    plt.xlim(0, len(loss_fmnist)-20)

    plt.ylim(0,max(loss_fmnist))
    num_points = len(loss_fmnist)
    print(num_points)
    x_ticks = np.arange(0, num_points, num_points // 2)
    x_labels = [str(x * 100) for x in x_ticks]
    plt.xticks(x_ticks, x_labels)
    plt.xlabel(r'Epochs')
    plt.ylabel(r'Test loss')
    plt.title(r'Fashion-MNIST')
    plt.legend(loc='upper right') 
    plt.show()
