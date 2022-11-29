import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import math
import re
import pylab
from pylab import figure, show, legend
from mpl_toolkits.axes_grid1 import host_subplot
train_loss = []
epoch = []
fp = open('new.log', 'r')
for ln in fp:
  # get train_iterations and train_loss
  if 'Training' in ln and 'loss' in ln:
    arr=re.findall('\d+', ln)
    loss_temp = arr[-2] + '.' +arr[-1]
    loss_temp = eval(loss_temp)
    epoch_temp = arr[-4]
    epoch_temp = eval(epoch_temp)
    loss_temp = math.log10(loss_temp)
    train_loss.append(loss_temp)
    epoch.append(epoch_temp)
print(train_loss)
print(epoch)


plt.plot(epoch,train_loss)
plt.show()
