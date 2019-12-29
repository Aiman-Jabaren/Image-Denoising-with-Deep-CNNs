#%matplotlib notebook
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as td
import torchvision as tv
from PIL import Image
import matplotlib.pyplot as plt
import nntools as nt


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

#Part 1
dataset_root_dir = '/datasets/ee285f-public/bsds/'



# Part 2

class NoisyBSDSDataset(td.Dataset):
    def __init__(self, root_dir, mode='train', image_size=(180, 180), sigma=30):
        super(NoisyBSDSDataset, self).__init__()
        self.mode = mode
        self.image_size = image_size
        self.sigma = sigma
        self.images_dir = os.path.join(root_dir, mode)
        self.files = os.listdir(self.images_dir)
    def __len__(self):
        return len(self.files)
    def __repr__(self):
        return "NoisyBSDSDataset(mode={}, image_size={}, sigma={})". \
            format(self.mode, self.image_size, self.sigma)
    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.files[idx])
        clean= Image.open(img_path).convert('RGB')
        i = np.random.randint(clean.size[0] - self.image_size[0])
        j = np.random.randint(clean.size[1] - self.image_size[1])
        

        clean = clean.crop((i,j,self.image_size[0]+i,self.image_size[1]+j)) #clean1[i:i+180,j:j+180]

        transform = tv.transforms.Compose([
 
            tv.transforms.ToTensor(), #Normalizes the image 
            tv.transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
            ])
        
        clean = transform(clean)        

        noisy = clean + 2 / 255 * self.sigma * torch.randn(clean.shape)
        return noisy, clean
# Part 3
def myimshow(image, ax=plt):
    image = image.to('cpu').numpy()
    image = np.moveaxis(image, [0, 1, 2], [2, 0, 1])
    image = (image + 1) / 2
    image[image < 0] = 0
    image[image > 1] = 1
    h = ax.imshow(image)
    ax.axis('off')
    return h


train_set = NoisyBSDSDataset(root_dir = dataset_root_dir, mode = 'train', image_size=(180, 180), sigma=30)
test_set = NoisyBSDSDataset(root_dir = dataset_root_dir, mode = 'test', image_size=(320, 320), sigma=30)
x, y = train_set[12]

print(type(x))
plt.figure()
myimshow(x)
plt.figure()
myimshow(y)


class NNRegressor(nt.NeuralNetwork):
    def __init__(self):
        super(NNRegressor, self).__init__()
        self.MSELoss = nn.MSELoss()
    def criterion(self, y, d):
        return self.MSELoss(y, d)
#Part 5

class DnCNN(NNRegressor):
    def __init__(self, D, C=64):
        super(DnCNN, self).__init__()
        self.D = D
        self.conv = nn.ModuleList()
        self.conv.append(nn.Conv2d(3, C, 3, padding=1))

        nn.init.kaiming_normal_(self.conv[0].weight.data)
        
            
        self.bn = nn.ModuleList()
        for k in range(D):
            self.bn.append(nn.BatchNorm2d(C, C))
            self.conv.append(nn.Conv2d(64, C, 3, padding=1))
            
            nn.init.constant_(self.bn[k].weight.data, 1.25 * np.sqrt(C))
            nn.init.kaiming_normal_(self.conv[k+1].weight.data)

            
        self.conv.append(nn.Conv2d(C, 3, 3, padding=1))
        nn.init.kaiming_normal_(self.conv[D+1].weight.data)

            
 
            
    def forward(self, x):
        D = self.D
        h = F.relu(self.conv[0](x))
        for k in range(D):
            t = self.bn[k](self.conv[k+1](h))
            h = F.relu(t)
        
        y = self.conv[D+1](h) + x
        return y


x, y = train_set[-1]
print(x.shape)
x = x.unsqueeze(0)
print(x.shape)


D_list = [0,1,2,4,8]
for d in D_list:
    net = DnCNN(d)
    y = net.forward(x)
    
    print('D= ',d)
    plt.figure()
    plt.subplot(1,3,1)
    myimshow(x[0])
    #plt.figure()
    plt.subplot(1,3,2)    
    myimshow(y[0].detach())
       

    plt.subplot(1,3,3)
    myimshow(x[0] - y[0].detach())



D_list = [0,1,2,4,8]
for d in D_list:
    net = DnCNN(d)
    y = net.forward(x)
    
    
    plt.figure()
    plt.subplot(1,3,1)
    myimshow(x[0])
    #plt.figure()
    plt.subplot(1,3,2)    
    myimshow(y[0].detach())
       
    #plt.figure()
    plt.subplot(1,3,3)
    myimshow(x[0] - y[0].detach())




#Part 9

class DenoisingStatsManager(nt.StatsManager):
    def __init__(self):
        super(DenoisingStatsManager, self).__init__()
    def init(self):
        super(DenoisingStatsManager, self).init()
        self.running_psnr = 0
            
    def accumulate(self, loss, x, y, d):
        super(DenoisingStatsManager, self).accumulate(loss, x, y, d)

        n = y.numel()
        norm2 = torch.pow(torch.norm(y-d),2) # not sure if x or d TODO 
        self.running_psnr += 10*torch.log10(4*n/norm2)
        
        
        
    def summarize(self):
        loss = super(DenoisingStatsManager, self).summarize()
        avg_psnr = self.running_psnr/self.number_update
        return {'loss': loss, 'avg_psnr': avg_psnr}
#Part 10
D = 6
lr = 1e-3
net = DnCNN(D)
net = net.to(device)
adam = torch.optim.Adam(net.parameters(), lr=lr)
stats_manager = DenoisingStatsManager()
val_set = test_set
exp1 = nt.Experiment(net, train_set, val_set, adam, stats_manager,
                    output_dir="denoising21", batch_size=4, perform_validation_during_training=False)
exp2 = nt.Experiment(net, train_set, val_set, adam, stats_manager,
                    output_dir="denoising22", batch_size=4, perform_validation_during_training=False)
#Part 11
def plot(exp, fig, axes, noisy, visu_rate=2):
    if exp.epoch % visu_rate != 0:
        return
    with torch.no_grad():
        denoised = exp.net(noisy[np.newaxis].to(exp.net.device))[0]
    axes[0][0].clear()
    axes[0][1].clear()
    axes[1][0].clear()
    axes[1][1].clear()
    myimshow(noisy, ax=axes[0][0])
    axes[0][0].set_title('Noisy image')
    # COMPLETE
    myimshow(denoised, ax=axes[0][1])
    axes[0][1].set_title('Denoised image')
    
    axes[1][0].plot([exp.history[k]['loss'] for k in range(exp.epoch)],
    label="traininng loss")
    axes[1][1].plot([exp.history[k]['avg_psnr'] for k in range(exp.epoch)],
    label="traininng psnr")

    axes[1][0].legend()
    axes[1][1].legend()
    axes[1][0].set_xlabel("Epoch")
    axes[1][1].set_xlabel("Epoch")
    
    plt.tight_layout()
    fig.canvas.draw()
    
fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(7,6))
#exp2.run(num_epochs=200, plot=lambda exp: plot(exp, fig=fig, axes=axes,noisy=test_set[73][0]))    

plot(exp2, fig=fig, axes=axes,noisy=test_set[73][0]) # Added

#Part 12
#%matplotlib notebook

def plot_comparision(exp, fig, axes, test_image, visu_rate=2):
    noisy = test_image[0]
    clean = test_image[1]
    if exp.epoch % visu_rate != 0:
        return
    with torch.no_grad():
        denoised = exp.net(noisy[np.newaxis].to(exp.net.device))[0]
    axes[0].clear()
    axes[1].clear()
    axes[2].clear()
    

    myimshow(noisy, ax=axes[0])
    axes[0].set_title('Noisy image')
    # COMPLETE
    myimshow(denoised, ax=axes[1])
    axes[1].set_title('Denoised image')
    myimshow(clean, ax=axes[2])
    axes[2].set_title('Clean image')
    
    
    
    plt.tight_layout()
    fig.canvas.draw()
    
fig, axes = plt.subplots(ncols=3, nrows=1, sharex='all', sharey='all',figsize=(7,6))

plot_comparision(exp2, fig=fig, axes=axes,test_image=test_set[34]) # Added

#Part 15
import math
class UDnCNN(NNRegressor):
    def __init__(self, D, C=64):
        super(UDnCNN, self).__init__()
        self.D = D
        self.conv = nn.ModuleList()
        self.conv.append(nn.Conv2d(3, C, 3, padding=1))
        # COMPLETE
        nn.init.kaiming_normal_(self.conv[0].weight.data)#Part 7
        
            
        self.bn = nn.ModuleList()

        
        for k in range(D):
            self.bn.append(nn.BatchNorm2d(C, C))
            self.conv.append(nn.Conv2d(64, C, 3, padding=1))
       
            nn.init.constant_(self.bn[k].weight.data, 1.25 * np.sqrt(C))#Part 7
            nn.init.kaiming_normal_(self.conv[k+1].weight.data)#Part 7

            
        self.conv.append(nn.Conv2d(C, 3, 3, padding=1))
        nn.init.kaiming_normal_(self.conv[D+1].weight.data)#Part 7
          
    def forward(self, x):
        D = self.D
        h = F.relu(self.conv[0](x))
        print('debug1 :', hh)
        saved_h = []
        saved_h.append(h)

        for k in range(D):
            t = self.bn[k](self.conv[k+1](h))
            h = F.relu(t)
            
            if k < D/2 -1:
                h = F.max_pool2d(h, kernel_size=2, return_indices=True)
                saved_h.append(h)
            if k >= (D/2):
                if k >= (D/2)+1:
                    temp = saved_h.pop()
                    h = F.max_unpool2d(h, kernel_size=2, return_indices=True) + temp/math.sqrt(2)
            
            
            
            
        print('shit')
        y = self.conv[D+1](h) + x
        return y
    
#Part 16

D = 6
lr = 1e-3
net2 = UDnCNN(D)
net2 = net.to(device)
adam = torch.optim.Adam(net.parameters(), lr=lr)
stats_manager = DenoisingStatsManager()
val_set = test_set
exp99 = nt.Experiment(net2, train_set, val_set, adam, stats_manager,
                    output_dir="denoising40", batch_size=4, perform_validation_during_training=False)
fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(7,6))
#exp10.run(num_epochs=200, plot=lambda exp: plot(exp, fig=fig, axes=axes,noisy=test_set[73][0]))  
exp99.run(num_epochs=11, plot=lambda exp: plot(exp, fig=fig, axes=axes,noisy=test_set[73][0]))  
