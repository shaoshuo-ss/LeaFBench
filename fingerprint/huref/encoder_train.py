import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import random
def seed_everything(seed):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"Global seed set to {seed}")

# Define encoder network
class CNNEncoder(nn.Module):
    def __init__(self,k=48):
        super(CNNEncoder, self).__init__()
        #kernel=5,H2=(H1-K+2Pad+Stride)/Stride
        k=k
        s=4
        p=(k-s)//2
        hiden_size =512
        self.conv1 = nn.Conv2d(6, 8,kernel_size=k,stride=s,padding=p)#256*256
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 64,kernel_size=k,stride=s,padding=p)#64*64
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 256,kernel_size=k,stride=s,padding=p)#16*16
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, hiden_size,kernel_size=k,stride=s,padding=p)#4*4
        self.bn4 = nn.BatchNorm2d(hiden_size)
        self.relu=nn.ReLU()
    def forward(self, x):
        x = self.bn1(self.relu(self.conv1(x.squeeze(0))))
        x = self.bn2(self.relu(self.conv2(x)))
        x = self.bn3(self.relu(self.conv3(x)))
        x = self.bn4(self.conv4(x))
        x = torch.mean(x.view(x.shape[0],x.shape[1],-1),dim=-1)
        return x.view(-1,x.shape[1])

#Discriminator
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.fc(x)
# Define Contrastive Loss function
class ContrastiveLoss(nn.Module):
    def __init__(self, positive_rate=1.3):
        super(ContrastiveLoss, self).__init__()
        self.positive_rate = positive_rate
        self.cossim = torch.cosine_similarity
    def forward(self, anchor, positive, negative):
        loss_contrastive=torch.mean(self.positive_rate*(1-self.cossim(anchor, positive))+(2-self.positive_rate)*torch.abs(self.cossim(anchor, negative)))
        return loss_contrastive

def stand_normalize2d(input_matrix):
    normalized_matrix = torch.empty_like(input_matrix)
    for i in range(input_matrix.shape[0]):
        mean = torch.mean(input_matrix[i])
        std = torch.std(input_matrix[i])
        normalized_matrix[i]=(input_matrix[i]-mean)/std
    return normalized_matrix

def stand_normalize(input_matrix):
    for i in range(input_matrix.shape[0]):
        for j in range(input_matrix.shape[1]):
            mean = torch.mean(input_matrix[i][j])
            std = torch.std(input_matrix[i][j])
            input_matrix[i][j]=(input_matrix[i][j]-mean)/std
    return input_matrix

class GenDataset(torch.utils.data.Dataset):
    def __init__(self,batch_size,batch_len,noise):
        super(GenDataset, self).__init__() 
        self.sizes = (batch_size,6,4096,4096)
        self.len=batch_len
        self.noise=noise
        self.i=0  
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        q=torch.randn(self.sizes)
        k=torch.randn(self.sizes)
        x=torch.randn(self.sizes)
        y=torch.randn(self.sizes)
        anchor_sample = x.matmul(q).matmul(k).matmul(y)
        qp=q+0.1*torch.randn(self.sizes)
        kp=k+0.1*torch.randn(self.sizes)
        xp=x+self.noise*torch.randn(self.sizes)
        yp=xp.permute(0, 1, 3, 2)
        positive_sample = xp.matmul(qp).matmul(kp).matmul(yp)
        qn=torch.randn(self.sizes)
        kn=torch.randn(self.sizes)
        xn=torch.randn(self.sizes)
        yn=xn.permute(0, 1, 3, 2)
        negative_sample = xn.matmul(qn).matmul(kn).matmul(yn)
        return stand_normalize(anchor_sample), stand_normalize(positive_sample), stand_normalize(negative_sample)

class DisDataset(torch.utils.data.Dataset):
    def __init__(self,batch_size,batch_len):
        super(DisDataset, self).__init__() 
        self.sizes = (batch_size,6,4096,4096)
        self.len=batch_len
        self.i=0  
    def __len__(self):
        return self.len
 
    def __getitem__(self, idx):
        q=torch.randn(self.sizes)
        k=torch.randn(self.sizes)
        x=torch.randn(self.sizes)
        y=x.permute(0, 1, 3, 2)
        anchor_sample = x.matmul(q).matmul(k).matmul(y)
        return stand_normalize(anchor_sample)

def parse_args():
    parser = argparse.ArgumentParser(description='Encoder Training Script')
    parser.add_argument('--batchsize', type=int, default=10, help='Batch size for one forward pass')
    parser.add_argument('--steps_per_epoch', type=int, default=10, help='Number of steps per epoch')
    parser.add_argument('--outputpath', type=str, default='/encoders/', help='Path to save trained encoder')
    parser.add_argument('--learningrate', type=float, default=0.0001, help='Learning rate for optimization')
    parser.add_argument('--noise', type=float, default=0.4, help='Noise intensity')
    parser.add_argument('--k', type=int, default=48, help='kernel size')
    parser.add_argument('--positiverate', type=float, default=1.3, help='Scale factor for positive sample loss ')
    return parser.parse_args()   
       
if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    args = parse_args()
    batch_size = args.batchsize
    learning_rate = args.learningrate
    noise = args.noise
    k = args.k
    positive_rate = args.positiverate
    # Set random seed
    seed_everything(100)
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # Initialize encoder and optimizer
    hiden_size =512
    encoder = CNNEncoder(k).cuda()
    optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,eta_min=0.000005,T_max=1500)
    discriminator=Discriminator(hiden_size).cuda()
    optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate)
    scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_D,eta_min=0.000005,T_max=1500)

    # Initialize data loader and loss function
    dataset = GenDataset(batch_size=batch_size,batch_len=args.steps_per_epoch,noise=noise)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
    Disdataset = DisDataset(batch_size=batch_size,batch_len=args.steps_per_epoch)
    disdataloader = torch.utils.data.DataLoader(Disdataset, batch_size=1)
    criterion= ContrastiveLoss(positive_rate)
    criterion_D=nn.BCELoss()


    # Training loop
    for epoch in range(17):
        running_loss = 0.0
        encoder.eval()
        discriminator.train()
        # Train the encoder
        for i, anchor in enumerate(disdataloader):
            real_samples = torch.randn(batch_size, 512) 
            real_labels = torch.ones(batch_size, 1)
            fake_labels = torch.zeros(batch_size, 1)
            
            optimizer_D.zero_grad()
            real_outputs = discriminator(real_samples.cuda())
            loss_real = criterion_D(real_outputs, real_labels.cuda())

            anchor_embedding = encoder(anchor.cuda())
            fake_samples = stand_normalize2d(anchor_embedding)
            fake_outputs = discriminator(fake_samples)
            loss_fake = criterion_D(fake_outputs, fake_labels.cuda())
            loss_D = loss_real + loss_fake
            loss_D.backward()
            optimizer_D.step()
            scheduler_D.step()
            running_loss += loss_D.item()
            print('[%d, %5d] discriminator loss: %.3f' %
            ( epoch, i + 1, running_loss ))
            running_loss = 0.0
        discriminator.eval()
        encoder.train()
        # Train the discriminator
        for i, (anchor, positive, negative) in enumerate(dataloader):
            optimizer.zero_grad()
            real_labels = torch.ones(batch_size, 1)
            anchor_embedding = encoder(anchor.cuda())
            positive_embedding = encoder(positive.cuda())
            negative_embedding = encoder(negative.cuda())
            loss_C = criterion(anchor_embedding, positive_embedding, negative_embedding)
            
            fake_samples = stand_normalize2d(anchor_embedding)
            fake_outputs = discriminator(fake_samples)
            loss_D = criterion_D(fake_outputs, real_labels.cuda())
           
            loss = loss_C+loss_D/2
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()
            print('[%d, %5d] encoder loss: %.3f loss_c:%.3f' %
            ( epoch, i + 1, running_loss,loss_C.item() ))
            running_loss = 0.0
 
    # Save the trained encoder
    torch.save(encoder, args.outputpath+"encoder.pth")
    print('Finished training')