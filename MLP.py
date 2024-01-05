# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 10:31:11 2021

@author: lxw
"""
import torch
import torch.utils.data
from torch import optim, nn
from torchvision import datasets
# import torch.utils.data as Data
from torchvision.transforms import transforms
import torchvision
import torch.nn.functional as F
from Nets import MLP,RNN
import scipy.io as scio
import h5py
import numpy as np
from MyUtils import FeatureDataset

batch_size = 200
learning_rate = 0.01
epochs = 20
model_paras_path = "./Model/model.pth"
feature_path = "./Feature/feature.hdf5"
train_loader = torch.utils.data.DataLoader(
     datasets.MNIST('mnistdata',train=True,download=True, transform=transforms.Compose(
         [transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))
          ])),
     batch_size = batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('mnistdata', train=True, download=False,transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))
         ])),
    batch_size=batch_size, shuffle=False)

device = torch.device('cuda:0')
net = MLP().to(device)
rnn = RNN().to(device)
# net.cuda()
optimizer = optim.Adam(rnn.parameters(),lr = learning_rate)

criterion = nn.CrossEntropyLoss().to(device)
loss_func = nn.CrossEntropyLoss().to(device)



 
train_data=torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=False,
)
BATCH_SIZE=200
train_loader=torch.utils.data.DataLoader(dataset=train_data,batch_size=BATCH_SIZE,shuffle=True)
test_data=torchvision.datasets.MNIST(root='./mnist/',train=False)
test_x=torch.unsqueeze(test_data.data,dim=1).type(torch.FloatTensor)[:2000]/255 #shape from(2000,28,28)to(2000,1,28,28)
test_y=test_data.targets[:2000]


class FeatureObject:
    def __init__(self, id, label, feature):
        self.id = id
        self.label = label
        self.feature = feature

def train(model,train_loader,epochs):
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            model.train()
            data = data.view(-1,28*28)
            data, target = data.to(device), target.to(device)
        
            logits,_ = model(data)
        
            loss = criterion(logits, target)
            # print("backward")
            loss.backward()
            optimizer.step()
            if batch_size % 1000== 0:
                print('Train Epoch : {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx*len(data), len(train_loader.dataset),
                    100.*batch_idx/len(train_loader), loss.item()
                    ))
    
        test_loss = 0
        correct = 0
    
        for batch_idx, (data, target) in enumerate(test_loader):
            model.eval()
            data = data.view(-1, 28*28)
            data, target = data.to(device), target.to(device)
            logits,_ = model(data)
            test_loss += criterion(logits, target).item() 
            pred = logits.data.max(1)[1]
            correct += pred.eq(target.data).sum()

        test_loss /= len(test_loader.dataset)
        print('\nTest set : Averge loss: {:.4f}, Accurancy: {}/{}({:.3f}%)'.format(test_loss, correct, len(test_loader.dataset),
            100.*correct/len(test_loader.dataset)
            ))        
        torch.save(model.state_dict(),model_paras_path)

def train_rnn(model,train_loader,epochs):
    for epoch in range(epochs):
        for step,(b_x,b_y) in enumerate(train_loader):
            # print(b_x.size(),b_y.size())
            b_x, b_y=b_x.view(-1,28,28).to(device), b_y.to(device)  # reshape x to (batch,time_step,input_size)
            # print(b_x.size())
            output=rnn(b_x)
            
            # total_loss = 0
            # for i in range(28):
            #     # print(output.size())
            #     # print(outputs[:,i,:].size())
            #     # l = loss_func(outputs[:,i,:],b_y)
            #     # print(l)
            #     total_loss+=loss_func(outputs[:,i,:],b_y)
            loss=loss_func(output,b_y)
            # loss = total_loss/28.0
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
               
            # if step == 0:
            #     break;
            
            if step%50==0:
                # break;
                x = test_x.view(-1,28,28).to(device)
                # test_y = test_y.to(device)
                test_out = rnn(x)
                test_out=test_out.cpu()
                pred_y = torch.max(test_out, 1)[1].data.squeeze()  # torch.max(input,dim)返回每一行中的最大值的标签
                accuracy = (pred_y == test_y).numpy().sum() / test_y.size(0)
                print('step: {} | train loss: {} | test accuracy: {} '.format(step, loss.data, accuracy))

        # test_loss = 0
        # correct = 0
        # for batch_idx, (data, target) in enumerate(test_loader):
        #     model.eval()
        #     data = data.view(-1, 28,28)
        #     data, target = data.to(device), target.to(device)
        #     logits = model(data)
        #     test_loss += criterion(logits, target).item() 
        #     pred = logits.data.max(1)[1]
        #     correct += pred.eq(target.data).sum()

        # test_loss /= len(test_loader.dataset)
        # print('\nTest set : Averge loss: {:.4f}, Accurancy: {}/{}({:.3f}%)'.format(test_loss, correct, len(test_loader.dataset),
        #     100.*correct/len(test_loader.dataset)
        #     ))     

def retrain():
    print("retrain:")
    

def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    # f = h5py.File(feature_path,"w")
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data = data.view(-1, 28*28)
            print(data.shape,data,target.shape,target)
            
            x, label = data.cuda(), target.cuda()
            break
            # optimizer.zero_grad()
            logits,feature2_map = model(x)
            # feature2_map = feature2_map.cpu()
            # print(batch_idx)
            # f.create_dataset(str(batch_idx), data = feature2_map)                
           
            # print(logits.size())
            print("logits=",logits)
            print("label=",label)
            test_loss += criterion(logits, label).item()
            clss = F.softmax(logits,dim =1)
            print(clss)
            print("test_loss=",test_loss)
            print("max = ", logits.max(1, keepdim=True))
            pred = logits.max(1, keepdim=True)[1]
            # print("pred=", pred)
            break;
            correct += pred.eq(label.view_as(pred)).sum().item()
        test_loss /= (batch_idx+1)
        # print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
    # f.close()
    

def save_feature(model, test_loader):
    model.eval()
#    test_loss = 0
#    correct = 0
    f = h5py.File(feature_path,'w')
    group1 = f.create_group("/data")
    group2 = f.create_group("/label")
    group3 = f.create_group("/idx")
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data = data.view(-1, 28*28)
            x, label = data.cuda(), target.cuda()
            logits,feature2_map = model(x)
            feature2_map = np.array(feature2_map.cpu())
            label = np.array(label.cpu())
            print(batch_idx,label, batch_idx)
            group1.create_dataset(str(batch_idx), data = feature2_map)   
            group2.create_dataset(str(batch_idx), data = label)   
            group3.create_dataset(str(batch_idx), data = batch_idx)
#            print(logits.size())

    f.close()
    
def load_feature(feature_path):
#    f = h5py.File(feature_path,'r')
    # data_num = 0
    # label_num = 0
    
    # for group in f.keys():
    #     print(group)
    #     grp = f[group]
        # for sub_group in grp.keys():
            # if group == "data":
            #     data_num = data_num + 1
            # if group == "label":
            #     label_num = label_num + 1
            # dataset = f[group+'/'+sub_group]
            # dataset = grp[sub_group]
            # print("sub_group=",sub_group,"name=",dataset.name, "shape=",dataset.shape)
            # print("value=", dataset[:])
            # idx += 1
            # if idx == 100:
                
            #     print("100")
            
    # idice = 0
    # grp = f["data"]
    # print(len(grp))
    
    # data = grp["1"]
    # print(data.shape,data.name,data[:])
    # group_data = "data/"
    # group_label = "label/"
    # # out_arr =[]# np.empty(shape=(1,250))
    # for i in range (0,60000):
    #     ds = f[group_data+'/'+str(i)]
    #     data = ds[:]
    #     print(i)
    #     # print("data shape = ",data.shape)
    #     if i == 0:
    #         print(data.shape)
    #         out_arr = data    
    #         print("out_arr appended shape = " ,out_arr.shape)
    #     else:
    #         out_arr = np.append(out_arr,data,axis = 0)
    #         print("out_arr appended shape = " ,out_arr.shape)
    #     i +=1
    #     if i==6:
    #         break

    
    
    # print(data_num, label_num)
    print(feature_path)
    dataset =  FeatureDataset("./Feature","feature.hdf5",False)
    # print(dataset[0])
    dataloader = torch.utils.data.DataLoader( dataset,  batch_size = 15, shuffle=False,collate_fn = collate_fn)
    for i_batch,(img, label, idx) in enumerate(dataloader):
        print(i_batch)
        print(img.shape,label.shape, idx.shape)
        # break
    
def collate_fn(batch):
    img, label,idx = zip(*batch)
#    img, label  = zip(*batch)
    imgs  = img[0]
    labels = label[0]
    idxs = idx[0]
#    idxs = idx[0]
    length = len(img)
    for i in range(1,length):
        imgs = np.append(imgs,img[i],axis = 0)
        labels = np.append(labels,label[i],axis = 0)
    return torch.tensor(imgs), torch.tensor(labels) , torch.tensor(idxs)
        
def main():
#    train(net,train_loader,epochs)
    print(rnn)
    train_rnn(rnn,train_loader,5)
    # checkpoint  = torch.load(model_paras_path)
    # net.load_state_dict(checkpoint)
    # test(net, test_loader)    
#    save_feature(net,test_loader)
    
#    load_feature(feature_path)    
    print("stop")    
        
if __name__ == '__main__':
    main()      
        
        
        
        
    
    
    
    