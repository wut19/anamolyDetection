from transformer_model import Model
from onehot_dataload import make_datasets,make_dataloader
import pickle
import os
import torch
from torch import optim
import torch.nn.functional as F
import torch.nn as nn


def train():
    n_vocab = 167657
    embedd = 200
    d_model = 200
    nhead = 4
    num_encoder = 2
    dropout = 0.1
    batch_size = 4
    lr = 0.002
    epochs = 80
    
    device = torch.device('cpu')
    
    train_set,eval_set,_ = make_datasets()
    sen_size = train_set.reviews.shape[1]
    model = Model(n_vocab=n_vocab, embedd=embedd, sen_size=sen_size,d_model=d_model,nhead=nhead,num_encoder=num_encoder,dropout=dropout,device=device)
    # model.load_state_dict(torch.load(os.path.join("models/transformer.pkl")))
    optimizer = optim.Adam(model.parameters(),lr=lr,betas=(0.5,0.999))
    BCE_loss = nn.BCELoss()
    
    os.makedirs('models', exist_ok=True)
    

    for epoch in range(epochs):
        model.train()
        
        train_loader = make_dataloader(train_set,batch_size,device)
        train_set.shuffle()
        
        if (epoch + 1) % 10 == 0:
            optimizer.param_groups[0]['lr'] /= 4
        for y, x in train_loader:
            # print(x.shape)
            model.zero_grad()
            y,x = (y>0).float(),x.long()
            y_p = model(x)
            loss = BCE_loss(y_p.squeeze(),y.squeeze())
            loss.backward()
            
            optimizer.step()
        
        model.eval()
        eval_loader = make_dataloader(eval_set,batch_size=batch_size,device=device)
        eval_set.shuffle()
        n_count = 0
        r_count = 0
        with torch.no_grad():
            for y, x in eval_loader:
                y,x = (y>0).float(),x.long()
                y_p = model(x)
                #print(em)
                #print(y_p.squeeze())
                y_p = torch.round(y_p)
                
                #print(y_p.squeeze(),y)
                r_count += torch.sum(y_p.squeeze()==y.squeeze())
                n_count += y.shape[0]
        #print(r_count,n_count)
        with open('accuracy.txt','a+') as f:
            f.write("epoch:%d accuracy:%f \n"%(epoch,r_count/n_count))
        torch.save(model.state_dict(),"models/transformer%d.pkl"%epoch)

if __name__=="__main__":
    train()
        
            