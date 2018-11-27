import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import numpy as np
from torchvision import transforms
from sklearn.base import BaseEstimator
from PIL import Image
from matplotlib import cm

class AlexNetModel(BaseEstimator):
    def __init__(self, learning_rate=1e-3, nb_epoch=10, verbose=False, batch_size=32, n_class=13, use_cuda=False):
        super(AlexNetModel, self).__init__()
        self.nb_epoch = nb_epoch
        self.batch_size = batch_size
        
        self.model_conv = torchvision.models.alexnet(pretrained=True)
        
        num_ftrs = self.model_conv.classifier[6].in_features
        # convert all the layers to list and remove the last one
        features = list(self.model_conv.classifier.children())[:-1]
        ## Add the last layer based on the num of classes in our dataset
        features.extend([nn.Linear(num_ftrs, n_class)])
        ## convert it into container and add it to our model class.
        self.model_conv.classifier = nn.Sequential(*features)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optim = optim.SGD(list(filter(lambda p: p.requires_grad, self.model_conv.parameters())), lr=learning_rate, momentum=0.9)
        
        self.use_cuda = use_cuda
        
        if self.use_cuda:
            self.model_conv.cuda()
            self.criterion.cuda()
            
        # Image transformation
        input_shape = 224
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        scale = 360
        input_shape = 224
        use_parallel = True

        self.data_transforms = transforms.Compose([
            transforms.Resize(scale),
            transforms.RandomResizedCrop(input_shape),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=90),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.Lambda(lambda t: t.cuda() if self.use_cuda else t)])

    def fit(self, X, y):
        '''
            param X: numpy.ndarray
                shape = (num_sample, C * W * H)
                with C = 3, W = H = 256
            param Y: numpy.ndarray
                shape = (num_sample, 1)
        '''
        X = self.process_data(X)
        n_sample = X.size(0)
        nb_batch = int(n_sample / self.batch_size)
        
        y = self.process_label(y)
        
        self.model_conv.train()
        for e in range(self.nb_epoch):
            sum_loss = 0
            for i in range(nb_batch):
                self.model_conv.zero_grad()
                
                start = i * self.batch_size
                end = max((i + 1) * self.batch_size, n_sample)
             
                out = self.model_conv(X[start:end])
                loss = self.criterion(out, y[start:end])
                
                loss.backward()
                self.optim.step()
                
                sum_loss += loss.item()
            print("Epoch %d : loss = %f" % (e, sum_loss / nb_batch))
            
        
    def process_data(self, X):
        n_sample = X.shape[0]
        X = X.reshape(n_sample, 3, 256, 256)
        res = torch.zeros(1,3,224,224)
        if self.use_cuda:
            res = res.cuda()
        for i in range(n_sample):
            x = np.moveaxis(X[i], 0, -1)
            img = Image.fromarray((x*255).astype('uint8'))
            t = self.data_transforms(img).unsqueeze(0)
            res = torch.cat((res,t))
        return res[1:]
    
    def process_label(self, y):
        self.label_dico = {}
        res = torch.zeros(1)
        if self.use_cuda:
            res = res.cuda()
        for i in range(y.shape[0]):
            l = y[i,0]
            if l not in self.label_dico:
                self.label_dico[l] = len(self.label_dico)
            l = torch.Tensor([self.label_dico[l]])
            if self.use_cuda:
                l = l.cuda()
            res = torch.cat((res, l))
        return res[1:].type(torch.long) 
        
    def predict(self, X):
        '''
            param X: numpy.ndarray
                shape = (num_sample, C * W * H)
                with C = 3, W = H = 256
            return: numpy.ndarray
                of int with shape (num_sample) ?
                of float with shape (num_sample, num_class) ?
                of string with shape (num_sample) ?
        '''
        inverted_dico = {v:k for k,v in self.label_dico.items()}
        self.model_conv.eval()
        X = self.process_data(X)
        pred = self.model_conv(X).argmax(dim=1).detach().cpu()
        res = []
        for i in range(pred.size(0)):
            res.append(inverted_dico[pred[i].item()])
        return np.asarray(res)

    def save(self, path="./"):
        pickle.dump(self, open(path + '_model.pickle', "wb"))

    def load(self, path="./"):
        modelfile = path + '_model.pickle'
        if isfile(modelfile):
            with open(modelfile, 'rb') as f:
                self = pickle.load(f)
            print("Model reloaded from: " + modelfile)
        return self
