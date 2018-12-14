import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
from torchvision import transforms
from sklearn.base import BaseEstimator
from sklearn.preprocessing import normalize
from PIL import Image
from os.path import isfile

def requires_grad(p):
    return p.requires_grad


class SimpleConvModel(nn.Module):
    def __init__(self, out_channel_conv=16, linear_hidden_size=256, out_class=13):
        super(SimpleConvModel, self).__init__()

        self.out_channel_conv = out_channel_conv
        self.linear_hidden_size = linear_hidden_size
        self.out_class = out_class

        self.seq1 = nn.Sequential(nn.Conv2d(3, self.out_channel_conv, (5,5), stride=(3,3)),
                                  nn.MaxPool2d((5,5)),
                                  nn.ReLU())

        self.seq2 = nn.Sequential(nn.Linear(16 * 8 * 8, self.out_class),#self.linear_hidden_size),
                                  #nn.ReLU(),
                                  #nn.Linear(self.linear_hidden_size, self.out_class),
                                  nn.Softmax(dim = 1))

    def forward(self, x):
        out1 = self.seq1(x)
        out2 = out1.view(-1, 16*8*8)
        out3 = self.seq2(out2)
        return out3

class BasicCNN(BaseEstimator):
    def __init__(self, learning_rate=1e-3, nb_epoch = 10, batch_size = 32, verbose=False, use_cuda=False):
        super(BasicCNN, self).__init__()

        if learning_rate is None:
            learning_rate = 1e-3
        if nb_epoch is None:
            nb_epoch = 10
        if batch_size is None:
            batch_size = 32
        if verbose is None:
            verbose = False
        if use_cuda is None:
            use_cuda = False

        self.nb_epoch = nb_epoch
        self.batch_size = batch_size
        self.verbose = verbose
        self.use_cuda = use_cuda

        self.model_conv = SimpleConvModel()

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        if self.use_cuda:
            self.model_conv.cuda()
            self.criterion.cuda()

        # Optimizer
        self.optim = optim.Adagrad(self.model_conv.parameters(), lr=1e-3, weight_decay=0.05)

    def fit(self, X, Y):
        '''
            param X: numpy.ndarray
                shape = (num_sample, C * W * H)
                with C = 3, W = H = 128
            param Y: numpy.ndarray
                shape = (num_sample, 1)
        '''
        X = self.process_data(X)

        Y = self.process_label(Y)

        self.model_conv.train()

        nb_batch = int(X.shape[0] / self.batch_size)

        for e in range(self.nb_epoch):
            sum_loss = 0
            for i in range(nb_batch):
                self.optim.zero_grad()
                beg = i * self.batch_size
                end = min(X.shape[0], (i + 1) * self.batch_size)

                x = X[beg:end]
                y = Y[beg:end]
                if self.use_cuda:
                    x, y = x.cuda(), y.cuda()
                out = self.model_conv(x)
                loss = self.criterion(out, y)

                loss.backward()
                self.optim.step()

                sum_loss += loss.item()
            sum_loss /= nb_batch
            if self.verbose:
                print("Epoch %d : loss = %f" % (e, sum_loss))

    def process_data(self, X):
        n_sample = X.shape[0]
        mean = np.mean(X, axis=1)[:, np.newaxis]
        std = np.std(X, axis=1)[:, np.newaxis]
        X = (X - mean) / std
        X = X.reshape(n_sample, 3, 128, 128)
        X = X.astype(np.float)# / 255.
        #print(X[0])
        return torch.Tensor(X)

    def process_label(self, y):
        res = torch.zeros(1)
        for i in range(y.shape[0]):
            l = torch.Tensor([y[i,0]])
            res = torch.cat((res, l))
        return res[1:].type(torch.long)

    def predict(self, X):
        '''
            param X: numpy.ndarray
                shape = (num_sample, C * W * H)
                with C = 3, W = H = 128
            return: numpy.ndarray
                of int with shape (num_sample) ?
                of float with shape (num_sample, num_class) ?
                of string with shape (num_sample) ?
        '''
        # inverted_dico = {v:k for k,v in self.label_dico.items()}
        self.model_conv.eval()
        X = self.process_data(X)

        if self.use_cuda:
            X = X.cuda()
        pred = self.model_conv(X).argmax(dim=1).cpu().numpy()
        return pred

    def save(self, path="./"):
        pickle.dump(self, open(path + '_model.pickle', "wb"))

    def load(self, path="./"):
        modelfile = path + '_model.pickle'
        if isfile(modelfile):
            with open(modelfile, 'rb') as f:
                self = pickle.load(f)
            print("Model reloaded from: " + modelfile)
        return self
