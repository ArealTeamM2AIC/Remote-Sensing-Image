import torch as th
import torch.nn as nn
import pickle
import numpy as np   # We recommend to use numpy arrays
from os.path import isfile
from sklearn.base import BaseEstimator
import sample_code_submission.utils as utils

class ConvModel(nn.Module):
    def __init__(self, in_channels, out_channels):
        '''
            Single convolution layer module
        '''
        super(ConvModel, self).__init__()
        self.out_channels = out_channels

        # convolution with square kernel of size 5
        # padding is equal to 2 beceause we need to produce
        # for each pixels on vector (with stride equals to 1)
        self.conv1 = nn.Conv2d(in_channels, self.out_channels, kernel_size=(5,5), padding=2)
        self.rel_conv1 = nn.ReLU()

    def forward(self, data):
        '''
            Forward function

            param data: troch.tensor of size (1, in_channels, W, H)
            return: torch.tensor of size (W * H, out_channels)
        '''
        # data.size() = (1, in_channels, w, h)
        out = self.conv1(data)
        out = self.rel_conv1(out)
        # out.size() = (1,self.out_channels, w, h)

        # squeeze(0) -> (self.out_channels, w, h)
        # permute(1, 2, 0) -> (w, h, self.out_channels)
        out = out.squeeze(0).permute(1, 2, 0)

        # view(-1, out_channels) -> (w * h, out_channels)
        return out.contiguous().view(-1, self.out_channels)

class LinModel(nn.Module):
    def __init__(self, out_channels):
        '''
            Single linear layer module
        '''
        super(LinModel, self).__init__()
        self.out_channels = out_channels

        # Output layer
        self.lin = nn.Linear(self.out_channels, 1)
        self.act = nn.Sigmoid()

    def forward(self, data):
        '''
            param data: torch.tensor of size (w * h, out_channels)
            return: torch.tensor of size (w * h, 1)
        '''
        out = self.lin(data)
        return self.act(out)


class model (BaseEstimator):
    def __init__(self, out_channels=5, lr=1e-5, nb_epoch=4, verbose=False, batch_size=1000):
        super(model, self).__init__()
        '''
        This constructor is supposed to initialize data members.
        Use triple quotes for function documentation.
        '''
        self.in_channel = 3
        self.out_channels = out_channels
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        self.verbose = verbose

        self.conv_model = ConvModel(3, self.out_channels)
        self.lin_model = LinModel(self.out_channels)

        self.loss_fn = nn.BCELoss()
        self.optim = th.optim.Adagrad(list(self.conv_model.parameters()) + list(self.lin_model.parameters()), lr=lr)

        self.is_trained=False

    def fit(self, X, y):
        '''
        param X : numpy.ndarray
            numpy.ndarray.shape = (N, 1, C, H, W)
            C : channels
            H : height
            W : width
            N : number of image
        param y : numpy.ndarray
            numpy.ndarray.shape = (N * W * H)
            N : number of sample
            H : height
            W : width
        '''
        if len(X.shape) != 5:
            exit("X.shape = (N, 1, C, H, W), len(X.shape) != 5 !")
        if X.shape[1] != 1:
            exit("X.shape = (N, 1, C, H, W), X.shape[1] must be 1 !")
        if X.shape[2] != 3:
            exit("X.shape = (N, 1, C, H, W), X.shape[2] must be 3 !")

        # pass both models in train mode
        self.conv_model.train()
        self.lin_model.train()

        # reshape for loop over samples
        y = y.reshape(X.shape[0], X.shape[-2] * X.shape[-1])

        # loop over epochs
        for i in range(self.nb_epoch):
            # loss sum for one epoch
            loss_sum = 0

            # iter over images and ground truths
            for j, (img, gt) in enumerate(zip(X, y)):

                # pass image to CNN
                out = self.conv_model(utils.to_float_tensor(img))

                # split the flatten image and ground truth in mini batchs
                splitted_out = th.split(out, self.batch_size, dim=0)
                splitted_gt = th.split(utils.to_float_tensor(gt), self.batch_size, dim=0)

                # iter over mini batchs
                for o, g in zip(list(splitted_out), list(splitted_gt)):
                    self.optim.zero_grad()

                    # pass mini batch to linear model
                    out_batch = self.lin_model(o)
                    # compute loss value
                    loss = self.loss_fn(out_batch, g.view(-1,1))

                    # backward over the graph model
                    loss.backward(retain_graph=True)
                    # update weights
                    self.optim.step()

                    # add loss value
                    loss_sum += loss.item()

                if self.verbose:
                    print("Epoch %d, image %d" % (i, j))

            # divide loss sum by the number of image
            loss_mean = loss_sum / X.shape[0]
            if self.verbose:
                print("[Epoch %d] loss = %f" % (i, loss_mean))

        self.is_trained = True

    def predict(self, X):
        '''
        param X : numpy.ndarray
            numpy.ndarray.shape = (N, 1, C, H, W)
            C : channels
            H : height
            W : width
            N : number of image
        return : numpy.ndarray
            numpy.ndarray.shape = (N * W * H)
        '''
        if len(X.shape) != 5:
            exit("X.shape = (N, 1, C, H, W), len(X.shape) != 5 !")
        if X.shape[1] != 1:
            exit("X.shape = (N, 1, C, H, W), X.shape[1] must be 1 !")
        if X.shape[2] != 3:
            exit("X.shape = (N, 1, C, H, W), X.shape[2] must be 3 !")

        # switch models to eval mode
        self.conv_model.eval()
        self.lin_model.eval()

        # create empty ndarray for results
        res = np.zeros((X.shape[0]* X.shape[3] * X.shape[4]))

        # iter over images
        for j, img in enumerate(X):
            # CNN
            out_conv = self.conv_model(utils.to_float_tensor(img))
            # Linear
            pred = self.lin_model(out_conv)
            # add results for the j-th image
            res[j * X.shape[3] * X.shape[4]:(j + 1) * X.shape[3] * X.shape[4]] = pred.detach().numpy().reshape(-1)

        return res

    def save(self, path="./"):
        pickle.dump(self, open(path + '_model.pickle', "wb"))

    def load(self, path="./"):
        modelfile = path + '_model.pickle'
        if isfile(modelfile):
            with open(modelfile, 'rb') as f:
                self = pickle.load(f)
            print("Model reloaded from: " + modelfile)
        return self
