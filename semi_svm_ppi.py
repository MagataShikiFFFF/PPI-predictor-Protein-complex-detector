import numpy as np
import time
import h5py
import pandas as pd

from data_seq import data_seq, protein_pair, protein_pair0, n_dgram, fea

from chainer import cuda, optimizers, Variable, function_hooks, backend, serializers
from vgg16 import vgg_16, Linear
# sequences feature etraction by n-gram 

lunion = ['A','C','E','D','G','F','I','H','K','M','L','N','Q','P','S','R','T','W','V','Y']

def ppi_data_output(catalogue, lunion):
    ppi_ = data_seq(catalogue)
    pp0, pp1 = protein_pair(ppi_, lunion)
    UN = n_dgram(lunion)
    pro0 = np.array(pd.DataFrame(fea(pp0, UN), columns=UN)).astype(np.float32)
    pro1 = np.array(pd.DataFrame(fea(pp1, UN), columns=UN)).astype(np.float32)

    ppi_data = 0.5*(pro0 + pro1)/pro0.shape[1]

    return ppi_data
DIP_ppi = ppi_data_output('/nfshome/yuanxin/manjusaka/n-gram_data/DIP_dataset', lunion)

Ne_ppi = ppi_data_output('/nfshome/yuanxin/manjusaka/n-gram_data/Negatome', lunion)

#2-Norm
DIP_ppi = np.sqrt(np.square(vgg16(reshapei(DIP_ppi0)).data) + np.square(vgg16(reshapei(DIP_ppi1)).data))
Ne_ppi = np.sqrt(np.square(vgg16(reshapei(Ne_ppi0)).data) + np.square(vgg16(reshapei(Ne_ppi1)).data))

# Add label, put postive data and negative data together and shuffle
pdatat = np.c_[DIP_ppi.data, np.ones([DIP_ppi.shape[0], 1])]
ndatat = np.c_[Ne_ppi.data, np.zeros([Ne_ppi.shape[0], 1])]

data_set = np.r_[ndatat, pdatat]

np.random.shuffle(data_set)


x_train = data_set[0:7000, 0:-1]

x_test = data_set[7000:12800, 0:-1]

y_train = data_set[0:7000, -1].astype(np.int32)

y_test = data_set[7000:12800, -1].astype(np.int32)

# Standardize features
from sklearn.preprocessing import StandardScaler as SS

Scalar = SS()
Scalar.fit(x_train)

x_train_transformed = Scalar.transform(x_train).astype(np.float32)
x_test_transformed = Scalar.transform(x_test).astype(np.float32)

#Extractor feature by the transfer DCNN feature extractor
h = vgg16(reshapei(x_train_transformed))

#Build the SVM and train with labeled samples 
from sklearn.svm import SVC

clf_rbf = SVC(kernel='rbf', probability=True)
clf_rbf.fit(x_train_transformed, y_train)


#Train the semi-SVM with label and unlabeled samples
class semi_svm_trainer(object):
    """docstring for semi_svm_trainer"""
    def __init__(self, train_data, train_label, semi_data, svm, batchsize):
        super(semi_svm_trainer, self).__init__()
        self.batchsize = batchsize
        self.train_data = train_data
        self.train_label = train_label
        self.semi_data = semi_data
        self.svm = svm

    def run(self, test_data=None, test_label=None):

        result = []
        indexes1 = np.random.permutation(self.semi_data.shape[0])
        for i in range(0, self.semi_data.shape[0] , self.batchsize):
            start = time.time()

            x_batch = np.r_[self.train_data, self.semi_data[indexes1[i : i + self.batchsize]]]
            y_semi = self.svm.predict(self.semi_data[indexes1[i : i + self.batchsize]])
            y_batch = np.r_[self.train_label, y_semi]

            self.svm.fit(x_batch, y_batch)
            
            semi_score = self.svm.score(x_batch, y_batch)
            
            print ('time', i)
            print (str(time.time()-start))

            #print ('semi_score:' + str(semi_score))
            

            if test_data is not None:
                test_score = self.svm.score(test_data, test_label)
                #print ('test_score:' + str(test_score))
            
            result.append([semi_score, test_score])

        print (result)

svm_trainer = semi_svm_trainer(train_data=x_train, train_label=y_train, semi_data=x_semi, svm=clf_rbf, batchsize=1000) 
svm_trainer.run(test_data=x_test, test_label=y_test)




