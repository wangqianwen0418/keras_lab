import pickle
import os
import numpy as np
def unpickle(file, label_key='labels'):
    import pickle
    fo = open(file, 'rb')
    d = pickle.load(fo, encoding='bytes')
    d_decode={}
    for k,v in d.items():
        d_decode[k.decode('utf8')]=v
    fo.close()
    data = d_decode['data']
    labels =d_decode[label_key]
    data = data.reshape(data.shape[0], 3, 32, 32)
    return data,labels

def readCifra10():
    data_path = 'cifar-10-batches-py'
    num_train_samples = 60000
    x_train = np.zeros((num_train_samples, 3, 32, 32), dtype='uint8')
    y_train = np.zeros((num_train_samples,), dtype='uint8')
    for i in range(1, 6):
            fpath = os.path.join(data_path, 'data_batch_' + str(i))
            imgs, labels = unpickle(fpath)
            x_train[(i - 1) * 10000: i * 10000, :, :, :] = imgs
            y_train[(i - 1) * 10000: i * 10000] = labels
    
    tpath = os.path.join(data_path, 'test_batch')
    x_test, y_test = unpickle(tpath)

    return (x_train,y_train),(x_test,y_test)

def readCifra100(key='fine'):
    data_path = 'cifar-100-python'
    fpath = os.path.join(data_path, 'train')
    x_train,y_train = unpickle(fpath, '{}_labels'.format(key))
    fpath = os.path.join(data_path, 'test')
    x_test,y_test = unpickle(fpath, '{}_labels'.format(key))
    return (x_train,y_train),(x_test,y_test)

readCifra10()
readCifra100()
