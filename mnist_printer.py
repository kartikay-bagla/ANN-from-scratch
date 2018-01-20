from mnist_loader import *
import numpy as np #using numpy for arrays and optimized matrix multiplication
np.random.seed(1) #seeding so that repeated results are the same and we can observe
#changes from editing.
np.seterr(over = 'ignore')
np.set_printoptions(precision=0, suppress=True)
import pickle
import os.path
from matplotlib import pyplot as plt


if os.path.isfile('X_test.dat'):
    X_train = []
    with open('X_test.dat', 'rb') as f:
        X_test = pickle.load(f)
        y_test = pickle.load(f)
        y_train = pickle.load(f)
        while True:
            try:
                X_train+=pickle.load(f)
            except EOFError:
                break
    
    #print(len(X_train))

else:
    print("Get testset")
    testing = get_labeled_data('t10k-images-idx3-ubyte.gz',
                               't10k-labels-idx1-ubyte.gz')
    print("Got %i testing datasets." % len(testing[1]))
    X_test = testing[0]
    X_test=X_test.tolist()
    for i in range(len(X_test)):
        flat_list=[]
        for sublist in X_test[i]:
            for item in sublist:
                flat_list.append(item)
        X_test[i] = flat_list

    f=open('X_test.dat', 'wb')
    pickle.dump(file=f, obj=X_test, protocol = 4)
    
    y_test = testing[1].tolist()
    y_test=[i for i in y_test]
    #print (y_test, len(y_test))
    for i in range(len(y_test)):
        
        a=[0,0,0,0,0,0,0,0,0,0]
        a[y_test[i][0]]=1
        y_test[i]=a 

    
    pickle.dump(file=f, obj=y_test, protocol = 4)

    print("Get trainingset")
    training = get_labeled_data('train-images-idx3-ubyte.gz',
                                'train-labels-idx1-ubyte.gz')
    print("Got %i training datasets." % len(training[1]))
    
    y_train = training[1].tolist()
    y_train=[i for i in y_train]
    #print (y_train, len(y_train))
    for i in range(len(y_train)):
        
        a=[0,0,0,0,0,0,0,0,0,0]
        a[y_train[i][0]]=1
        y_train[i]=a 
    pickle.dump(file=f, obj=y_train, protocol = 4)

    X_train = training[0]
    X_train=X_train.tolist()
    for i in range(len(X_train)):
        flat_list=[]
        for sublist in X_train[i]:
            for item in sublist:
                flat_list.append(item)
        X_train[i] = flat_list
    for i in range(0, len(X_train), 1000):
        pickle.dump(file=f, obj=X_train[i:i+1000], protocol = 4)
    f.close()

first_image = X_train[0]
first_image = np.array(first_image, dtype='float')
pixels = first_image.reshape((28, 28))
plt.imshow(pixels, cmap='gray')
plt.show()