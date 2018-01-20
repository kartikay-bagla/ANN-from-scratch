from mnist_loader import *
import numpy as np #using numpy for arrays and optimized matrix multiplication
np.random.seed(1) #seeding so that repeated results are the same and we can observe
#changes from editing.
np.seterr(over = 'ignore')
np.set_printoptions(precision=0, suppress=True)
import pickle
import os.path

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

#Sigmoid Function
def sigmoid (x):
    return 1/(1 + np.exp(-x))

#Derivative of Sigmoid Function
def derivatives_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

#print (X_test[0])

#Variable initialization
no_epoch = 100 #Setting training iterations
lrate = 0.1 #Setting learning rate
inputlayer_neurons = len(X_train[0]) #no of inputs (in this case 4)
#print (inputlayer_neurons)

hiddenlayer_neurons = 500 #number of hidden layers neurons
output_neurons = 10 #number of neurons at output layer

if os.path.isfile('w_b.dat'):
    with open('w_b.dat', 'rb') as f:
        wh=pickle.load(f)
        bh=pickle.load(f)
        bout = pickle.load(f)
        wout = pickle.load(f)
else:
    wh = np.random.uniform(size = (inputlayer_neurons, hiddenlayer_neurons))
    bh = np.random.uniform(size = (1, hiddenlayer_neurons))


    wout = np.random.uniform(size = (hiddenlayer_neurons, output_neurons))
    bout = np.random.uniform(size = (1, output_neurons))


def network(X, y, epoch, lr, wh, bh, wout, bout):
    for i in range(epoch):
        correctl = 0
        corrects = 0
        for j in range(len(X)):
            
            hl_wht = np.dot(np.array([X[j]]), wh)
            #print(hl_wht.shape)
            hl_b = hl_wht + bh
            #print (hl_b.shape)
            hiddenlayer_input = sigmoid(hl_b)
            #print(hiddenlayer_input.shape)
            o_wht = np.dot(hiddenlayer_input, wout)
            o_b = o_wht + bout
            output = sigmoid(o_b)
            #print(output.shape)


            E=y[j]-output
            #print(y[j])
            slope_output = derivatives_sigmoid(output)
            slope_hidden_layer = derivatives_sigmoid(hiddenlayer_input)

            d_output = E * slope_output
            #print (E.shape, slope_output.shape)
            #print (d_output.shape)
            E_hidden_layer = np.dot(d_output, wout.T)
            
            #print(E_hidden_layer.shape, slope_hidden_layer.shape)
            
            d_hidden_layer = E_hidden_layer * slope_hidden_layer
            
            wout += np.dot(hiddenlayer_input.T, d_output)*lr
            wh += np.dot(np.array([X[j]]).T, d_hidden_layer)*lr

            bh += np.sum(d_hidden_layer, axis=0) * lr
            bout += np.sum(d_output, axis = 0) * lr
            ol = output.tolist()
            #print(output, y[j])
            #print (ol.index(max(ol)), y[j].index(max(y[j])))
            if ol.index(max(ol)) == y[j].index(max(y[j])):
                corrects += 1
                correctl += 1
            if j%100 == 0:
                accuracy = (corrects/100)*100
                print("Epoch:", i,"Image No: ", j, "Accuracy:", accuracy)
                corrects=0
        accuracy = (correctl/len(X))*100
        if i%1==0:
            print("Accuracy:", accuracy)
        with open('w_b.dat', 'wb') as f:
            pickle.dump(file = f, obj = wh)
            pickle.dump(file = f, obj = bh)
            pickle.dump(file = f, obj = bout)
            pickle.dump(file = f, obj = wout)

network(X_train, y_train, no_epoch, lrate, wh, bh, wout, bout)