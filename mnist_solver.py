from mnist_loader import * #custom module(.py file) for loading the MNIST database.
from mnist_printer import *
import numpy as np #using numpy for arrays and optimized matrix multiplication
np.random.seed(1) #seeding so that repeated results are the same and we can observe
#changes from editing.
np.seterr(over = 'ignore') #ignore any overflows i.e. when value of decimal is
#than the variable can store.
np.set_printoptions(precision=0, suppress=True) #to round off all printed numpy 
#arrays to integers as values lie in the range of 10^-34.
import pickle 
import os.path

if os.path.isfile('X_test.dat'): #checking if file exists.
    
    X_train = []
    with open('X_test.dat', 'rb') as f: #loading data from file.
        
        X_test = pickle.load(f)

        
        y_test = pickle.load(f)

        
        y_train = pickle.load(f)

        
        
        while True:
            try:
                X_train+=pickle.load(f)
            except (EOFError, MemoryError):
                break

else: #if file DNE, creating the data using function in mnist_loader.
    
    X_train, y_train, X_test, y_test = create_data() 

#Sigmoid Function
def sigmoid (x):
    return 1/(1 + np.exp(-x))

#Derivative of Sigmoid Function
def derivatives_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

#Variable initialization
no_epoch = 100 #Setting training iterations
lrate = 0.1 #Setting learning rate
inputlayer_neurons = len(X_train[0]) #no of inputs (in this case 4)

hiddenlayer2_neurons = 500 ############################################################################################################

hiddenlayer_neurons = 500 #number of hidden layers neurons
output_neurons = 10 #number of neurons at output layer

if os.path.isfile('w_b.dat'):
    with open('w_b.dat', 'rb') as f:
        wh=pickle.load(f)
        bh=pickle.load(f)
        wh2=pickle.load(f)
        bh2=pickle.load(f)
        bout = pickle.load(f)
        wout = pickle.load(f)
else:
    wh = np.random.uniform(size = (inputlayer_neurons, hiddenlayer_neurons))
    bh = np.random.uniform(size = (1, hiddenlayer_neurons))

    wh2=np.random.uniform(size=(hiddenlayer_neurons, hiddenlayer2_neurons)) #############################################################
    bh2 = np.random.uniform(size = (1, hiddenlayer2_neurons)) ############################################################################

    wout = np.random.uniform(size = (hiddenlayer2_neurons, output_neurons)) ##### revert to hiddenlayer_neurons
    bout = np.random.uniform(size = (1, output_neurons))


def network(X, y, epoch, lr, wh, bh, wh2, bh2, wout, bout):
    for i in range(epoch):
        correctl = 0
        corrects = 0
        for j in range(len(X)):
            
            hl_wht = np.dot(np.array([X[j]]), wh)
            hl_b = hl_wht + bh
            hiddenlayer_input = sigmoid(hl_b)

            hl2_wht = np.dot(hiddenlayer_input, wh2)
            hl2_b = hl2_wht + bh2
            hiddenlayer2_input = sigmoid(hl2_b)

            o_wht = np.dot(hiddenlayer2_input, wout)
            o_b = o_wht + bout
            output = sigmoid(o_b)
            #print(output.shape)


            
            #print(y[j])
            slope_output = derivatives_sigmoid(output)
            slope_hidden_layer2 = derivatives_sigmoid(hiddenlayer2_input)
            slope_hidden_layer = derivatives_sigmoid(hiddenlayer_input)

            E=y[j]-output
            d_output = E * slope_output

            E_hidden_layer2 = np.dot(d_output, wout.T)
            d_hidden_layer2 = E_hidden_layer2 * slope_hidden_layer2

            E_hidden_layer = np.dot(d_hidden_layer2, wh2.T)
            d_hidden_layer = E_hidden_layer * slope_hidden_layer
            

            wout += np.dot(hiddenlayer2_input.T, d_output) *lr
            bout += np.sum(d_output, axis = 0) * lr

            wh2 += np.dot(hiddenlayer_input.T, d_hidden_layer2) *lr
            bh2 += np.sum(d_hidden_layer2, axis = 0) *lr

            wh += np.dot(np.array([X[j]]).T, d_hidden_layer) *lr
            bh += np.sum(d_hidden_layer, axis=0) * lr
            
            ol = output.tolist()
            #print(output, y[j])
            #print (ol.index(max(ol)), y[j].index(max(y[j])))
            if ol.index(max(ol)) == y[j].index(max(y[j])):
                corrects += 1
                correctl += 1
            if j%100 == 0:
                accuracy = (corrects/100.0)*100
                print("Epoch:", i,"Image No:", j, "Accuracy:", accuracy)
                corrects=0
                #print_img(X[j])
        accuracy = (correctl/len(X))*100
        if i%1==0:
            print("Accuracy:", accuracy)
        with open('w_b.dat', 'wb') as f:
            pickle.dump(file = f, obj = wh)
            pickle.dump(file = f, obj = bh)
            pickle.dump(file = f, obj = wh2)
            pickle.dump(file = f, obj = bh2)
            pickle.dump(file = f, obj = bout)
            pickle.dump(file = f, obj = wout)

network(X_train, y_train, no_epoch, lrate, wh, bh, wh2, bh2, wout, bout)