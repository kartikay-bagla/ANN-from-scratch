import numpy as np #using numpy for arrays and optimized matrix multiplication
np.random.seed(1) #seeding so that repeated results are the same and we can observe
#changes from editing.
from sklearn.model_selection import train_test_split #used to randomly split the 
#dataset into 2 parts so that we can train and test on different datapoints

#Sigmoid Function
def sigmoid (x):
    return 1/(1 + np.exp(-x))

#Derivative of Sigmoid Function
def derivatives_sigmoid(x):
    return x * (1 - x)

def input_table(x): #function to a truth table with n variables
    l = []
    for i in range(2**x):
        a = bin(i).lstrip("0b")
        while len(a) < x:
            a = '0' + a
        l.append(list(a))

    for i in range(len(l)):
        l[i] = [int(x) for x in l[i]]
    return np.array(l)
 

X = input_table(8) #primary input which is a dataset of a 8 variable
#truth table
print (X)
y = [] #outputs for corresponding values in table
with open('abc.txt', 'r') as f: 
    s = f.read() #outputs stored in a txt file as numbers eg 010101101....
    x = list(s)
    y = x
y = [[int(i)] for i in y]
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
#splitting the data into 2 parts


#Variable initialization
epoch = 5001 #Setting training iterations
lr = 0.1 #Setting learning rate
inputlayer_neurons = X.shape[1] #no of inputs (in this case 4)
hiddenlayer_neurons = 3 #number of hidden layers neurons
output_neurons = 1 #number of neurons at output layer

#weight and bias initialization
wh = np.random.uniform(size=(inputlayer_neurons,hiddenlayer_neurons))
bh = np.random.uniform(size=(1,hiddenlayer_neurons))
#hidden layer weights and biases

wout = np.random.uniform(size=(hiddenlayer_neurons,output_neurons))
bout = np.random.uniform(size=(1,output_neurons))
#output layer weights and biases


#training loop
for i in range(epoch):

    #Forward Propogation
    hidden_layer_input1 = np.dot(X_train,wh)
    hidden_layer_input = hidden_layer_input1 + bh
    hiddenlayer_activations = sigmoid(hidden_layer_input)
    output_layer_input1 = np.dot(hiddenlayer_activations,wout)
    output_layer_input = output_layer_input1+ bout
    output = sigmoid(output_layer_input)

    #Backpropagation
    E = y_train-output
    slope_output_layer = derivatives_sigmoid(output)
    slope_hidden_layer = derivatives_sigmoid(hiddenlayer_activations)
    d_output = E * slope_output_layer
    Error_at_hidden_layer = d_output.dot(wout.T)
    d_hiddenlayer = Error_at_hidden_layer * slope_hidden_layer
    wout += hiddenlayer_activations.T.dot(d_output) *lr
    bout += np.sum(d_output, axis=0,keepdims=True) *lr
    wh += X_train.T.dot(d_hiddenlayer) *lr
    bh += np.sum(d_hiddenlayer, axis=0,keepdims=True) *lr
    if i%50 == 0:
        accuracy = 0
        for x in range(len(output)):
            if int(round(output[x][0])) == y_train[x][0]:  
                accuracy += 1
        print (i, "Accuracy: ", accuracy)

#test
hidden_layer_input1 = np.dot(X_test,wh)
hidden_layer_input = hidden_layer_input1 + bh
hiddenlayer_activations = sigmoid(hidden_layer_input)
output_layer_input1 = np.dot(hiddenlayer_activations,wout)
output_layer_input = output_layer_input1+ bout
output = sigmoid(output_layer_input)
accuracy = 0
for x in range(len(output)):
    if int(round(output[x][0])) == y_test[x][0]:  
        accuracy += 1
print ("Test Accuracy: ", accuracy)