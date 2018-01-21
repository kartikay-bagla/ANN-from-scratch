from struct import unpack
import gzip
from numpy import zeros, uint8, float32
from pylab import imshow, show, cm
import pickle

def get_labeled_data(imagefile, labelfile):
    """Read input-vector (image) and target class (label, 0-9) and return
       it as list of tuples.
    """
    # Open the images with gzip in read binary mode
    images = gzip.open(imagefile, 'rb')
    labels = gzip.open(labelfile, 'rb')

    # Read the binary data

    # We have to get big endian unsigned int. So we need '>I'

    # Get metadata for images
    images.read(4)  # skip the magic_number
    number_of_images = images.read(4)
    number_of_images = unpack('>I', number_of_images)[0]
    rows = images.read(4)
    rows = unpack('>I', rows)[0]
    cols = images.read(4)
    cols = unpack('>I', cols)[0]

    # Get metadata for labels
    labels.read(4)  # skip the magic_number
    N = labels.read(4)
    N = unpack('>I', N)[0]

    if number_of_images != N:
        raise Exception('number of labels did not match the number of images')

    # Get the data
    x = zeros((N, rows, cols), dtype=float32)  # Initialize numpy array
    y = zeros((N, 1), dtype=uint8)  # Initialize numpy array
    for i in range(N):
        if i % 5000 == 0:
            print("i: %i" % i)
        for row in range(rows):
            for col in range(cols):
                tmp_pixel = images.read(1)  # Just a single byte
                tmp_pixel = unpack('>B', tmp_pixel)[0]
                x[i][row][col] = tmp_pixel
        tmp_label = labels.read(1)
        y[i] = unpack('>B', tmp_label)[0]
    return (x, y)

def view_image(image, label=""):
    """View a single image."""
    print("Label: %s" % label)
    imshow(image, cmap=cm.gray)
    show()

def create_data(): 
    
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

    return(X_train, y_train, X_test, y_test)