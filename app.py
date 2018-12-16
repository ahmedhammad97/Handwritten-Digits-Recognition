import os, numpy, random, struct

##############################
### Coded by: Ahmed Hammad ###
###    ahmedhammad.co.nf   ###
##############################

# Load Header
trainingInput = open(os.path.relpath("datasets/train-images-idx3-ubyte"), 'rb')
trainingInput.seek(0)
MSB = struct.unpack('>4B', trainingInput.read(4))
imageCount = struct.unpack('>I', trainingInput.read(4))[0]
rows = struct.unpack('>I', trainingInput.read(4))[0]
columns = struct.unpack('>I', trainingInput.read(4))[0]


# Load images to numeric matrix
images = numpy.asarray(struct.unpack('>' + 'B'*imageCount*rows*columns, trainingInput.read(imageCount*rows*columns)))
images = numpy.reshape(images, (imageCount, rows, columns))


# Faltten to vector
vectors = numpy.reshape(images, (imageCount, rows*columns)) / 255


# Load labels
labelInput = open(os.path.relpath("datasets/train-labels-idx1-ubyte"), 'rb')
labelInput.seek(0)
MSB = struct.unpack('>4B', labelInput.read(4))
labelCount = struct.unpack('>I', labelInput.read(4))[0]


# Convert labels to vector
labels = numpy.asarray(struct.unpack('>' + 'B'*labelCount, labelInput.read(labelCount))).reshape((labelCount))


# Shuffle
permutation = numpy.random.permutation(len(vectors))
vectors = vectors[permutation]
labels = labels[permutation]


# Split records 3:1
splitRatio = int(len(vectors)*0.25)
trainX = vectors[splitRatio:]
trainY = labels[splitRatio:]
testX = vectors[:splitRatio]
testY = labels[:splitRatio]
