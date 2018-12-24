import os, numpy, random, struct
from helper import verify
from matplotlib import pyplot
from PIL import Image

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


#######################
### Build the model ###
#######################

# Set needed containers
Ks = [20]
centers = []
maps = []
predictions = []
trainResults = [0.] * len(Ks)
testResults = [0.] * len(Ks)

# Call model
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score


###DISCLAIMER: Copied code###
def cluster(k):
    kmeans = KMeans(n_clusters = k, init='random', n_init = 5).fit(trainX)
    trainMapping , trainPredictions = verify(trainY, kmeans.labels_)
    trainResults[Ks.index(k)] = accuracy_score(trainY, trainPredictions)
    testLabels = kmeans.predict(testX)
    testPredictions = [trainMapping[label] for label in test_labels]
    testResults[Ks.index(k)] = accuracy_score(testY, testPredictions)
    centers.append(kmeans.cluster_centers_)
    maps.append(train_mapping)
    predictions.append(trainPredictions)

###DISCLAIMER: Copied code###
def doTheJob(k):
    trainResult = trainResults[Ks.index(k)]
    testResult = testResults[Ks.index(k)]
    clusterCenters = centers[Ks.index(k)]
    mapping = maps[Ks.index(k)]
    Dpredictions = predictions[Ks.index(k)]
    for k_ in numpy.arange(k):
        clusterCenter = clusterCenters[k_]
        meanImage = numpy.reshape(cluster_center, (rows, columns))
        pyplot.imshow(meanImage, interpolation='none')
        pyplot.show()
        sampleVectors= trainX[Dpredictions==mapping[k_], :]
        sampleVector = sampleVectors[numpy.random.randint(sampleVectors.shape[0], size=1), :]
        sampleImage = numpy.reshape(sampleVector, (rows, columns))
        pyplot.imshow(sampleImage)
        pyplot.show()

for k in Ks:
    cluster(k)

for k in Ks:
    doTheJob(k)
