import os, numpy, random, struct

# Load Images
trainingInput = open(os.path.relpath("datasets/train-images-idx3-ubyte"), 'rb')
trainingInput.seek(0)
MSB = struct.unpack('>4B', trainingInput.read(4))
imageCount = struct.unpack('>I', trainingInput.read(4))[0]
rows = struct.unpack('>I', trainingInput.read(4))[0]
columns = struct.unpack('>I', trainingInput.read(4))[0]


# Convert to numeric matrix
images = numpy.asarray(struct.unpack('>' + 'B'*imageCount*rows*columns, trainingInput.read(imageCount*rows*columns)))
images = numpy.reshape(images, (imageCount, rows, columns))


# Faltten to vector
vectors = numpy.reshape(images, (imageCount, rows*columns)) / 255

# Plot
# Start training
