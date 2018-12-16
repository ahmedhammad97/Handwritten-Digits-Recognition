import os, numpy, random, struct

# Load Images
trainingInput = open(os.path.relpath("datasets/train-images-idx3-ubyte"), 'rb')
trainingInput.seek(0)
MSB = struct.unpack('>4B', trainingInput.read(4))
imageCount = struct.unpack('>I', trainingInput.read(4))[0]
rows = struct.unpack('>I', trainingInput.read(4))[0]
columns = struct.unpack('>I', trainingInput.read(4))[0]

print(MSB, imageCount, rows, columns)


# Convert to numeric matrix
images = numpy.asarray(struct.unpack('>' + 'B'*imageCount*rows*columns, trainingInput.read(imageCount*rows*columns)))
images = numpy.reshape(images, (imageCount, rows, columns))

print(len(images))

# Faltten to vector
# Add to numpy Array
# Plot
# Start training
