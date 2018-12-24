import numpy

def verify(trueLabels, kLabels):
    mapp = {k: k for k in numpy.unique(kLabels)}
    for k in numpy.unique(kLabels):
        k_mapping = numpy.argmax(numpy.bincount(trueLabels[kLabels==k]))
        mapp[k] = k_mapping
    predictions = [mapp[label] for label in kLabels]
    return mapp, predictions
