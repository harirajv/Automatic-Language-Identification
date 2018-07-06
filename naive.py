import csv
import random
import math
def loadCsv(filename):
    lines=csv.reader(open(filename,"r"))
    dataset=list(lines)
    for i in range(len(dataset)):
        dataset[i]=[float(x) for x in dataset[i]]
    return dataset


#filename=r'C:\Users\Sesha.parthasarathy\Desktop\pima-indians-diabetes.csv'
#dataset=loadCsv(filename)
#print("loaded data file {0} with {1} rows".format(filename,len(dataset)))

def splitDataset(dataset,splitRatio,offset):
    trainSize=int(len(dataset)*(1-splitRatio))
    trainSet=[]
    copy2=[]
    copy1=[]
    copy=list(dataset)
    for i in copy:
        if i not in offset:
            copy1.append(i)
        else:
            copy2.append(i)
    for i in range(len(copy1)):
        if i<trainSize:
            trainSet.append(copy1[i])

    #while(len(trainSet)<trainSize):
     #   index=random.randrange(len(copy))
      #  trainSet.append(copy.pop(index))
    return [trainSet,copy2]

#filename=r'C:\Users\Sesha.parthasarathy\Desktop\pima-indians-diabetes.csv'
#dataset=loadCsv(filename)
#splitRatio=0.67
#train,test=splitDataset(dataset,splitRatio)
#print("split {0} rows into train with {1} and test with {2}".format(len(dataset),train,test))

def separateByClass(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separated):
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    return separated
#filename=r'C:\Users\Sesha.parthasarathy\Desktop\pima-indians-diabetes.csv'
#dataset=loadCsv(filename)
#seperated=separateByClass(dataset)
#print("Seperated instances {0}".format(seperated))

def mean(numbers):
    return sum(numbers) / float(len(numbers))


def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x - avg, 2) for x in numbers]) / float(len(numbers) - 1)
    return math.sqrt(variance)

def summarize(dataset):
    summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
    del summaries[-1]
    return summaries

#filename=r'C:\Users\Sesha.parthasarathy\Desktop\pima-indians-diabetes.csv'
#dataset=loadCsv(filename)
#summary=summarize(dataset)
#print("attribute summaries:{0}".format(summary))

def summarizeByClass(dataset):
    separated = separateByClass(dataset)
    summaries = {}
    for classValue, instances in separated.items():
        summaries[classValue] = summarize(instances)
    return summaries


#filename=r'C:\Users\Sesha.parthasarathy\Desktop\pima-indians-diabetes.csv'
#dataset=loadCsv(filename)
#summary=summarizeByClass(dataset)
#print("Summary by class value:{0}".format(summary))

def calculateProbability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent


def calculateClassProbabilities(summaries, inputVector):
    probabilities = {}
   # count=len(summaries)
    for classValue, classSummaries in summaries.items():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = inputVector[i]
            probabilities[classValue] *= (calculateProbability(x, mean, stdev))#*len(summaries[classValue])/count
           # probabilities[classValue]/=len(summaries)#summaries contains all classes as it has split the classes
    return probabilities#probabilities has the conditional probability.we need to multiply by probability of class.
                        #probability of class is (1/total no of classes).len(summaries) has total no of classes

def predict(summaries, inputVector):
    probabilities = calculateClassProbabilities(summaries, inputVector)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.items():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel

def getPredictions(summaries, testSet):
    predictions = []
    for i in range(len(testSet)):
        result = predict(summaries, testSet[i])
        predictions.append(result)
    return predictions

def getAccuracy(testSet, predictions):
    correct = 0
    for i in range(len(testSet)):
        if testSet[i][-1] == predictions[i]:
            correct += 1
    return (correct / float(len(testSet))) * 100.0
d=[]
filename = r'C:\Users\Hari\Desktop\nlp lang\data.csv'
splitRatio = 0.1
dataset = loadCsv(filename)
for i in range(0,10):
    offset = []
    b=i*(len(dataset)*splitRatio)
    b1=b+(len(dataset)*splitRatio)
    for i in range(math.ceil(b),math.ceil(b1)):
        offset.append(dataset[i])
    trainingSet, testSet = splitDataset(dataset, splitRatio,offset)
    print("Split {0} rows into train={1} and test={2} rows".format(len(dataset), len(trainingSet), len(testSet)))
    summary=summarizeByClass(trainingSet)
    predictions=getPredictions(summary,testSet)
    accuracy=getAccuracy(testSet,predictions)
    d.append(accuracy)
acc=sum(d)/len(d)
print("Accuracy:{0}".format(acc))

