# Multi Layer Perceptron Backpropagation (1 hidden layer) example

Based on the great article: [Neural network backpropagation with Java](https://kunuk.wordpress.com/2010/10/11/neural-network-backpropagation-with-java/)

Other resources:
[A Step by Step Backpropagation Example](https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/)

# PARAMETERS

## FUNCTION APPROXIMATION ARGUMENTS
in=1
inf=1
hidden=20
out=1
outf=2
maxEpoch=200000
printApproximation=true
printError=false
trainDSPath=data\approximation_train_1.txt
testDSPath=data\approximation_test.txt

## TRANSFORMATION
in=4
inf=1,2,3,4
hidden=2
out=4
outf=1,2,3,4
maxEpoch=1500000
normalizeFeatures=false
printApproximation=false
printError=true
trainDSPath=data\transformation.txt
testDSPath=data\transformation.txt

## CLASSIFICATION
in=4
inf=1,2,3,4
hidden=8
out=3
outf=5
outFeatureAs01=true
maxEpoch=50000
normalizeFeatures=false
printApproximation=false
printError=true
trainDSPath=data\classification_train.txt
testDSPath=data\classification_test.txt

