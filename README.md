Following are the assignments I coded during the Machine Learning Course:

A detailed instructions for the assignments and the sample input files and output files are given here:
http://www.dke-research.de/en/Studies/Courses/Winter+Term+2018_2019/Machine+Learning.html 

1. Linear Regression using batch gradient descent method.

All weights initialized to 0 at the beginning. The program reads an input file and considers the last value in each line as the target output.
The program implements the gradient descent method and return for each iteration the weights and sum of squared errors until a given threshold
of change in the error is reached. The output looks like this:
iteration_number,weight0,weight1,weight2,...,weightN,sum_of_squared_errors

The program accepts the following parameters:
i) Location of the input data file
ii) Learning rate of gradient descent approach
iii) threshold for the algorithm to terminate

example statements to use the program:
py 01_Linear_Regression.py inputfile.csv 0.0001 0.001

Sample contents in input file:
-3.4000,-0.6100,2.738
-3.1100,3.8500,0.3666666666666665
-0.1200,-2.7200,0.44266666666666676

Another sample:
-2.3,0.568,4.78,3.99,3.17,0.125,0.11
-2.3,0.568,4.78,3.99,3.17,0.150,0.27
-2.3,0.568,4.78,3.99,3.17,0.175,0.47
-2.3,0.568,4.78,3.99,3.17,0.200,0.78
-2.3,0.568,4.78,3.99,3.17,0.225,1.18
-2.3,0.568,4.78,3.99,3.17,0.250,1.82

2. Decision Tree implementation using ID3 algorithm and information gain as the measure.

The program implements the ID3 algorithm and return the final tree without stopping early. 
The output of the program is an XML file with the final tree structure. Program takes two inputs:
i) Location of the input file
ii) Location to which the output XML file to be written

Sample contents in input file:
vhigh,vhigh,2,2,small,low,unacc
vhigh,vhigh,2,2,small,med,unacc
vhigh,vhigh,2,2,small,high,unacc
vhigh,vhigh,2,2,med,low,unacc
vhigh,vhigh,2,2,med,med,unacc

example statements to use the program:
py 02_DecisionTree_ID3.py inputfile.csv output.xml


3. Single Perceptron using Delta rule

The output is a tsv file, which contains exactly two rows after 100 iterations (per
variant):
i) The first row contains the tabular separated values for the error of each iteration
(starting from iteration 0) with the constant learning rate.
ii) The second row follows the same format, but with the annealing learning rate

example statements to use the program:
py 03_Perceptron.py inputfile.csv outputfile.tsv

4. 2 class Naive Bayes classifier 

The program implements a 2-class Naive Bayes algorithm with an apriori decision
rule using a multinomial estimation for the classes and a gaussian estimation for the
attributes. The output of the program is a single tsv file per data set, which contains a row for each class:
mean and variance and the probability
The last (third) row contains the absolute number of misclassications for the data.

Program takes two inputs:
i) Location of the input file
ii) Location to which the output file to be written

example statements to use the program:
py 04_NaiveBayes_Classifier.py inputfile.tsv output.tsv
