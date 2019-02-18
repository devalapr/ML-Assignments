# how to run this script:
# Python Rakesh_MLAsgn01.py DSrandom.csv 0.0001 0.0001
import sys
import csv

# input dataframe, weights, prevSSE, currSSE
def linearregr(df, weights, prevSSE, currSSE,iteration):
	prevSSE = currSSE
	currSSE = 0.0
	
	# initialize learning function, error and gradient
	fnofx = [0.0] * num_rows
	errorfn = [0.0] * num_rows
	gradient = [0.0] * num_cols
	
	# Calculate Error and SSE
	for i in range(0,num_rows):
		fnofx[i] = weights[0] # this is including w0 in the calculation
		for j in range(0,num_cols-1): 
			fnofx[i] += weights[j+1]* df[i][j]
		# every row, last column is y. And error function is y - f(x)
		# And currSSE to be updated here. It is the sqaure of Error [y-f(x)]
		errorfn[i] = df[i][num_cols-1]- fnofx[i] 
		
		currSSE += errorfn[i] * errorfn[i]
	
	print(iteration,weights, currSSE)
	#error calculation and SSE calculation is done
	
	#Calculate Gradient now
	for j in range(0,num_cols):
		for i in range(0,num_rows):
			if j == 0:
				gradient[j] += errorfn[i] # taking into consideration x0 = 1
			else:
				gradient[j] += df[i][j-1] * errorfn[i]
	
	# Update weights after calculating gradient 
	for i in range(0,num_cols):
		weights[i] += eeta*gradient[i]
	#print (currSSE, prevSSE)
	
	iteration += 1 #increase the iteration by 1
	
	return currSSE, prevSSE, iteration

####### MAIN EXECUTION STARTS HERE ######
# Open the file and count the number of columns and rows
# Read the file into a dataframe (array)
# read inline arguments into parameters
data_file = sys.argv[1]
eeta = float(sys.argv[2])
threshold = float(sys.argv[3])

with open(data_file, 'r') as infile:
	reader = csv.reader(infile,quoting=csv.QUOTE_NONNUMERIC)
	num_cols = len(next(reader)) # number of columns in the file
	infile.seek(0)
	df = [] # data frame
	for row in reader:  
		df.append(row)

num_rows = len(df) # number of rows in the file

# initialize the variables here	
prevSSE = 9999.0
currSSE = 0.0
iteration = 0
weights = [0.0] * num_cols # initializing the weights list to 0.0

while(prevSSE - currSSE > threshold):
#for i in range(0,5):
	prevSSE, currSSE, iteration = linearregr(df,weights,prevSSE,currSSE, iteration)