import sys
import math
import pandas as pd
import numpy as np

#initializing parameters
input_file = sys.argv[1]
output_file = sys.argv[2]
learning_rate = 1.0
w0 = 0.0
w1 = 0.0
w2 = 0.0
outstr = ""

# read the file into a dataframe
dfdata = pd.read_csv(input_file, sep='\t', header=None)

# Had to add the below if condition because Example.tsv is read as 4 columns
if dfdata.shape[1] == 4:
	dfdata.columns = ['Y','X1','X2','X3']
	df = dfdata.drop(columns=['X3'])
elif dfdata.shape[1] == 3:
	dfdata.columns = ['Y','X1','X2']
	df = dfdata

#mapping the value of A to 1 and B to 0
df["Y"] = df["Y"].map({'A':1,'B':0})

for i in range(0,101):
	df['fx'] = w0 + w1*df['X1'] + w2*df['X2']
	df['Ybar'] = 0
	# if fx is greater than 0, y^ = 1
	df.loc[df['fx'] > 0, 'Ybar'] = 1

	# Adding the y^ variable as well to the dataframe
	df['y-y^'] = df['Y'] - df['Ybar']
	df['y-y^x1'] = df['y-y^']*df['X1']
	df['y-y^x2'] = df['y-y^']*df['X2']

	# misclassified count. If 1 and 1, or 0 and 0 - it is rightly classified. So, the difference should not be equal to 0
	count = df[df['y-y^']!=0]
	outstr = outstr + str(count.shape[0]) + '\t'
	
	w0 = w0 + learning_rate * df['y-y^'].sum()
	w1 = w1 + learning_rate * df['y-y^x1'].sum()
	w2 = w2 + learning_rate * df['y-y^x2'].sum()
#print(outstr)
# initializing the weights to 0.0 for the annealing learning rate
w0 = 0.0
w1 = 0.0
w2 = 0.0
outstr1 = ""
for i in range(0,101):
	df['fx'] = w0 + w1*df['X1'] + w2*df['X2']
	df['Ybar'] = 0
	# if fx is greater than 0, y^ = 1
	df.loc[df['fx'] > 0, 'Ybar'] = 1

	# Adding the y^ variable as well to the dataframe
	df['y-y^'] = df['Y'] - df['Ybar']
	df['y-y^x1'] = df['y-y^']*df['X1']
	df['y-y^x2'] = df['y-y^']*df['X2']

	# misclassified count. If 1 and 1, or 0 and 0 - it is rightly classified. So, the difference should not be equal to 0
	count = df[df['y-y^']!=0]
	outstr1 = outstr1 + str(count.shape[0]) + '\t'
	
	w0 = w0 + (learning_rate/(i+1)) * df['y-y^'].sum()
	w1 = w1 + (learning_rate/(i+1)) * df['y-y^x1'].sum()
	w2 = w2 + (learning_rate/(i+1)) * df['y-y^x2'].sum()
#print(outstr1)
with open(output_file, 'w') as outputfile:
	outputfile.write(outstr)
	outputfile.write('\n')
	outputfile.write(outstr1)
