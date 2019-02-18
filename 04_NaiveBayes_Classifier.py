import sys
import math
import pandas as pd
import numpy as np

#initializing parameters
input_file = sys.argv[1]
output_file = sys.argv[2]

def calculateProbability(x, mean, variance):
	exponent = math.exp(-(math.pow(x - mean, 2) / (2 * variance)))
	return (1 / (math.sqrt(2 * math.pi * variance))) * exponent
	
# read the file into a dataframe
dfdata = pd.read_csv(input_file, sep='\t', header=None)

# Had to add the below if condition because Example.tsv is read as 4 columns
if dfdata.shape[1] == 4:
	dfdata.columns = ['Y','X1','X2','X3']
	df = dfdata.drop(columns=['X3'])
elif dfdata.shape[1] == 3:
	dfdata.columns = ['Y','X1','X2']
	df = dfdata
str1 = ""

# Printing the first two lines of the output. 
subdf = df.loc[df['Y'] == 'A']
str1 = str(subdf["X1"].mean()) + '\t' + str(subdf["X1"].var()) + '\t' + str(subdf["X2"].mean()) + '\t' + str(subdf["X2"].var()) + '\t' + str(subdf.shape[0]/df.shape[0])
#print(str1)
ax1mean = subdf["X1"].mean()
ax1var = subdf["X1"].var()
ax2mean = subdf["X2"].mean()
ax2var = subdf["X2"].var()
aprob = subdf.shape[0]/df.shape[0]

subdf = df.loc[df['Y'] == 'B']
str2 = str(subdf["X1"].mean()) + '\t' + str(subdf["X1"].var()) + '\t' + str(subdf["X2"].mean()) + '\t' + str(subdf["X2"].var()) + '\t' + str(subdf.shape[0]/df.shape[0])
#print(str1)
bx1mean = subdf["X1"].mean()
bx1var = subdf["X1"].var()
bx2mean = subdf["X2"].mean()
bx2var = subdf["X2"].var()
bprob = subdf.shape[0]/df.shape[0]

df['E'] = df['Y'].map({'A': 1, 'B': 0})

ax1 = []
ax2 = []
bx1 = []
bx2 = []


for ins in df['X1']:
	ax1_prob = calculateProbability(ins, ax1mean, ax1var)
	bx1_prob = calculateProbability(ins, bx1mean, bx1var)
        
	ax1.append(ax1_prob)
	bx1.append(bx1_prob)
        
for ins in df['X2']:
	ax2_prob = calculateProbability(ins, ax2mean, ax2var)
	bx2_prob = calculateProbability(ins, bx2mean, bx2var)
        
	ax2.append(ax2_prob)
	bx2.append(bx2_prob)

a_prob = np.multiply(ax1, ax2)
b_prob = np.multiply(bx1, bx2)

df['A'] = a_prob
df['B'] = b_prob

df['O'] = np.where((df['A'] > df['B']), 1, 0)
df['miscls'] = df['E'] - df['O']  
misclassification = str(df[df['miscls']!=0].shape[0])

with open(output_file, 'w') as outputfile:
	outputfile.write(str1)
	outputfile.write('\n')
	outputfile.write(str2)
	outputfile.write('\n')
	outputfile.write(misclassification)
