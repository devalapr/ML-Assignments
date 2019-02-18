import sys
import math
import pandas as pd
input_file = sys.argv[1]
output_file = sys.argv[2]


# ############# Entropy calculation ###############
def calculate_entropy(data, features, target, log_base):
	target_class = {}
	entropy = 0.0
	# Find index of the target attribute
	target_index = features.index(target)
	#print("target index is ", target_index)
	# Find the frequency of each target value
	
	for each_entry in data:
		if each_entry[target_index] in target_class:
			target_class[each_entry[target_index]] += 1
		else:
			target_class[each_entry[target_index]] = 1
	total_instances = sum(target_class.values())
	
	for key in target_class:
		class_fraction = target_class[key] / total_instances 
		entropy -= class_fraction * (math.log(class_fraction,log_base))
	#print(entropy)
	return entropy

# ############# Unique Values of Attribute ###############
def get_unique_values(data, features, attribute):
	unique_values = []
	index = features.index(attribute) # Attribute Index
	for row in data:
		if row[index] not in unique_values:
			unique_values.append(row[index])
	return unique_values

# ############# Whole data Entropy calculation ###############
def Data_entropy(rawdata, features, target):
	global log_base
	raw_panda_data = pd.read_csv(input_file,header=None)	# reading the input file to a pandas data frame with no header
	num_instances = raw_panda_data.shape[0]					# number of rows or training instances	
	num_features = raw_panda_data.shape[1]					# number of columns or features
	features_set = raw_panda_data.iloc[:,0:num_features-1]	# Features set.
	target_class = raw_panda_data.iloc[:,num_features-1:num_features]	# Target Class
	log_base = len(target_class[num_features-1].unique())			# Number of unique values of target class

	# Gets entropy of the entire dataset
	data_entropy = calculate_entropy(rawdata, features, target, log_base)
	#print(data_entropy)
	return data_entropy

# ################ Get Target Values ##################
def get_target_values(data, features, target):
	target_values = []
	for row in data:
		target_index = features.index(target)
		#value = row[target_index]
		target_values.append(row[target_index])
	#print target_values
	return target_values

# ################# Information Gain ###############
def information_gain(data, features, attribute, target):
	attribute_entropy = 0.0
	dictionary = {}
	index = features.index(attribute)

    # Calculate the frequency of each of the values in the target attribute
	for row in data:
		if row[index] in dictionary:
			dictionary[row[index]] += 1
		else:
			dictionary[row[index]] = 1

	for key in dictionary.keys():
		sub_data = []
		fraction = dictionary[key] / sum(dictionary.values())
		for row in data:
			if row[index] == key:
				sub_data.append(row)
		entropy = calculate_entropy(sub_data, features, target, log_base)
		attribute_entropy += fraction * entropy
    
	information_gain = calculate_entropy(data, features, target, log_base) - attribute_entropy
	return information_gain

# ### Get data for subtree  ####
def get_subtree_data(data, features, attribute, edge):
	sub_data = [[]]
	index = features.index(attribute)
	for row in data:
		if row[index] == edge:
			new_instance = []
			for i in range(0, len(row)):
				if i != index:
					new_instance.append(row[i])
			sub_data.append(new_instance)
	sub_data.remove([])
	return sub_data

# ### Function to choose the best attribute to split  ####
def best_split(data, features, target):
	max_infogain = 0.0
	split_on = features[0]
	
    # Iterates through the features
	for attribute in features:
		if attribute != target:
			new_infogain = information_gain(data, features, attribute, target)
		if new_infogain > max_infogain:
			max_infogain = new_infogain
			split_on = attribute
	return split_on

# ################# Start ID3 #######################
def id3(data, features, target):
	target_values = get_target_values(data, features, target)
	#print(target_values)
	if target_values.count(target_values[0]) == len(target_values):
		return target_values[0]
	else:
		split_attribute = best_split(data, features, target)
		#print(split_attribute)
		edges = get_unique_values(data, features, split_attribute)
		#print(edges)
		for edge in edges:
			next_node_data = get_subtree_data(data, features, split_attribute, edge)
			new_features = features[:]
			new_features.remove(split_attribute)
			
			index = new_features.index(target)
			frequency = {}
            # Calculate the frequency of each of the values in the target attribute
			for row in next_node_data:
				if row[index] in frequency:
					frequency[row[index]] += 1
				else:
					frequency[row[index]] = 1
			node_entropy = calculate_entropy(next_node_data, new_features, target, log_base)

			if node_entropy <= 0.0:
				for label in frequency:
					leaf_label = label
				#print(leaf_label)
				outstring = "feature=\"" + split_attribute + "\" value=\"" + edge + "\">" + leaf_label
			else:
				outstring = "feature=\"" + split_attribute + "\" value=\"" + edge + "\">"

			with open(output_file, 'a') as outputfile:
				outputfile.write("<node entropy=\"" + str(node_entropy) + "\" " + outstring)

			id3(next_node_data, new_features, target)

			with open(output_file, 'a') as outputfile:
				outputfile.write("</node>")

# ################ End of ID3 #######################

####################################################
############## Main code starts here ###############
####################################################
rawdata = []

# Read from file and store in data list
with open(input_file, 'r') as inputfile:
	for line in inputfile:
		line = line.strip("\r\n")
		rawdata.append(line.split(','))

if input_file == "car.csv":
	features = ['att0','att1','att2','att3','att4','att5','target']
elif input_file == "nursery.csv":
	features = ['att0','att1','att2','att3','att4','att5','att6','att7','target']
target = 'target'

with open(output_file, 'w') as outputfile:
	outputfile.write("<?xml version=\"1.0\" encoding=\"ISO-8859-1\"?>")
	outputfile.write("<tree entropy=\"" + str(Data_entropy(rawdata,features,target)) + "\">")
	
id3(rawdata, features, target)

# The last ending XML tag
with open(output_file, 'a') as outputfile:
        outputfile.write("</tree>")