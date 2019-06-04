import csv
import numpy as np
import pandas as pd

# get data
def import_data(filename):
	with open(filename, 'rt') as f:
		reader = csv.reader(f, delimiter=',', skipinitialspace=True)

		lineData = list()
		cols = next(reader)

		for col in cols:
			# create a list in lineData for each column of data
			lineData.append(list())

		for line in reader:
			for i in range(0, len(lineData)):
				lineData[i].append(line[i])
				# Copy the data from the line into the correct columns.
				# lineData[i].append(line[i])

		data = dict()

		for i in range(0, len(cols)):
			# Create each key in the dict with the data in its column.
			data[cols[i]] = lineData[i]

		return data

if __name__ == "__main__":

	files = [
		'classification-test-file/20test_results.csv',
		# 'classification-test-file/19test_results.csv',
		# 'classification-test-file/18test_results.csv',
		# 'classification-test-file/17test_results.csv',
		# 'classification-test-file/16test_results.csv',
		# 'Segment-PCP6.txt'
	]
	# files = ['Segment-PCP3.txt']
	flag = 0
	# result = np.zeros((4600,6))
	# result = dict()
	for file in files:
		print(file)
		flag += 1
		data = import_data(file)
		# print(data['3'])
		df = pd.DataFrame.from_dict(data)
		print(df.mode(axis=1))
		df.mode(axis=1).to_csv('ensamble.csv')
		# print(df['4599'].value_counts().idxmax())
		# for id1, label1 in data:
			# print(id1, label1)
			# if flag == 1:
				# result[id,0] = id
			# result[id,flag] = label

	# print(result)
		# X = df.drop(['Ub', 'Segment'], axis=1).values
		# Y = df.Ub.values




