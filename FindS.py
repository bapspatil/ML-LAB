import csv
import subprocess as s
s.call('curl https://raw.githubusercontent.com/bapspatil/ML-Lab/master/9.py >> p9.py', shell=True)
with open('trainingexamples.csv') as csvFile:
    data = [line[:-1] for line in csv.reader(csvFile) if line[-1] == "Y"]
print("POSITIVE EXAMPLES ARE:{}".format(data))
S = ['o']*len(data[0])   # Initializing.
print("output in each steps are:\n{}".format(S))
for example in data:
    i = 0
    for feature in example:
        S[i] = feature if S[i] == 'o' or S[i] == feature else '?'
        i += 1
    print(S)