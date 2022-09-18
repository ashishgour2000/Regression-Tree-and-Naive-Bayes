import numpy as np
import sys
import pandas as pd
from csv import reader
from random import seed
from random import randrange
from math import sqrt
from math import exp
from math import pi

df = pd.read_csv("Train_D_Bayesian.csv")

y_dict = {
    "Abnormal": 0,
    "Normal": 1
}
label = pd.DataFrame(df["Class_att"].map(y_dict))
label.rename({"Class_att": "label"}, inplace=True)
label
dataset = df
new=dataset.drop(columns=["Class_att"])
#dataset
new = pd.concat([new,label],axis=1)

n = np.shape(new)[0]
dimen = 12

mean = [0] * dimen
standard_deviation = [0] * dimen

mean = np.mean(new)
standard_deviation = np.std(new)
max_value = [0] * dimen
max_value = (2*mean) + (5*standard_deviation)

val=0
for i in range(len(new)):
    count=0
    for j in range(12):
        if new.iloc[i][j]>max_value[j]:
            count+=1
    #print(count)
    if count>6:
            #discard sample
            new = new.drop(labels=i, axis=0)
            i-=1
            val+=1


n = np.shape(new)[0]
indices = np.random.permutation(n)

num=n*0.7
num=int(num)
training_idx, test_idx = indices[:num], indices[num:]

training = new.iloc[training_idx]

test = new.iloc[test_idx]

def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    dataset_split = list()
    dataset2 = list(dataset)
    size = int(len(dataset) / n_folds)
    for _ in range(n_folds):
        fold = list()
        while len(fold) < size:
            index = randrange(len(dataset2))
            fold.append(dataset2.pop(index))
        dataset_split.append(fold)
    folds = dataset_split

    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores

def separate_by_class(dataset):
    separated = dict()
    for i in range(len(dataset)):
        vector = dataset[i]
        class_value = vector[-1]
        if (class_value not in separated):
            separated[class_value] = list()
        separated[class_value].append(vector)
    return separated

def mean(numbers):
    return sum(numbers)/float(len(numbers))

def stdev(numbers):
    avg = mean(numbers)
    variance = sum([(x-avg)**2 for x in numbers]) / float(len(numbers)-1)
    return sqrt(variance)

def summarize_dataset(dataset):
    summaries = [(mean(column), stdev(column), len(column)) for column in zip(*dataset)]
    del(summaries[-1])
    return summaries

def summarize_by_class(dataset):
    separated = separate_by_class(dataset)
    summaries = dict()
    for class_value, rows in separated.items():
        summaries[class_value] = summarize_dataset(rows)
    return summaries

def calculate_probability(x, mean, stdev):
    exponent = exp(-((x-mean)**2 / (2 * stdev**2 )))
    return (1 / (sqrt(2 * pi) * stdev)) * exponent

def naive_bayes(train, test):
    separated = separate_by_class(dataset)
    summaries = dict()
    for class_value, rows in separated.items():
        summaries[class_value] = summarize_dataset(rows)

    summarize = summaries

    predictions = list()
    for row in test:
        total_rows = sum([summaries[label][0][2] for label in summaries])
        probabilities = dict()
        for class_value, class_summaries in summaries.items():
            probabilities[class_value] = summaries[class_value][0][2]/float(total_rows)
            for i in range(len(class_summaries)):
                mean, stdev, _ = class_summaries[i]
                probabilities[class_value] *= calculate_probability(row[i], mean, stdev)

        best_label, best_prob = None, -1
        for class_value, probability in probabilities.items():
            if best_label is None or probability > best_prob:
                best_prob = probability
                best_label = class_value
        #output = predict(summarize, row)
        output = best_label
        predictions.append(output)
    return(predictions)


filename = 'Train_D_Bayesian.csv'
dataset = list()
with open(filename, 'r') as file:
    csv_reader = reader(file)
    fields = next(csv_reader)
    for row in csv_reader:
        for line in csv_reader:
            lists = []
            for number in range(0, 12):
                lists.append(line[number])
            val=0
            if line[12]=="Abnormal":
                val=1
            lists.append(val)
            dataset.append(lists)

for i in range(len(dataset[0])-1):
    for row in dataset:
        row[i] = float(row[i].strip())

n=len(dataset[0])-1
n_folds = 5
scores = evaluate_algorithm(dataset, naive_bayes, n_folds)

file = open("q2_output.txt", "w")

print('All Accuracies (in 5 fold cross validation): ')
print(scores)
print('Max Accuracy (in 5 fold cross validation): ')
print(max(scores))

file.write(f"All Accuracies (in 5 fold cross validation): \n{scores}\n")
file.write(f"Max Accuracy (in 5 fold cross validation):  \n{max(scores)}\n")

file.close()