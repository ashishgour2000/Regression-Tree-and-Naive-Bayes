import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

# Calculates mean square error between actual and predicted Y values
def mean_square_error(Y, Y_pred):
    return np.square(np.subtract(Y, Y_pred)).mean()

# Calculates the moving average in an array x with window size b
def moving_avg(x, b):
    return np.convolve(x, np.ones(b), 'valid')/b

# Class for regression tree node
class Node:

    def __init__(self, X=None, Y=None, node_type=None, depth=1, condition="", max_depth=10, min_samples_split=3):

        self.X = X
        self.Y = Y

        self.node_type = node_type
        self.condition = condition

        self.depth = depth
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

        self.y_avg = np.mean(Y)
        self.res = self.Y - self.y_avg
        self.mse = np.sum((Y-self.y_avg)**2)/len(Y)
        self.N = len(Y)

        self.features = list(self.X.columns)
        self.best_feature = None
        self.best_value = None

        self.left = None
        self.right = None

    # Finds best split in current node
    def find_best_split(self):

        curr_mse = self.mse
        best_feature = None
        best_value = None

        data = self.X.copy()
        data["Y"] = self.Y

        for feature in self.features:

            vals = []

            # If feature is a yes/no question, divide it without any numerical calculation
            # Else, divide the tree based on the average of adjacent unique feature values
            if data[feature].dtype != np.int64 and data[feature].dtype != np.float64:
                y_left = data[data[feature] == "yes"]["Y"].values
                y_right = data[data[feature] == "no"]["Y"].values

                res_left = y_left - np.mean(y_left) if len(y_left) > 0 else []
                res_right = y_right - np.mean(y_right) if len(y_right) > 0 else []

                res = np.concatenate((res_left, res_right), axis = 0)

                mse = np.sum(res**2)/len(res)

                if mse < curr_mse:
                    curr_mse = mse
                    best_feature = feature
                    best_value = None
            else:
                # Calculates the average of adjacent unique feature values
                # Tree will be split on these values
                vals = moving_avg(data[feature].unique(), 2)
                for val in vals:
                    y_left = data[data[feature] <= val]["Y"].values
                    y_right = data[data[feature] > val]["Y"].values

                    res_left = y_left - np.mean(y_left) if len(y_left) > 0 else []
                    res_right = y_right - np.mean(y_right) if len(y_right) > 0 else []

                    res = np.concatenate((res_left, res_right), axis = 0)

                    mse = np.sum(res**2)/len(res)

                    if mse < curr_mse:
                        curr_mse = mse
                        best_feature = feature
                        best_value = val

        return best_feature, best_value

    # Builds regression tree by recursively finding best splits at each node
    def build_regression_tree(self):

        data = self.X.copy()
        data["Y"] = self.Y

        # If max depth is reached or no. of observations are low, we don't split further
        if (self.depth < self.max_depth) and (self.N >= self.min_samples_split):

            # Getting the best split 
            best_feature, best_value = self.find_best_split()

            if best_feature is not None:
                # Saving the best split to the current node 
                self.best_feature = best_feature
                self.best_value = best_value

                if data[best_feature].dtype == np.int64 or data[best_feature].dtype == np.float64:

                    left_df, right_df = data[data[best_feature]<=best_value].copy(), data[data[best_feature]>best_value].copy()

                    # Creating left node
                    left = Node(
                        left_df[self.features], 
                        left_df['Y'].values.tolist(),
                        node_type='left_node',
                        depth=self.depth + 1, 
                        condition=f"{best_feature} <= {round(best_value, 3)}",
                        max_depth=self.max_depth, 
                        min_samples_split=self.min_samples_split
                    )

                    # Recursively building left node
                    self.left = left 
                    self.left.build_regression_tree()

                    # Creating right node
                    right = Node(
                        right_df[self.features], 
                        right_df['Y'].values.tolist(),
                        node_type='right_node',
                        depth=self.depth + 1, 
                        condition=f"{best_feature} > {round(best_value, 3)}",
                        max_depth=self.max_depth, 
                        min_samples_split=self.min_samples_split
                    )

                    # Recursively building right node
                    self.right = right
                    self.right.build_regression_tree()
                else:
                    left_df, right_df = data[data[best_feature]=="yes"].copy(), data[data[best_feature]=="no"].copy()

                    # Creating left node
                    left = Node(
                        left_df[self.features], 
                        left_df['Y'].values.tolist(),
                        node_type='left_node',
                        depth=self.depth + 1, 
                        condition=f"{best_feature} == yes",
                        max_depth=self.max_depth, 
                        min_samples_split=self.min_samples_split
                    )

                    # Recursively building left node
                    self.left = left 
                    self.left.build_regression_tree()

                    # Creating right node
                    right = Node(
                        right_df[self.features], 
                        right_df['Y'].values.tolist(),
                        node_type='right_node',
                        depth=self.depth + 1, 
                        condition=f"{best_feature} == no",
                        max_depth=self.max_depth, 
                        min_samples_split=self.min_samples_split
                    )

                    # Recursively building right node
                    self.right = right
                    self.right.build_regression_tree()

    # Pretty prints current node of the tree
    def print_node(self, width=4):
        indent = int(self.depth * width ** 1.5)
        dashes = "-" * indent
        spaces = ' ' * indent
        
        if self.node_type == 'root':
            print("Root")
        else:
            print(f"|{dashes} Split condition: {self.condition}")
        print(f"{spaces}   |-> Mean Square Error: {round(self.mse, 2)}")
        print(f"{spaces}   |-> No. of observations: {self.N}")

        if self.left is None and self.right is None:
            print(f"{spaces}   |-> Prediction: {round(self.y_avg, 3)}")   

    # Pretty prints entire tree
    def print_tree(self):
        self.print_node() 
        
        if self.left is not None: 
            self.left.print_tree()
        
        if self.right is not None:
            self.right.print_tree()

    # Predict value of a specific sample
    def predict_val(self, x):
        if self.left == None and self.right == None:
            return self.y_avg

        if self.best_value is None:
            if x[self.best_feature] == "yes":
                return self.left.predict_val(x)
            else:
                return self.right.predict_val(x)
        else:
            if x[self.best_feature] <= self.best_value:
                return self.left.predict_val(x)
            else:
                return self.right.predict_val(x)

    # Predicts value of entire dataset
    def predict(self, X):
        rows = X.to_dict(orient='records')
        Y = []

        for row in rows:
            Y.append(self.predict_val(row))

        return Y

    # Obtains depth of current node
    def get_depth(self):
        if self.left == None and self.right == None:
            return 1
        elif self.left == None:
            return 1+self.right.get_depth()
        elif self.right == None:
            return 1+self.left.get_depth()
        else:
            return 1+max(self.left.get_depth(), self.right.get_depth())
        
         
def main_func():

    # Reading data set and listing features
    df = pd.read_csv("Train_D_Tree.csv")
    features = ["Extra Cheeze", "Extra Mushroom", " Size by Inch", "Extra Spicy"]

    file = open("q1_output.txt", "w")

    # Initializing parameters
    best_tree = None
    best_tree_index = 0
    best_error = 0
    best_acc = 0
    best_X_train, best_Y_train, best_X_test, best_Y_test = None, None, None, None

    # Iterating through 10 random splits
    for i in range(10):
        print("Training Regression tree no.", i+1)
        df_train = df.sample(frac=0.7)
        df_test = df.drop(df_train.index)

        # Getting training and testing data
        X_train, Y_train = df_train[features], df_train["Price"].values.tolist()
        X_test, Y_test = df_test[features], df_test["Price"].values.tolist()

        # Building regression tree
        root = Node(X_train, Y_train)
        root.build_regression_tree()

        # Predicting values
        Y_pred = root.predict(X_test)

        # Calculating mean squared error and accuracy
        error = mean_square_error(Y_test, Y_pred)
        acc = (1-np.sqrt(error)/np.mean(Y_test))*100

        print(f"Accuracy : {round(acc, 3)}%")

        time.sleep(1)

        if best_tree is None:
            best_tree = root
            best_tree_index = i+1
            best_error = error
            best_acc = acc
            best_X_train, best_Y_train, best_X_test, best_Y_test = X_train, Y_train, X_test, Y_test
            continue

        # If tree shows less error than current best tree, update the best tree
        if error < best_error:
            best_tree = root
            best_tree_index = i+1
            best_error = error
            best_acc = acc
            best_X_train, best_Y_train, best_X_test, best_Y_test = X_train, Y_train, X_test, Y_test

    print("\nBest performing tree : ", best_tree_index)
    print("Best accuracy : ", round(best_acc, 3), "%")
    print("Depth of best tree : ", best_tree.get_depth())

    file.write(f"Accuracy of best tree : {round(best_acc, 3)}%\n")
    file.write(f"Depth of best tree : {best_tree.get_depth()}\n")

    time.sleep(2)

    print("\nStarting pruning process........\n")

    time.sleep(2)

    depths = []
    accuracy = []

    min_error = None
    best_depth = None

    print("Checking accuracy with varying max depth........\n")

    time.sleep(2)

    for i in range(1, 11):

        tree = Node(best_X_train, best_Y_train, max_depth=i)

        tree.build_regression_tree()

        Y_pred = tree.predict(best_X_test)

        error = mean_square_error(best_Y_test, Y_pred)

        acc = (1-np.sqrt(error)/np.mean(best_Y_test))*100

        print("Accuracy at max depth ", i, " : ", round(acc, 3), "%")

        if min_error is None:
            min_error = error
            best_depth = i
        elif min_error > error:
            min_error = error
            best_depth = i

        accuracy.append(acc)
        depths.append(i)

        time.sleep(1)

    time.sleep(1)

    print("\nMaximum accuracy at depth : ", best_depth)
    print("Overfitting at depth : ", best_depth+1)

    file.write(f"Maximum accuracy at depth : {best_depth}\n")
    file.write(f"Overfitting at depth :  {best_depth+1}\n")

    time.sleep(2)

    print("\nFinal Regression Tree :-\n")
    pruned_tree = Node(best_X_train, best_Y_train, max_depth=best_depth)
    pruned_tree.build_regression_tree()

    time.sleep(1)

    pruned_tree.print_tree()

    time.sleep(2)

    print("\nOutput Summary printed in q1_output.txt\n")
    file.close()

    # Plotting accuracy vs depth graph
    plt.plot(depths, accuracy)
    plt.xticks(range(1, 11))
    plt.xlabel("Depth")
    plt.ylabel("Accuracy(%)")
    plt.show()

# Calling main function
if __name__ == "__main__":
    main_func()