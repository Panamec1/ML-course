import math
import random
import numpy as np

class DT:
    def __init__(self, n_features=5, stop=3, H):
        
        self.MAX_HIGH = 23
        
        self.split_feature = [[] for i in range(2 ** self.MAX_HIGH + 1)]
        self.split_value = [[] for i in range(2 ** self.MAX_HIGH + 1)]
        self.prediction = [[] for i in range(2 ** self.MAX_HIGH + 1)]
        self.tree = [[] for i in range(2 ** self.MAX_HIGH + 1)]
        
        self.X = []
        self.y = []
        self.w = []
        self.N = 0
        
        
        self.cutting = True
        self.H = H
        self.vertices = 0
        
        self.n_features = n_features
        self.stop = stop

    def calculate_entropy(self, y):
        entropy = 0
        classes = {}
        for item in y: classes[item] = classes.get(item, 0) + 1
        for cls in classes.keys():
            classes[cls] /= len(y)
            p = classes[cls]
            entropy += p * math.log2(p)
        return -entropy

    def calculate_gain(self, y, y1, y2):
        e1 = self.calculate_entropy(y1)
        e2 = self.calculate_entropy(y2)
        return -len(y1) / len(y) * e1 - len(y2) / len(y) * e2

    def split(self, X, Y, feature, value):
        x1 = []
        x2 = []
        y1 = []
        y2 = []
        for i in range(len(X)):
            if X[i][feature] < value:
                x1.append(X[i])
                y1.append(Y[i])
            else:
                x2.append(X[i])
                y2.append(Y[i])
        return x1, y1, x2, y2
    
    def update_leaf(self, v):
        count = {}
        answer = -math.inf
        for candidate in self.tree[v][1]:
            count[candidate] = count.get(candidate, 0) + 1
            if answer == -math.inf or count[answer] < count[candidate]:
                answer = candidate
        self.prediction[v] = answer

    def create_node(self, v=1, depth=1):
        X, Y = self.tree[v]
        if (depth + 1 <= self.H or not self.cutting) and depth + 1 <= self.MAX_HIGH - 1 \
            and len(Y) > 1 and self.calculate_entropy(Y) > 0.01:
            
            feature_max = np.amax(X, axis=0)
            feature_min = np.amin(X, axis=0)

            best_gain = -math.inf
            best_feature = -math.inf
            best_value = -math.inf
            best_split = ()
            r = random.sample(range(self.N), min(self.N, self.n_features))
            for feature in r:
                if feature_min[feature] == feature_max[feature]: break
                stop = self.stop
                for t in range(stop):
                    value = random.uniform(feature_min[feature], feature_max[feature])
                    x1, y1, x2, y2 = self.split(X, Y, feature, value)
                    if len(x1) == 0 or len(x2) == 0: continue
                    gain = self.calculate_gain(Y, y1, y2)
                    if gain > best_gain:
                        best_gain = gain
                        best_feature = feature
                        best_value = value
                        best_split = (x1, y1, x2, y2)
                        
            if len(best_split) == 0: 
                self.update_leaf(v)
            else:
                x1, y1, x2, y2 = best_split

                self.split_feature[v] = best_feature
                self.split_value[v] = best_value
                self.tree[v * 2] = [x1, y1]
                self.tree[v * 2 + 1] = [x2, y2]
                self.vertices = max(self.vertices, v * 2 + 1)

                self.create_node(v * 2, depth + 1)
                self.create_node(v * 2 + 1, depth + 1)

        else: self.update_leaf(v)

    def fit(self, X, y,w):
        self.X = X
        self.y = y
        self.w = w
        self.N = len(self.X[0])
        if self.n_features is None: self.n_features = math.ceil(math.sqrt(self.N))
        self.tree[1] = [self.X, self.y]
        self.create_node()

    def predict(self, X):
        prediction = []
        for x in X:
            v = 1
            while self.split_feature[v] != []:
                feature = self.split_feature[v]
                value = self.split_value[v]
                if x[feature] < value:
                    v = 2 * v
                else:
                    v = 2 * v + 1
            prediction.append(self.prediction[v]*w[v])
        return np.array(prediction)

    

