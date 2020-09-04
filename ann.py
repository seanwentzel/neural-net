from math import e
import random

def logistic(x):
    return 1/(1+e**(-1.*x))

def dlogistic(x):
    return logistic(x)*logistic(-x)

def mul(mat, v):
    return [sum(x*y for x,y in zip(row,v)) for row in mat]

def trans(mat):
    return map(list, zip(*mat))

class ANN(object):

    def __init__(self, i, o, layers, learning_rate = 3):
        self.dim_i = i
        self.dim_o = o
        self.dim_l = len(layers)
        self.learning_rate = learning_rate
        self.weights = []
        for l_i, l_o in zip([i]+layers,layers+[o]):
            self.weights.append([[random.uniform(-1,1) for _ in range(l_i)] for _ in range(l_o)])

    def __propagate(self, i, expected=None):
        a=[i]
        z=[]
        dz = []
        for weight in self.weights:
            z.append(mul(weight,a[-1]))
            dz.append([dlogistic(x) for x in z[-1]])
            a.append([logistic(x) for x in z[-1]])
        if expected == None:
            return a[-1]
        dels = [None for i in range(len(z))]
        diffs = [x-y for x,y in zip(a[-1],expected)]
        dels[-1] = [x*y for x,y in zip(dz[-1],diffs)]
        for la in range(len(dels)-2,-1,-1):
            diffs = mul(trans(self.weights[la+1]), dels[la+1])
            dels[la] = [x*y for x,y in zip(dz[la],diffs)]
        for la in range(len(self.weights)):
            weight = self.weights[la]
            for y in range(len(weight)):
                for x in range(len(weight[0])):
                    weight[y][x] -= self.learning_rate * a[la][x] * dels[la][y]
        #print(self.weights)
        return a[-1]

    def classify(self, i):
        return self.__propagate(i)

    def train(self, i, expected):
        return self.__propagate(i, expected)

