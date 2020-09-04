from ann import ANN

from random import getrandbits

size = 100

net = ANN(100, 2, [2,2])

def generate_test_case():
    x = getrandbits(size)
    inp = []
    for i in range(size):
        inp.append(x%2)
        x//=2
    return inp

def majority(x):
    sums = [0, 0]
    for i in range(len(x)):
        sums[i%2] += x[i]
    if sums[0] < sums[1]:
        return [0, 1]
    elif sums[1] < sums[0]:
        return [1,0]
    else:
        return [0.5, 0.5]

for i in range(100000):
    inp = generate_test_case()
    net.train(inp, majority(inp))

for i in range(10):
    inp = generate_test_case()
    print(net.classify(inp), majority(inp))
