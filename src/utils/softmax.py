import math

def softmax(x):
    max_x = max(x) # for numerical stability
    exps = [math.exp(i - max_x) for i in x]
    sum_of_exps = sum(exps)
    return [j / sum_of_exps for j in exps]
