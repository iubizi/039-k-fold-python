from sklearn.datasets import load_iris
iris = load_iris()

x = iris.data
y = iris.target

print('x.shape =', x.shape)
print('y.shape =', y.shape)
print()



####################
# k-fold
####################

from sklearn.model_selection import KFold
kf = KFold( n_splits=5, shuffle=True, random_state=42 ) # 5-fold
# kf = KFold( n_splits=7, shuffle=True, random_state=42 ) # 7-fold

for train_index, test_index in kf.split(x):
    
    print('len(train) =', len(train_index), end=', ')
    print('len(test) =', len(test_index))
    
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]

'''
x.shape = (150, 4)
y.shape = (150,)

len(train) = 128, len(test) = 22
len(train) = 128, len(test) = 22
len(train) = 128, len(test) = 22
len(train) = 129, len(test) = 21
len(train) = 129, len(test) = 21
len(train) = 129, len(test) = 21
len(train) = 129, len(test) = 21
'''
