from sklearn.linear_model import LogisticRegression
import numpy as np

X = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y = np.array([0, 0, 0, 1, 1, 1])

logReg_model = LogisticRegression()
logReg_model.fit(X, y)  #fit = train = gradient_descent in this case

print("Accuracy on training set:", logReg_model.score(X, y)) 
# .score(X, y) = Coefficient of determination / R^2

print(logReg_model.predict(np.array([3.2,0.9]).reshape(1,-1)))