#!/usr/bin/env python3
from scipy.optimize import minimize


class SimpleLinearRegression:
    """
    Simple linear regression has only single variable (attribute) to learn from
    It takes the form of 
    Y = B0 + B1*X + ε
    Where, ε is error in the training set
    """
    b=[]

    def __init__(self, b_val=None):
        if b_val:
            self.b=b_val
    
    def set_b_val(self, b_val):
        self.b=b_val
    
    def train(self, X, y):
        if not self.b:
            print("initial B values are not set. Calculations could not be progressed further...")
            return
        try:
            sol = minimize(
                self.objective,
                self.b,
                args=(X,y)
            )
            self.b=list(sol.x)
            return True
        except Exception as ex:
            print("Could not train them model: ",ex)
            self.b=[]
            return False

    def test(self, x):
        if not self.b:
            print("Model not trained. First train the model")
            return
        return self.b[0]+(self.b[1]*x)

    def objective(self, b, X, y):
        sum_value=0
        for i in range(len(X)):
            # Squared Sum of Errors
            sum_value+=(y[i]-b[0]-(b[1]*X[i]))**2
        return sum_value
    # private methods
