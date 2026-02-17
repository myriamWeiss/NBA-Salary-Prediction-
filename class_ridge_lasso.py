from abc import ABC, abstractmethod
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV


class RegressionModel(ABC):

    @abstractmethod
    def fit(self, X, y,alpha):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def set_alpha(self, alpha):
        pass

    @abstractmethod
    def get_coefs(self):
        pass



class RidgeRegression(RegressionModel):
    def __init__(self):
        self.model = Ridge()

    def fit(self, X, y, alpha):
        self.set_alpha(alpha)
        self.model.fit(X, y)
        

    def predict(self, X):
        return self.model.predict(X)


    def get_coefs(self):
        return self.model.coef_
    
    def set_alpha(self, alpha):
        self.model.set_params(alpha=alpha)


class LassoRegression(RegressionModel):
    def __init__(self):
        self.model = Lasso(max_iter=10000)


    def fit(self, X, y, alpha):
        self.set_alpha(alpha)
        self.model.fit(X, y)
        

    def predict(self, X):
        return self.model.predict(X)


    def get_coefs(self):
        return self.model.coef_

    def set_alpha(self, alpha):
        self.model.set_params(alpha=alpha)