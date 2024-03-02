import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('/content/Felicidad_Alcohol.csv',',',
                  usecols=['GDP_PerCapita','HDI','HappinessScore'])

data.head()

A = data[['HDI','HappinessScore']]
A.tail()

matrix = np.array(A.values,'float')
matrix[0:5,:]    #first 5 rows of data

#Assign input and target variable
X = matrix[:,0]
y = matrix[:,1]

X, y

#feature normalization
# input variable divided by maximum value among input values in X
X = X/(np.max(X)) 

X

plt.plot(X,y,'bo')
plt.ylabel('Happiness Score')
plt.xlabel('HDI')
plt.legend(['Happiness Score'])
plt.title('HDI')
plt.grid()
plt.show()


plt.plot(X,y,'bo')
plt.ylabel('Happiness Score')
plt.xlabel('HDI')
plt.legend(['Happiness Score'])
plt.title('HDI')
plt.grid()
plt.show()

def computecost(x,y,theta):
    
    a = 1/(2*m)
    b = np.sum(((x@theta)-y)**2)
    j = (a)*(b)
    return j

#initialising parameter
m = np.size(y)
X = X.reshape([122,1])
x = np.hstack([np.ones_like(X),X])
theta = np.zeros([2,1])
print(theta,'\n',m)

print(computecost(x,y,theta))

def gradient(x,y,theta):
  alpha = 0.00001
  iteration = 2000
  #gradient descend algorithm
  J_history = np.zeros([iteration, 1]);

  for iter in range(0,2000):
        error = (x @ theta) -y
        temp0 = theta[0] - ((alpha/m) * np.sum(error*x[:,0]))
        temp1 = theta[1] - ((alpha/m) * np.sum(error*x[:,1]))
        theta = np.array([temp0,temp1]).reshape(2,1)
        J_history[iter] = (1 / (2*m) ) * (np.sum(((x @ theta)-y)**2))   #compute J value for each iteration 
  return theta, J_history

theta , J = gradient(x,y,theta)
print(theta)

theta , J = gradient(x,y,theta)
print(J)

#plot linear fit for our theta
plt.plot(X,y,'bo')
plt.plot(X,x@theta,'-')
plt.axis([0,1,3,7])
plt.ylabel('Happiness Score')
plt.xlabel('HDI')
plt.legend(['HAPPY','LinearFit'])
plt.title('HDI_Vs_Happiness')
plt.grid()
plt.show()

predict1 = [1,(689/np.max(matrix[:,0]))]

print(predict1)