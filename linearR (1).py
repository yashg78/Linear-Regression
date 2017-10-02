import numpy as np
import random
import matplotlib.pyplot as plt



def generateData(num, bias, variance):
    x = np.zeros(shape=(num,2))
    y = np.zeros(shape = num)    
    for i in range(0,num):
        x[i][0] = 1
        x[i][1] = i
        y[i] = x[i,1] + bias + (random.uniform(0,1)*(-2))*variance
        
    return x,y
    
    
def gradientDescent(x, y, theta, m, alpha, num_iter):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.ion()
    #plt.plot(train_X, train_Y, 'ro', label='Original data')
    fig.show()
    fig.canvas.draw()
    
    for i in range(1,num_iter+1): 
        hypo = np.dot(x, theta)
        loss = np.sum((hypo - y)**2)
        cost = loss/(2*m)
        if(num_iter) % i == 0:
            print("step {} cost {}".format(i, cost))
        xt = np.dot(x.transpose(),(hypo - y))   
        theta = theta - alpha*(xt/m)  
        yt = np.dot(x,theta)
        ax.clear()
        ax.scatter(x[:,1], y, s = 20)
        plt.plot(x[:,1], yt, 'r')
        fig.canvas.flush_events()
        plt.pause(0.01)
        
        
    plt.show()          
    return theta
    

x, y = generateData(15, 25, 5)
m, n = np.shape(x)

theta = [25,1]
num_iter = 5000
alpha = 0.025
print(theta)
theta = gradientDescent(x, y, theta, m, alpha, num_iter)
yt = np.dot(x,theta)
fig, ax = plt.subplots()
ax.scatter(x[:,1], y, s = 5 , alpha = 0.5)
plt.plot(x[:,1], yt, 'r')
#print(yt)
plt.show()