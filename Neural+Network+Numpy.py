
# coding: utf-8

# \author{ShengtanWu, YingZhang, Alnazer, JoevoneWells}

# In[67]:

import numpy as np
import random
import matplotlib.pyplot as plt

class hand_write():
    def __init__(self,write,sign):
        self.write = write
        self.sign = sign

class Neural_Network():
    def __init__(self):
        
        self.input_size = 256
        self.hidden_size = 50
        self.output_size = 10
        self.train_size = 1115
        
        self.learning_r = 0.35
        self.hidden = []
        self.output = []
        self.w1 = []
        self.w2 = []
        self.data_set = []
        #w1 size = hidden_size*(inputsize + 1 )
        for i in range(self.hidden_size):
            w = np.random.uniform(-0.1,0.1,self.input_size+1)
            self.w1.append(w)
        #w2 size = output_size *(hidden_size +1)
        for i in range(self.output_size):
            w = np.random.uniform(-0.1,0.1,self.hidden_size+1)
            self.w2.append(w)
        #w1 size = hidden_size*(inputsize + 1)
        self.w1 = np.asarray(self.w1)
        self.w2 = np.asarray(self.w2)
        
        with open("/Users/Wushengtan/Desktop/AI/semeion.data","r") as f:
            for line in f:
                numbers = np.array(line.split()).astype(np.float)
                self.data_set.append(hand_write(numbers[0:256],numbers[256:]))        
        f.close()
        
        self.train_error = []
        self.test_error = []
        
    def sigmoid(self, z):
        return 1/(1+np.exp(-z))
    
    def forward(self, X, target):
        # In this case np.dot(vector,matrix)
        biasX = np.insert(X,0,1)
        
        self.hidden = np.asarray(self.sigmoid(np.dot(self.w1,biasX)))
        
        biasH = np.insert(self.hidden,0,1)
        self.output = np.asarray(self.sigmoid(np.dot(self.w2,biasH)))
        index = np.argmax(self.output)
        output = np.zeros(10)
        output[index] = 1.0
        if np.array_equal(output,target):
            return 0.0
        else:
            return 1.0
    
    def backward(self, X, target):
        self.error2 = self.output*(1-self.output)*(target - self.output)
        #remove the first term
        self.error1 = self.hidden*(1-self.hidden)*np.delete(np.dot(self.error2,self.w2),0)
        biasX = np.insert(X,0,1)
        biasH = np.insert(self.hidden,0,1)
        self.w2 += self.learning_r*np.outer(self.error2,biasH)
        self.w1 += self.learning_r*np.outer(self.error1,biasX)
        
    def training(self):
        epoch = 1
        test_avg_err = 1
        while test_avg_err > 0.001:
            print("Epoch {0}:".format(epoch))
            self.train_set = random.sample(self.data_set,self.train_size)
            self.test_set = list(set(self.data_set)-set(self.train_set))
            train_sum = 0
            test_sum = 0
            for i in range(self.train_size):               
                train_e = self.forward(self.train_set[i].write, self.train_set[i].sign)                
                self.backward(self.train_set[i].write, self.train_set[i].sign)
                train_sum += train_e
                
            #Renew avg_err of training set            
            train_avg_err = round(float(train_sum)/1115,5)
            self.train_error.append(train_avg_err)
            print("training error = {0:5f}%".format(train_avg_err*100))
            

        
            #Testing
            for i in range(len(self.data_set)-self.train_size):
                test_e = self.forward(self.test_set[i].write, self.test_set[i].sign)
                test_sum += test_e
            #Renew avg_err of testing set
            test_avg_err = round(float(test_sum)/478,5)
            self.test_error.append(test_avg_err)
            print("testing error = {0:5f}% \n".format(test_avg_err*100))
          
            epoch +=1
            
        
        # Plot the function    
        self.plot()
       
            
    def test(self):
        print("w1:",self.w1.shape)
        print("w2:",self.w2)
        
    def plot(self):
        plt.gca().set_color_cycle(['red','blue'])
        x = np.arange(len(self.test_error))
        plt.plot(x, self.test_error)
        plt.plot(x, self.train_error)            
        plt.legend(["Train","Test"],loc = "upper left")
        plt.xlabel("Epoch")
        plt.ylabel("Error")
        plt.show()

if __name__== '__main__':
    Neural_Network().training()
    

