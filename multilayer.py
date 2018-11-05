import numpy as np

class network:
    def __init__(self,epoch,lr,middle1_num,middle2_num,output_num,a):
        self.epoch = epoch 
        self.lr = lr #learning rate
        self.x_train = x_train
        self.middle1_num = middle1_num #hidden layer 1
        self.middle2_num = middle2_num #hidden layer 2
        self.output_num = output_num #output layer
        self.a = a #sigmoid's constant
        
#add bias to input
    def add_bias(self,x):
         return np.append(x,1)
     
#sigmoid function   
    def sigmoid(self,x):
        return 1 / (1 + np.exp(self.a * (-x)))
    
#derived sigmoid function 
    def sigmoid_d(self,x):
        return x * (1 - x)
    
#ReLU function
    def ReLU(self,x):
        for i in range(len(x)):
            if(x[i][0] > 0):
                pass
        else:
            x[i][0] = 0
        return x
    
#derived ReLU function     
    def ReLU_d(self,x):
            if(x > 0):
                x = 1
            else:
                x = 0  
                
            return x
        
#calculate dot
    def calculate_u(self,w,x):
        dot = np.dot(w,x)
        return dot
    
#update weights    
    def update(self,d_1,d_2,d_3,z_0,z_1,z_2):
        
        for i in range(self.middle1_num):
            for j in range(self.input_num):
                self.w_1[i][j] -= self.lr * d_1[0][i] * z_0[j]
                
        for i in range(self.middle2_num):
            for j in range(self.middle1_num):
                self.w_2[i][j] -= self.lr * d_2[0][i] * z_1[j][0]
                
        for i in range(self.output_num):
            for j in range(self.middle2_num):
                self.w_3[i][j] -= self.lr * d_3[0][i] * z_2[j][0]
                
#prediction from test_data              
    def predict(self,x):
        z_0 = self.add_bias(x)
            
        u_1 = self.calculate_u(self.w_1,z_0.reshape(3,1))
        z_1 = self.ReLU(u_1)
        
        u_2 = self.calculate_u(self.w_2,z_1)
        z_2 = self.sigmoid(u_2)
        
        u_3 = self.calculate_u(self.w_3,z_2)
        z_3 = self.sigmoid(u_3)
        
        return z_3
    
#learning process
    def learning(self,x_train,y_train):
        
        #output1 and teaching data
        X = x_train
        t = y_train
        
        #output of 1st layer
        z_i = [self.add_bias(x) for x in X]
        self.input_num = len(z_i[0])
        
        #initialize weight
        self.w_1 = np.random.uniform(-0.15,0.15,(self.middle1_num,self.input_num))
        self.w_2 = np.random.uniform(-0.15,0.15,(self.middle2_num,self.middle1_num))
        self.w_3 = np.random.uniform(-0.15,0.15,(self.output_num,self.middle2_num))
             
        #forward calclating
        for j in range(self.epoch):
            for z,t in zip(z_i,y_train):
                z_0 = np.array(z) 
                
                u_1 = self.calculate_u(self.w_1,z_0.reshape(3,1))
                z_1 = self.ReLU(u_1)
                   
                u_2 = self.calculate_u(self.w_2,z_1)
                z_2 = self.sigmoid(u_2)

                u_3 = self.calculate_u(self.w_3,z_2)
                z_3 = self.sigmoid(u_3)
                #calculate delta
                d_3 = np.empty((1,self.output_num),float)
                d_3[0][0] = z_3 - t
                
                d_2 = np.empty((1,self.middle2_num),float)
                d_2 = self.w_3 * d_3[0][0] * self.sigmoid_d(u_2).T
                
                d_1 = np.empty((1,self.middle1_num),float)
                for i in range(self.middle1_num):
                    d_1[0][i] =  np.dot(d_2,self.w_2[:,i].T) * self.ReLU_d(u_1[i])

                self.update(d_1,d_2,d_3,z_0,z_1,z_2)
                
#dataset
x_train = [[0,0],[1,1],[1,0],[0,1]]
y_train = [0,0,1,1]

#parameters
epoch = 10000
lr = 0.01
middle1_num = 15
middle2_num = 15
output_num = 1
a = 50

mlp = network(epoch,lr,middle1_num,middle2_num,output_num,a)

#learning process
mlp.learning(x_train,y_train)

#prediction
result = [mlp.predict(x) for x in x_train]
[print(x, ":", p) for p, x in zip(result, x_train)]

