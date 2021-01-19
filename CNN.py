# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 09:57:53 2020

@author: Jenni
"""

import numpy as np
from matplotlib import image
from matplotlib import pyplot
import os
import random
from PIL import Image


class CNN:
    def __init__(self, imgPath, hiddenLayer):
        self.imgPath = imgPath
        self.hiddenLayer = hiddenLayer
        self.train = []
        self.imgNames = []
        self.imgLabels = []
        self.uLabels = []
        self.targetList = []
        self.uTargets = {}
        self.test = []

    def setTrain(self, Array):
        """Set self.train to a new array of arrays. """
        self.train = Array
        print("trainindata set")
    
    def printImage(self, img):
        """
        print an image
        input: image object
        """
        pyplot.imshow(img)
        pyplot.show()

    def arrayToImg(self, array):
        """ 
        Converts a 2 x 2 array to an image for printing.
        returns: image object 
        """
        img = Image.fromarray(array)
        imgC = img.convert("L")
        return (imgC)
    
    def normalizeTrainArray(self):
        """
        Normalizes the training data to 0-1
        Not currently in use, isn't working like I wanted it to.
        """
        trainNorm = []
        for L in self.train:
            norm = np.linalg.norm(L)
            trainNorm.append(L/norm)
        self.train = trainNorm
        
    def normalizeTestArray(self):
        """
        Normalizes the test data to 0-1
        Not currently in use, isn't working like I wanted it to.
        """
        trainNorm = []
        for L in self.test:
            norm = np.linalg.norm(L)
            trainNorm.append(L/norm)
        self.test = trainNorm
        
    def printTrippel(self,imgA, imgB, imgC):
        """ 
        Prints three images beside each other. 
        input: three images to print. 
        """
        fig = pyplot.figure()
        ax1 = fig.add_subplot(1,3,1)
        ax1.imshow(imgA)
        ax2 = fig.add_subplot(1,3,2)
        ax2.imshow(imgB)
        ax3 = fig.add_subplot(1,3,3)
        ax3.imshow(imgC)
        pyplot.show()
        
    def listImageArrays(self):
        """
        Sets self.imgList to [], loops through the image-folder 
        self.imgPath and appends their arrays  to the self.imgList. 
        """
        self.imgList = []
        #Loop through images. 
        for L in os.listdir(self.imgPath):
            #List images
            img = image.imread(os.path.join(self.imgPath, L))
            self.imgList.append(img)
            

    def listTestArrays(self, imgTestPath):
        """
        Sets self.imgTestList to [], loops through the image-folder 
        sef.imgTestPath and appends their arrays  to the self.imgList. 
        Input: path to the folder with images for testing the CNN.
        """
        #Sets self.imgTestPath
        self.imgTestPath = imgTestPath
        self.imgTestList = []
        #Loop through images. 
        for L in os.listdir(self.imgTestPath):
            #List images
            img = image.imread(os.path.join(self.imgTestPath, L))
            self.imgTestList.append(img)
        
        
            
    def arrayConvolution(self, imgArr, convType):
        """ 
        Convolutes a 2D array. Has multiple filters to choose from. 
        Input: 
        imgArr = input array
        convType = the type of filter to use 
        return: convoluted array that is smaller than the original 
        by two rows and two columns. 
        """
        if convType == "V":
            #Vertical filter
            Filter = ([1, 0, -1, 1, 0, -1, 1, 0, -1])
    
        elif convType == "H":
            #Horizontal filter
            Filter =  ([1, 1, 1,0, 0, 0, -1, -1, -1])
    
        elif convType == "D":
            # Diagonal 1 filter
            Filter = ([1, 2, 0, 2, 0, -2, 0, -2, -1])
            
        elif convType == "V2":
              # Vertical sobel filter
            Filter = ([-1, 0, 1, -2, 0, 2, -1, 0, 1])
        elif convType =="H2":
            # Horizontal sobel filter
            Filter = ([1,2,1,0,0,0,-1,-2,-1])
            
        tempImg = []
        #r = rows, c = columns
        for r in range(0, len(imgArr)-2):
            tempOut = []
            for c in range(0, len(imgArr[0][:-2])):
                # Create temp array with values from imgArr
                # 1D array corresponding to a 3x3 moving window 
                Temp = np.asarray([imgArr[r][c], imgArr[r][c+1], imgArr[r][c+2], imgArr[r+1][c],imgArr[r+1][c+1], imgArr[r+1][c+2],imgArr[r+2][c], imgArr[r+2][c+1], imgArr[r+2][c+2]])
               #multiply the filter chosed with the Temp array. 
               #Add the result to the output array, row by row, column by column. 
                tempOut.append(np.dot(Temp, Filter))
            tempImg.append(tempOut)

        return(tempImg)

    
    def maxPooling(self, imgArray): 
        """ 
        Compresses an array by maxpooling. 
        Input: 2D array with numbers. 
        return: new 2D array where every cell has the max value of 4 
        cells in the original. 
        """
        outPool = []
        #r = rows, c = columns
        #loop while there are an even number of cells left in r or c. 
        r = 0
        while r < int(len(imgArray)/2)*2:
            tempRow = []
            c = 0
            while c < int(len(imgArray[0])/2)*2:
                # print([imgArray[r][c], imgArray[r][c+1], imgArray[r+1][c], imgArray[r+1][c+1]])
                # Create 1D array corresponding to 2x2 moving window.
                # Select max value and append to output array, row by row, 
                #column by column. 
                tempRow.append(max([imgArray[r][c], imgArray[r][c+1], imgArray[r+1][c], imgArray[r+1][c+1]]))
                c += 2
            outPool.append(tempRow)
            r += 2
            
        return(np.asarray(outPool))
    
    
    def createNNInputArray(self, List):
        """
        Takes in a list of 2D arrays and creates one 1D array. 
        Saves the result to self.train
        Input: List of 2D arrays
        """
        temp = []
        self.train = []
        for imgArray in List:
            for row in imgArray:
                for elem in row:
                    temp.append(elem)
        self.train.append(temp)
        
    def createTestArray(self, List):
        """
        Takes in a list of 2D arrays and creates one 1D array
        Saves the result to self.test
        Input: List of 2D arrays
        return: 1D array with the values from List
        """
        temp = []
        self.test = []
        for L in List:
            for a in L:
                for b in a:
                    temp.append(b)
        self.test.append(temp)
        
    def labelsAndNames(self):
        """
        lists image names and labels from the images in the folder imgPath
        sets three variables:
        self. imgNames = list of names of the images
        self.imgLabels = list of labels of the images
        self.uLabels = list of unique lables
        """
        self.imgNames = []
        self.imgLabels = []
        self.uLabels = []
        for L in os.listdir(self.imgPath):
          #Lists names
          self.imgNames.append(L)
          #Create ans list imgLabelss
          imgLabelsTemp = ''.join([i for i in L.split(".")[0] if not i.isdigit()])
          self.imgLabels.append(imgLabelsTemp)
          if imgLabelsTemp not in self.uLabels:
              self.uLabels.append(imgLabelsTemp)
             
    def testLabelsAndNames(self):
        """
        lists image names and labels from the images in the folder self.imgTestPath
        sets three variables:
        self. testNames = list of names of the images
        self.testLabels = list of labels of the images
        """
        self.testNames = []
        self.testLabels = []
        
        for L in os.listdir(self.imgTestPath):
          #Lists names
          self.testNames.append(L)
          #Create ans list imgLabelss
          imgLabelsTemp = ''.join([i for i in L.split(".")[0] if not i.isdigit()])
          self.testLabels.append(imgLabelsTemp)

             
    def createTargetList(self):
        """ 
        Creates a list of 1D target matrixes for the training data. 
        Image with index 1 has target matrix with index 1.
        sets self.targetList to all targets.
        Self.uTargets is a dictionary with unique lables as keys and unique 
        target matrixes as values. 
        """
        noU = len(self.uLabels)
        targetList = []
        self.uTargets = {}
       
        #Create identity matrix with the size of unique lables. 
        # Each row is a target matrix for a unique lable
        idMatrix = np.identity(noU, int)
        
        #Fill uTarget dictionary
        for i in range(0, noU):
            self.uTargets[self.uLabels[i]] = idMatrix[i]
        
        #For each lable in imgLables, find the unique target matrix that 
        #matches that label. Add the matrix to targetList
        for i in self.imgLabels:
            targetList.append(self.uTargets[i])
        #Set self.targetList to targetList.  
        self.targetList = np.asarray(targetList)
        
    def createWeights(self, rows, columns):
        """
        Create weight array
        input:
        rows = number of rows in the array
        columns = number of columns in the array
        return: rows x columns array with random numbers between 0 - 1
        """
        ws = []
        r= 0
        while r < rows:
            c = 0
            rowList = []
            while c < columns:
                rowList.append(round(random.random(), 2))#Append random number
                c += 1
            ws.append(rowList)
            r += 1
            
        return(np.asarray(ws))
            
    def prepAllWeights(self):
        """ 
        Preps weights for both layers of nodes
        """
        outLayer = len(self.targetList[0])  #length of target matrix
        outTrain = len(self.train[0])        #length of trainingdata array
        
        self.w = self.createWeights(outTrain, self.hiddenLayer)
        self.wh = self.createWeights(self.hiddenLayer, outLayer)

    #****************************************************************************
#Run training
        
    def forward_pass(self, weight, array):
         """ 
        Calculates the dot-product of array and weight and uses a sigmoid 
        function as activation function. 
        input:
        weight = array with float numbers
        array = array with numbers. 
        return:
        net = the dot-product
        out = the complete calculation incl. sigmoid function. 
        """
         net=np.dot(array,weight) # dot-product 
         out=np.divide(1,1+np.exp(np.multiply(-1,net)))  # sigmoid = 1./(1+np.exp(-net))
         return net, out
     
    def backward_init(self, target, out1, out2):
        """ 
        Calculates first step in backpropagation
        input:
        target = the target matrix
        out1 = output from first round of forward propagation
        out2 = output from second round of forward propagation
        return:
        updated wh
        """
       #Derivative Etot over out2
        dEtot_dout2=-(target-out2)
        #Derivative out2 over net
        dout2_dnet=out2 * (1-out2)
        # Derivative net over w
        dnet_dw=out1
        
        #rows = len(out1), cols = len(out2)
        diff =[[0 for x in range(len(out2))] for y in range(len(out1))]       
        #Calculate temp backwards weights
        for i in range(0,len(out2)):
            for j in range(0,len(out1)):
                diff[j][i]=dEtot_dout2[i]*dout2_dnet[i]*dnet_dw[j]
        # Calc new wh from the diff. 
        self.wh = self.wh - 0.5*(np.asarray(diff)) 
        return self.wh
    
    def backward_hidden(self,target,out1,out2, imgArray):
        """ 
        Calculates second step in backpropagation
        input:
        target = the target matrix
        out1 = output from first round of forward propagation
        out2 = output from second round of forward propagation
        imgArray = training data for an image. 
        return:
        updated w
        """
        #Dericative of out2 over net1
        dout_dnet=np.multiply(out2,(np.subtract(1,out2))) #out2.*(1-out2)
        #Derivative tot Error over out2
        dEtot_dout_o=-1*(target-out2)
        #Derivative out1 over net1
        dout1_o_dnet=np.multiply(out1,(np.subtract(1,out1))) #activation.*(1-activation)
        #Derivative Etot over net1
        dEtot_dnet=np.multiply(dEtot_dout_o,dout_dnet)
        # Derivative net over out1
        dnet_dout1=self.wh
        #Derivative net over w
        dnet_dwi=imgArray
        
        #Derivative Etot over w
        ii=0
        diff=np.zeros(len(dout1_o_dnet)*len(dnet_dwi))
        for j in range(0,len(dout1_o_dnet)):
            for k in range(0,len(dnet_dwi)):
                diff[ii]=np.sum(np.multiply(dEtot_dnet,dnet_dout1[j][:])*dout1_o_dnet[j]*dnet_dwi[k])
                ii += 1
       
        # Calc new w  
        ii=0
        weights2=[[0 for x in range(len(self.w[0,:]))] for y in range( len(self.w[:,0]))]
        
        for j in range(0,len(self.w[0,:])): 
            for k in range(0,len(self.w[:,0])): 
                weights2[k][j]=self.w[k][j]-0.5*diff[ii] 
                ii += 1
        return np.asarray(weights2)
    
    def trainCNN(self, noLoops):
        """ 
        Run training. 
        input: number of iterations to run. 
        """
        #Variables for trying to manage when the error stops evolving
        countErr = 0
        totError1 = 0
        totError = 1
        
        #For each iteration, run each dataset in train.
        for i in range(0, noLoops):
            for index, imgArray in enumerate(self.train):
                
                #Find target
                target = self.targetList[index]
              
                #Run forward propagation
                [net1, out1] = self.forward_pass(self.w, imgArray)
                [net2, out2] = self.forward_pass(self.wh, out1)
                
                #Calc totError and check if it is the same as the 
                #last round. If id doesn't change for 3 rounds, set random
                #weight to 0 in w and wh
                totError1 = totError
                totError = np.sum(0.5*(target-out2)**2)
                if totError == totError1:
                    countErr += 1
                    if countErr == 3:
                        self.w[random.randrange(len(self.w))][random.randrange(len(self.w[0]))] = 0
                        self.wh[random.randrange(len(self.wh))][random.randrange(len(self.wh[0]))] = 0
                else:
                    countError = 0
                
                
               #Run backpropagation
                self.wh = self.backward_init(target, out1, out2)
                self.w = self.backward_hidden(target,out1,out2, imgArray)
                
            if i%10 == 0:
                print(i, " ",   totError)
                #print(out2)
        
        print ("\nTraining complete.\n Error: ", totError)
        print(out2) #Final result of training. 


#Testing*******************************************************************

    def runANNTest(self):
        """
        Test if the trained model can classify data correctly. 
        """
        #Classify test images.
        for index, array in enumerate(self.test):
            Softmax = []
            
            #Run forward propagation
            temp = self.forward_pass(self.w, array)
            Result = self.forward_pass(self.wh, temp[1])[1]
            
            # Calculate the probability of the image being in each 
            # class (targets) by running a Softmax calculation. 
            Sum = np.sum(np.exp(Result))
            for R in Result:
                Softmax.append(round(np.exp(R)/Sum, 2))
            
            #Find the target matrix with the value 1 at the same index
            #as the largest probability in the Softmax matrix. 
            maxIndex = Softmax.index(np.max(Softmax))
            for i in self.uTargets:
                if self.uTargets[i][maxIndex] == 1:
                    Class = i
            
            #Print the image to make it easy to see it the result is correct. 
            img = image.imread(os.path.join(self.imgTestPath, self.testNames[index]))
            self.printImage(img)
            
            #Print image name, predicted class, probability and softmax. 
            print(self.testNames[index], "\n")
            print("Bilden föreställer klass ", Class, " med ", round(Softmax[maxIndex]*100, 2), " % säkerhet")
            print("Alla sannolikheter: ", Softmax, "\n")
    