import numpy as np 
from PIL import Image
import os
import collect_data

px=8    #image dimension in pexils to use in taining and testing

data,labels=collect_data.collect(px) #this method returns (px,px) images matrix after converting it to gray scale
									 #if the image name contain 'a1' the lables will be 1 else 0

def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    
    return 1/(1+np.exp(-x))


traing_data=np.array(data)
traing_labels=np.array(labels)
np.random.seed(1)

weight1=2*np.random.random((px**2,px**2))-1
weight2=2*np.random.random((px**2,1))-1

for j in range(60000):
	input_layer=traing_data
	hidden_layer=nonlin(np.dot(input_layer,weight1))
	output_layer=nonlin(np.dot(hidden_layer,weight2))

	output_layer_error=traing_labels-output_layer

	if (j%10000)==0:
		print('error rate: ',str(np.mean(np.abs(output_layer_error))))

	output_layer_delta=output_layer_error*nonlin(output_layer,deriv=True)
	hidden_layer_error=output_layer_delta.dot(weight2.T)
	hidden_layer_delta=hidden_layer_error*nonlin(hidden_layer,deriv=True)
	weight2+=hidden_layer.T.dot(output_layer_delta)
	weight1+=input_layer.T.dot(hidden_layer_delta)


def recognize(char):
	k = Image.open(char).convert('L').resize((px, px), Image.ANTIALIAS)
	z=np.array(k).ravel()
	a=nonlin(np.dot(z,weight1))
	b=nonlin(np.dot(a,weight2))
	if abs(1.0-b)<=0.05:
		print (char,"%.7f"%b,'its an a')
	else:
		print (char,"%.7f"%b,'not an a')

for test in os.listdir('test_data'):
	recognize('test_data/'+test)

print('if it near to 1 it means it a small')