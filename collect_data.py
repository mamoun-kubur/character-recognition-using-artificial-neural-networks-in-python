import numpy as np 
from PIL import Image
import os


def collect(pxile):
	tx=[]
	ty=[]


	di=os.getcwd()
	tr=os.listdir(di+"/"+'training_set')
	px=pxile
	for i in tr:
		if 'a1.' in i:
			img = Image.open(di+"/"+'training_set'+'/'+i).convert('L').resize((px, px), Image.ANTIALIAS)
			img.save('learning'+'/'+i)
			tx.append(np.array(img).ravel())
			ty.append([1])
		else:
			img = Image.open(di+"/"+'training_set'+'/'+i).convert('L').resize((px, px), Image.ANTIALIAS)
			img.save('learning'+'/'+i)
			tx.append(np.array(img).ravel())
			ty.append([0])		
	return (tx,ty)

