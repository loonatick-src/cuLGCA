import matplotlib.pyplot as plt
import numpy as np

f= open('graph_inp.txt')
file=f.read()
data=file.split('[')
#print(data)
x,y = np.meshgrid(np.arange(0, 32, 1), np.arange(31, -1, -1))
u=[
1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	
1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	
1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	
1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	
1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	
1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	
1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	
1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	
1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	
1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	
1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	
1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	
1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	
1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	
1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	
1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	
1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	
1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	
1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	
1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	
1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	
1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	
1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	
1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	
1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	
1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	
1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	
1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	
1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	
1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	
1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	
1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	1.5, 	
]
u=np.array(u)
print(u.shape)
u=np.reshape(u,(32,32))
print(u)
v=[
0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	
0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	
0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	
0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	
0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	
0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	
0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	
0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	
0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	
0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	
0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	
0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	
0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	
0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	
0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	
0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	
0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	
0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	
0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	
0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	
0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	
0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	
0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	
0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	
0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	
0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	
0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	
0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	
0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	
0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	
0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	
0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	0.866025, 	
]
v=np.array(v)
print(v.shape)
v=np.reshape(v,(32,32))
print(v)
print(x,'\n')
print(y)
fig, ax = plt.subplots()
q = ax.quiver(x,y,u,v)
plt.show()