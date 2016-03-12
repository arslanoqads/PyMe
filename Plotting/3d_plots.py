from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np

fig=plt.figure()

ax1=fig.add_subplot(111,projection='3d')

x=[4,2,5,6,3,4,5,6,4,5,6,3,4,5]
y=[7,8,9,5,6,3,4,5,2,3,4,2,3,4]
z=[1,2,3,2,4,5,4,6,7,3,4,5,2,3]
w=np.zeros(14)

#ax1.bar3d(x,y,w,0.1,0.5,z,color='g')

ax1.set_xlabel('x axis')
ax1.set_ylabel('y axis')
ax1.set_zlabel('z axis')

a,b,c =axes3d.get_test_data(0.1)

#for wireframe plotting
a=([[1,2,1],[2,5,2]])
b=([[3,5,2],[4,3,2]])
c=([[5,6,2],[6,5,4]])

ax1.plot_wireframe(a,b,c)


#ax1.plot_wireframe(x,y,z)


#ax1.scatter(x,y,z,c='r', marker='o')
#ax1.scatter(z,y,x,c='g', marker='o')
#ax1.scatter(y,z,x,c='y', marker='o')
#ax1.scatter(x,z,y,c='c', marker='o')



    

plt.show()