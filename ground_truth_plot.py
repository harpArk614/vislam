import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.spatial.transform import Rotation as R
from colour import Color
from sklearn.neighbors import NearestNeighbors
import networkx as nx
import os

path=os.getcwd()

file_name="/colmap_traj_sequence.txt"
#file_name=input("Enter file name")
file_name=path+file_name

with open(file_name, 'r') as reader:
    file=reader.readlines()

l=len(file)
pos=np.zeros((l,3))
rotM=np.zeros((l,3,3))
quat=np.zeros((l,4))

for i,line in enumerate(file):
    line_list=line.split()
    pos[i]=np.array(line_list[1:4])
    quat[i]=np.array(line_list[4:])
    r=R.from_quat(quat[i])
    rotM[i]=r.as_matrix().copy()


clf = NearestNeighbors(2).fit(pos)
G = clf.kneighbors_graph()
T = nx.from_scipy_sparse_matrix(G)
paths = [list(nx.dfs_preorder_nodes(T, i)) for i in range(len(pos))]
mindist = np.inf
minidx = 0

for i in range(len(pos)):
    p = paths[i]           # order of nodes
    ordered = pos[p]    # ordered nodes
    # find cost of that order by the sum of euclidean distances between points (i) and (i+1)
    cost = (((ordered[:-1] - ordered[1:])**2).sum(1)).sum()
    if cost < mindist:
        mindist = cost
        minidx = i
opt_order = paths[minidx]

posop=pos[opt_order]





fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot(posop[:, 0], posop[:, 1], posop[:, 2], label='Ground-truth',color='r')
ax.legend()


red=Color("red")
colors = list(red.range_to(Color("green"),len(pos)))
#for i,(p,axis) in enumerate(zip(pos,rotM)):
    #ax.scatter(p[0], p[1], p[2], c=colors[i].rgb, marker='^')
    #ax.scatter(p[0], p[1], p[2], c='y', marker='o')
    #ax.quiver(p[0], p[1], p[2], axis[0][0], axis[0][1], axis[0][2], length=0.08, normalize=True,color='y')
    #ax.quiver(p[0], p[1], p[2], axis[1][0], axis[1][1], axis[1][2], length=0.08, normalize=True,color='b')
    #ax.quiver(p[0], p[1], p[2], axis[2][0], axis[2][1], axis[2][2], length=0.08, normalize=True,color='g')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()


