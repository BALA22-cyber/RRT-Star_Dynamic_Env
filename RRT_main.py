import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import time
import imageio
import io
from utils import *

xs = int(input('Give x - cordinate of starting point: '))
while xs not in range(0,31) :
    print('Give input between 0 and 30')
    xs = int(input('Give x - cordinate of starting point: '))

ys = int(input('Give y - cordinate of starting point: '))
while ys not in range(0,31):
    print('Give input between 0 and 30')
    ys = int(input('Give y - cordinate of starting point: '))

xg = int(input('Give x - cordinate of end point: '))
while xg not in range(0,31):
    print('Give input between 0 and 30')
    xg = int(input('Give x - cordinate of end point: '))

yg = int(input('Give y - cordinate of end point: '))
while yg not in range(0,31):
    print('Give input between 0 and 30')
    yg = int(input('Give y - cordinate of end point: '))

tolerance = float(input('Give goal tolerance (recommended to be between 0.25 to 0.5 for accuracy): ')) #near goal tolerance

start = [xs, ys]
goal = [xg, yg]
goal_cor = [xg, yg,tolerance]

# start = [2,28]
# goal = [28.,3.]
# tolerance = 0.5 
# goal_cor = [goal[0],goal[1],tolerance]
chaos = 0.05
xmin, ymin, xmax, ymax = 0,0,30,30 #grid world borders
obst1 = Obstacle('rect',[5, 5, 2,3], [0,0], chaos*np.eye(2), 1.5, goal_cor = goal_cor)
obst2 = Obstacle('circle',[13,25,2], [0,0], chaos*np.eye(2), 1.5, goal_cor = goal_cor)
obst3 = Obstacle('rect', [24,5,2,6], [0,0], chaos*np.eye(2), 1.5, goal_cor = goal_cor)
obst4 = Obstacle('rect', [13,17,7,1], [0,0], chaos*np.eye(2), 1.5, goal_cor = goal_cor)
obst5 = Obstacle('circle', [10,10,2], [0,0], chaos*np.eye(2), 0.5, goal_cor = goal_cor)
obst6 = Obstacle('rect', [12,12,8,6], [0,0], chaos*np.eye(2), 0, goal_cor = goal_cor)
obstacles = [obst1, obst2, obst3, obst4, obst5, obst6] #list of obstacles
maxNumNodes = np.inf #upper limit on tree size 
eta = 1.0 #max branch length
gamma = 15.0 #param to set for radius of robot
resolution = 0.0001
goalFound = False
plot_and_save_gif = True

# Creating a list to store images at each frame
if plot_and_save_gif:
	images = []


startTime = time.time()

#1. Initialize Tree and growth
print("Initializing RRT* Tree.....")
tree = Tree(start,goal,obstacles,xmin,ymin,xmax,ymax,maxNumNodes,resolution,eta,gamma,tolerance)



#2. Get Solution Path
solPath,solPathID = tree.initialize_Growth(exhaust = True)

# Plot
if plot_and_save_gif:
	im = gen_plot(tree,solPath)
	w, h, _ = im.shape
	# Appending to list of images
	images.append(im)
	


#3. Init movement()-->> update pcurID 
solPath,solPathID,dt = tree.nextSolNode(solPath,solPathID)

#4. Begin replanning loop, while pcur is not goal, do...
while np.linalg.norm(tree.nodes[tree.pcurID,0:2] - tree.goal) > tree.tolerance:
	if plot_and_save_gif:
		im = gen_plot(tree,solPath)
		# Appending to list of images
		images.append(im)
	
	#5. Obstacle Updates
	tree.updateObstacles(dt)

	#6. if solPath breaks, reset tree and replan
	if tree.Collision_detect(solPath):
		tree.reset(inheritCost = True)
		solPath,solPathID = tree.initialize_Growth(exhaust = False)

		if solPath is None:
			print("Algorithm terminated ! \nUnable to connect to Goal even after drawing 100000 new samples this iteration ! \n")
			break

	######## END REPLANNING Block #######
	solPath,solPathID,dt = tree.nextSolNode(solPath,solPathID)

print("Total Run Time: {} secs".format(time.time() - startTime))

if solPath is not None:
	costToGoal, goalID = tree.minGoalID()
	print("Final Total Cost to Goal: {}".format(costToGoal))



if plot_and_save_gif:
	cv.destroyAllWindows()

	# Saving the list of images as a gif
	print("The resulting path is saved as Dynamic_rrt_star.gif")
	imageio.mimsave('Dynamic_rrt_star.gif',images,duration = 0.5)


