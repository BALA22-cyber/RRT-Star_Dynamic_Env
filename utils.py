import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import time
import imageio
import io



def samplepoints(xmin, ymin, xmax, ymax):
	return np.array([np.random.uniform(xmin, xmax), np.random.uniform(ymin, ymax) ])

def steer(eta,qnear, qrandom):
	dist = np.linalg.norm(qrandom - qnear)
	branchLength = min(eta,dist)
	qdir = branchLength * (qrandom - qnear)/dist
	return qnear + qdir


def gen_edge(a, b, ax, color = 'black', width = 1):
    path = Path([(a[0], a[1]), (b[0], b[1])], [Path.MOVETO, Path.LINETO])
    pathpatch = patches.PathPatch(path, facecolor='white', edgecolor= color, linewidth = width)
    ax.add_patch(pathpatch)
def gen_tree(mat, ax, color = 'black'):
	for i in range(np.shape(mat)[0]):
		if mat[i, 3] != -1:
			parentID = int(mat[i, 3])
			gen_edge(mat[i, 0:2], mat[parentID, 0:2], ax, color)
   
def gen_Shape(patch, ax):
	ax.add_patch(patch)
 
def gen_Path(path, ax, color = 'green'):
	for i in range(np.shape(path)[0]-1):
		gen_edge(path[i], path[i+1], ax, color, 2)
  
def plot_Envir(tree, ax):
	for obs in tree.obstacles:
		gen_Shape(obs.toPatch('black'), ax)
	#start and goal
	Goal = patches.Circle((tree.goal[0], tree.goal[1]), 0.5, facecolor = 'blue' )
	Start = patches.Circle((tree.start[0], tree.start[1]), 0.5, facecolor = 'red' )
	gen_Shape(Goal, ax)
	gen_Shape(Start, ax)

def saveimage(fig, dpi= 180):
	buf = io.BytesIO()
	fig.savefig(buf, format = "png", dpi = dpi)
	buf.seek(0)
	img_arr = np.frombuffer(buf.getvalue(), dtype = np.uint8)
	buf.close()
	img = cv.imdecode(img_arr,1)
	return img

def gen_plot(tree,solPath):
	fig,ax = plt.subplots()
	plt.ylim((0,30))
	plt.xlim((0,30))
	ax.set_aspect('equal',adjustable='box')
	pcur = tree.nodes[tree.pcurID,0:2]
	gen_Shape(patches.Circle((pcur[0],pcur[1]),0.5,facecolor = 'grey'),ax)
	gen_tree(tree.nodes,ax,'red')
	gen_Path(solPath,ax)
	plot_Envir(tree,ax)
	im = saveimage(fig)
	cv.imshow('frame',im)
	cv.waitKey(100)
	# Converting from BGR (OpenCV representation) to RGB (ImageIO representation)
	im = cv.cvtColor(im,cv.COLOR_BGR2RGB)
	plt.close()

	return im


class Tree(object):
	def __init__(self,start,goal,obstacles,xmin,ymin,xmax,ymax,maxNumNodes = 1000,res = 0.0001,eta = 1.,gamma = 20.,tolerance = 0.5):
		self.nodes = np.array([start[0],start[1],0,-1]).reshape(1,4)
		#4th column of self.nodes == parentID of root node is None
		#3rd column of self.nodes == costs to get to each node from root		
		self.obstacles = obstacles # a list of Obstacle Objects
		self.goalIDs = np.array([]).astype(int) # list of near-goal nodeIDs
		self.update_q = [] # for cost propagation
		self.resolution = res # Resolution for obstacle check along an edge
		self.orphanedTree = np.array([0,0,0,0]).reshape(1,4)
		self.separatePathID = np.array([]) # IDs along path to goal in the orphaned tree
		self.pcurID = 0 # ID of current node (initialized to rootID)
		self.xmin, self.ymin, self.xmax, self.ymax = xmin, ymin, xmax, ymax
		self.start = start
		self.goal = goal
		self.eta = eta
		print("Eta: ",eta)
		self.gamma = gamma
		self.temp_tree = np.array([0,0,0,-1]).reshape(1,4)
		self.tolerance = tolerance
		self.maxNumNodes = maxNumNodes

	
	def edge(self, parentID, child, cost):
		if parentID < 0 or parentID > np.shape(self.nodes)[0]-1:
			print("INVALID Parent ID when adding a new edge")
			return
		new_node = np.array([[child[0],child[1],float(cost),int(parentID)]])
		self.nodes = np.append(self.nodes,new_node,axis = 0)
		return len(self.nodes)-1 #return child node's ID

	def NearestNode(self, sample):
		# Returns nearest neighbour to the sample from the nodes of the self
		temp = self.nodes[:,0:2] - sample
		distance = np.linalg.norm(temp,axis = 1)
		nearest_nodeID = np.argmin(distance)
		nearest_node = self.nodes[nearest_nodeID,0:2]
		return nearest_node, nearest_nodeID

	def retrace(self, nodeID, rootID = -1):
		#returns nodeID sequence from the root node to the given node
		path_ID = np.array([nodeID])
		parentID = int(self.nodes[nodeID, 3])
		while parentID != rootID:
			path_ID = np.append(path_ID, [parentID])
			parentID = int(self.nodes[parentID,3])
		if rootID != -1:
			path_ID = np.append(path_ID, [rootID])	
		return np.flipud(path_ID)

	def collisionFree(self, node):
		#node contains either the x-y coord of the robot or the x-y coords along an edge
		for obs in self.obstacles:
			if not obs.isCollisionFree(node):
				return False
		return True

	def isBranch(self, x1, x2, branchLength):
		#returns a boolean whether or not a branch is feasible
		num_points = int(branchLength / self.resolution)
		x = np.linspace(x1,x2,num_points)

		return self.collisionFree(x)
	
	def addGoalID(self, goalID):
		self.goalIDs = np.append(self.goalIDs, int(goalID))
	
	def updateObstacles(self,dt):
		for obst in self.obstacles:
			obst.moveObstacle(self.nodes[self.pcurID],dt)
	
	
	def NeighbourID(self, new_node, radius):
		#returns nodeIDs of neighbors within robot
		temp = self.nodes[:,0:2] - new_node
		distances = np.linalg.norm(temp,axis = 1)
		distances = np.around(distances,decimals = 4)
		neighbour_indices = np.argwhere(distances <= radius)
		return distances,neighbour_indices

	
	def RemoveNode(self, xnewID, goal, goalFound):
		#1. find childless nodes 
		parentIDs = self.nodes[:, 3].copy().tolist()
		parentIDs = set(parentIDs)
		nodeIDs = set(np.arange(np.shape(self.nodes)[0]))
		childlessIDs = nodeIDs - parentIDs

		#2. Get the tail node of best path towards goal

		if goalFound:
			#returns best near goal nodeID and its cost
			minCostToGoal, goalID = self.minGoalID()
			bestLastNodeID = goalID
		#else get best path to node closest to goal
		else:
			nearestToGoal, ntgID = self.getNearest(goal)
			bestLastNodeID = ntgID

		#3. Exclude xnew and bestLastNode from childless list. Then draw
		childlessIDs = list(childlessIDs - {bestLastNodeID, xnewID})
		if len(childlessIDs) < 1:
			return
		xremoveID = np.random.choice(childlessIDs)
		#4. Remove
		self.nodes = np.delete(self.nodes, xremoveID, axis = 0)
		if xremoveID in self.goalIDs:
			self.goalIDs = np.delete(self.goalIDs,np.argwhere(self.goalIDs == xremoveID))
		#adjust parentIDs
		parents = self.nodes[:, 3]
		self.nodes[np.where(parents > xremoveID), 3]= self.nodes[np.where(parents > xremoveID), 3]-1
		#adjust goalIDs		
		self.goalIDs[np.where(self.goalIDs > xremoveID)]= self.goalIDs[np.where(self.goalIDs > xremoveID)]-1
		# print("REMOVED CHILDLESS NODE: {}".format(self.nodes[xremoveID, :]))


	def minGoalID(self):
		self.goalIDs = self.goalIDs.astype(int)
		costsToGoal = self.nodes[self.goalIDs, 2]
		minCostID = np.argmin(costsToGoal)
		return costsToGoal[minCostID], self.goalIDs[minCostID]

	def chooseParent(self,new_node,neighbour_indices,distances):
		#choosing Best Parent
		nayID = neighbour_indices[0]
		parent_index = nayID
		branchCost = distances[nayID]
		costToNay = self.nodes[nayID,2]	
		min_cost = branchCost + costToNay

		for nayID in neighbour_indices:
			branchCost = distances[nayID]
			costToNay = self.nodes[nayID,2]	
			cost = branchCost + costToNay
			if cost < min_cost and self.isBranch(self.nodes[nayID, 0:2], new_node, branchCost):
				min_cost = cost
				parent_index = nayID

		return parent_index, min_cost


	# Rewiring the tree nodes within the robot after a new node has been added to the tree. 
	# The new node becomes the parent of the rewired nodes
	def rewire(self,new_nodeID,neighbour_indices,distances):
		distance_to_neighbours = distances[neighbour_indices] #branch costs to neighbor
		new_costs = distance_to_neighbours + self.nodes[new_nodeID,2]
		for i in range(neighbour_indices.shape[0]):
			if  new_costs[i] < self.nodes[neighbour_indices[i],2]:
				self.nodes[neighbour_indices[i],3] = self.nodes.shape[0] - 1 #change parent
				self.nodes[neighbour_indices[i],2] = new_costs[i] #change cost
				children_indices = np.argwhere(self.nodes[:,3] == neighbour_indices[i]) 
				children_indices = list(children_indices)
				self.update_q.extend(children_indices)
				#COST PROPAGATION ####
				while len(self.update_q) != 0:
					child_index = int(self.update_q.pop(0))
					parent_index = int(self.nodes[child_index,3])
					dist = self.nodes[child_index,0:2] - self.nodes[parent_index,0:2]
					self.nodes[child_index,2] = self.nodes[parent_index,2] + np.linalg.norm(dist) #update child's cost
					next_indices = np.argwhere(self.nodes[:,3] == child_index)
					next_indices = list(next_indices)
					self.update_q.extend(next_indices)

	
	def initialize_Growth(self, exhaust = False, N = 5000):
		#exhaust: if true, finish all N iterations before returning solPath
		#initial tree growth. Returns solution path and its ID sequence
		goalFound = False
		num_iterations = 0
		max_iterations = 20


		def iterate(goalFound):
			if num_iterations >= max_iterations:
				return None,None,goalFound
			for i in range(N):
				#2. Sample
				qrandom = samplepoints(self.xmin, self.ymin, self.xmax, self.ymax)
				#3. Find nearest node to qrandom
				qnear, qnearID = self.NearestNode(qrandom)
				qnew = steer(self.eta,qnear,qrandom)
			 
				if self.isBranch(qnear, qnew, np.linalg.norm(qnear-qnew)):
					#4. Find nearest neighbors within robot
					n = np.shape(self.nodes)[0] #number of nodes in self
					radius = min(self.eta, self.gamma*np.sqrt(np.log(n)/n))
					distances, NNids = self.NeighbourID(qnew, radius) 
					#distances are branch costs from every node to qnew
					
					#5. Choose qnew's best parent and insert qnew
					naysID = np.append(np.array([qnearID]),NNids)
					qparentID, qnewCost = self.chooseParent(qnew, naysID, distances)	
					qnewID = self.edge(int(qparentID), qnew, qnewCost)	
					
					#6. If qnew is near goal, store its id
					if np.linalg.norm(qnew - self.goal) < self.tolerance:
						goalFound = True
						# Append qnewID(goalID) to self.goalIDs list		
						self.addGoalID(int(qnewID))
					#7. Rewire within the robot's vicinity
					self.rewire(qnewID,naysID,distances)

				if not exhaust:
					if goalFound:
						costToGoal, goalID = self.minGoalID()
						solpath_ID = self.retrace(goalID)
						return self.nodes[solpath_ID, 0:2], solpath_ID, goalFound

			if goalFound:
				costToGoal, goalID = self.minGoalID()
				solpath_ID = self.retrace(goalID)
				return self.nodes[solpath_ID, 0:2], solpath_ID, goalFound

			else:
				return -1,-1,goalFound

		while not goalFound:
			solPath, solPathID, goalFound = iterate(goalFound)
			if solPath is None:
				return None,None
		return solPath, solPathID 
	
	def Collision_detect(self,solpath):
		path_list = []
		
		num_points = 10000
		path_list = np.linspace(solpath[0:-1],solpath[1:],num_points)
		path_list = path_list.reshape(-1,2)

		# Returns True if a collision is detected
		return np.logical_not(self.collisionFree(path_list))

	def rerootAtID(self,newrootID,tree,pathIDs=None,goalIDs=None):
		# check if root
		papaIDs = tree[:,-1]
		rootID = np.where(papaIDs==-1)[0][0]
		if newrootID == rootID:
			raise ValueError('This is already the root node, dummy')
		# save copy of tree as self.temp_tree to allow recursion
		self.temp_tree = np.copy(tree)
		# recursively strip lineage starting with root node
		self.recursivelyStrip(newrootID,papaIDs,rootID)
		strippedToNodeID = np.cumsum(np.isnan(self.temp_tree[:,-1]))
		for ID in range(self.temp_tree.shape[0]):
			parentID = self.temp_tree[ID,-1]
			if not np.isnan(ID) and not np.isnan(parentID):
				self.temp_tree[ID,-1] -= strippedToNodeID[int(parentID)]
		self.temp_tree[newrootID,-1] = -1
		# delete nodes before newroot (where parentID==None)
		removeIDs = np.argwhere(np.isnan(self.temp_tree[:,-1]))
		self.temp_tree = np.delete(self.temp_tree,removeIDs,axis=0)
		out_tree = self.temp_tree
		self.temp_tree = np.array([0,0,0,-1]).reshape(1,4)
		# shift subpathIDs
		returnpath = False
		if not pathIDs is None:
			returnpath = True
			q = np.array([np.argwhere(pathIDs == i)[0][0] if i in pathIDs else np.nan for i in removeIDs])
			q = q[ ~np.isnan(q)]
			pathIDs = np.delete(pathIDs, q, axis = 0)
			sub_pathIDs = [int(ID)-strippedToNodeID[int(ID)] for ID in pathIDs]
			try:
				sub_pathIDs = np.array(sub_pathIDs)[np.greater_equal(sub_pathIDs,0,dtype=int)]
			except:
				sub_pathIDs = np.empty(0)
		# shift remaining subset of goalIDs
		returngoal = False
		if not goalIDs is None:
			returngoal = True
			q = np.array([np.argwhere(goalIDs == i)[0][0] if i in goalIDs else np.nan for i in removeIDs])
			q = q[ ~np.isnan(q)]
			goalIDs =  np.delete(goalIDs,q, axis = 0)
			rem_goalIDs = [int(ID)-strippedToNodeID[int(ID)] for ID in goalIDs]
			rem_goalIDs = np.array(rem_goalIDs)
		if returnpath and returngoal:
			return out_tree,sub_pathIDs,rem_goalIDs
		if returnpath:
			return out_tree,sub_pathIDs
		if returngoal:
			return out_tree,rem_goalIDs
		return out_tree
	
	def recursivelyStrip(self,newrootID,parentIDs,nodeID):
		# Strip this node
		self.temp_tree[nodeID,-1] = None
		# Find all children
		childrenIDs = np.argwhere(parentIDs==nodeID).flatten()
		# for each child { if not newroot { continue recursion } }
		if not childrenIDs.shape[0]==0:
			childrenIDs = childrenIDs.tolist()
			for childID in childrenIDs:
				if not childID == newrootID:
					self.recursivelyStrip(newrootID,parentIDs,nodeID=childID)
	
	def selectBranch(self,solnpathIDs):
		#return the adjusted solpathID(shorter and ID-correct), passs solpathID to validPath()
		self.nodes,subpathIDs,self.goalIDs = self.rerootAtID(self.pcurID,tree = self.nodes,pathIDs = solnpathIDs,goalIDs = self.goalIDs)
		return subpathIDs

	def destroyLineage(self, ancestorIDs, tree):
		#returns new tree with lineage(s) rooted at ancestorID(s) removed 

		#1. Nan-mark nodes to be removed
		rootID = np.argwhere(tree[:, -1] == -1)
		self.temp_tree = np.copy(tree)
		for ancesID in ancestorIDs:
			self.recursivelyStrip(rootID,tree[:, -1], ancesID)
		#2. delete nodes 
		

		strippedToNodeID = np.cumsum(np.isnan(self.temp_tree[:,-1]))
		for ID in range(self.temp_tree.shape[0]):
			parentID = self.temp_tree[ID,-1]
			if not np.isnan(ID) and not np.isnan(parentID):
				self.temp_tree[ID,-1] -= strippedToNodeID[int(parentID)]
		self.temp_tree[rootID,-1] = -1
		removeIDs = np.argwhere(np.isnan(self.temp_tree[:,-1]))
		self.temp_tree = np.delete(self.temp_tree,removeIDs,axis=0)
		out_tree = self.temp_tree
		self.temp_tree = np.array([0,0,0,-1]).reshape(1,4)

		#adjust goalIDs
		q = np.array([np.argwhere(self.goalIDs == i)[0][0] if i in self.goalIDs else np.nan for i in removeIDs])
		q = q[ ~np.isnan(q)]
		self.goalIDs =  np.delete(self.goalIDs,q, axis = 0)
		self.goalIDs  = np.array([int(int(ID)-strippedToNodeID[int(ID)]) for ID in self.goalIDs])

		return out_tree
	
	def IsvalidPath(self, solPathID):
		solPathID = np.array(solPathID)
		#returns pathID relative to orphanRoot, and the orphaned tree
		#1. Find in-collision nodes
		mask = np.logical_not([ self.collisionFree(self.nodes[i, 0:2]) for i in solPathID]) #node wise
		if not(np.any(mask)): #assert that solpath is in collision
			# use branch-wise mask
			solpath = self.nodes[solPathID, 0:2]
			num_points = 10000
			path_list = np.linspace(solpath[0:-1],solpath[1:],num_points)
			path_list = path_list.reshape(-1,2)

			mask2 = [not self.isBranch(self.nodes[solPathID[i], 0:2], self.nodes[solPathID[i+1],0:2], np.linalg.norm(self.nodes[solPathID[i], 0:2]- self.nodes[solPathID[i+1],0:2])) for i in range(solPathID[:-1].shape[0])]
			mask2 = np.append(mask2, False)
			mask = mask|mask2
		
		mask[0] = False

		maskShifted = np.append(np.array([0]), mask[:-1])
		maskSum = mask + maskShifted
		#2. Find all nodes between in-collision nodes as well
		leftSentinel = np.where(mask)[0][0]
		rightSentinel =  np.where(mask)[0][-1]+1
		mask[leftSentinel: rightSentinel ] = [True for i in range(rightSentinel -leftSentinel)]
		p_separateID = solPathID[np.where(maskSum == 1)[0][-1]]
		deadNodesID = solPathID[mask]

		##### FIND all in-collision nodes 
		allDeadNodesID = np.argwhere([not self.collisionFree(self.nodes[i, 0:2]) for i in range(np.shape(self.nodes)[0])]).reshape(1, -1)[0]
		allDeadNodesID = np.delete(allDeadNodesID, np.argwhere(allDeadNodesID == self.pcurID))
		deadNodesID = list(set(deadNodesID)| set(allDeadNodesID)) #union the 2 sets in case nodes inbetween in-collisions have to be removed as well
		#3. Extract orphan subtree and separate_path to goal
		print("EXTRACTING SUBTREE >>>>")
		self.orphanedTree, self.separatePathID, orphanGoalIDs = self.rerootAtID(p_separateID, self.nodes, solPathID, self.goalIDs)
		#4. Destroy in-collision lineages and update main tree
		self.nodes = self.destroyLineage(deadNodesID,self.nodes)

		return self.separatePathID, self.orphanedTree

	def adoptTree(self, parentNodeID, orphanedTree):
		#1.Adjust orphan ParentIDs and set parent of orphanroot to parentNodeID
		orphanRootNewID = np.where(orphanedTree[:, 3] == -1)[0][0] + np.shape(self.nodes)[0]
		orphanedTree[np.where(orphanedTree[:, 3] != -1),3] = orphanedTree[np.where(orphanedTree[:, 3] != -1),3] + np.shape(self.nodes)[0]
		orphanedTree[np.where(orphanedTree[:, 3] == -1), 3] = parentNodeID #assign parent 
		#2. concat orphanedTree matrix to mainTree matrix and update orphanroot's cost
		fullTree = np.concatenate((self.nodes,orphanedTree), axis = 0)
		fullTree[orphanRootNewID, 2] = fullTree[parentNodeID, 2] + np.linalg.norm(fullTree[parentNodeID, 0:2]- fullTree[orphanRootNewID, 0:2])
		#3. propagate cost from main tree
		q = [] #queue
		children_indices = np.argwhere(fullTree[:,3] == orphanRootNewID) 
		children_indices = list(children_indices)
		q.extend(children_indices)
		#4.COST PROPAGATION ####
		while len(q) != 0:
			child_index = int(q.pop(0))
			parent_index = int(fullTree[child_index,3])
			dist = fullTree[child_index,0:2] - fullTree[parent_index,0:2]
			fullTree[child_index,2] = fullTree[parent_index,2] + np.linalg.norm(dist) #update child's cost
			next_indices = np.argwhere(fullTree[:,3] == child_index)
			next_indices = list(next_indices)
			q.extend(next_indices)
		#5. Recover goalIDs
		self.nodes = fullTree
		normOfDiffs  = np.linalg.norm(self.nodes[:, 0:2] - self.goal, axis =1)
		self.goalIDs = np.argwhere(normOfDiffs < self.tolerance).reshape(-1,)
		
		return fullTree
	
	def reconnect(self, separatePathID):
		print("RECONNECTING >>>>><<<<<<")
		#returns 2 booleans: 1 indicates whether a path to goal already exists, 1 whether reconnect succeeds
		reconnectSuccess  = False
		separatePathID = np.flip(separatePathID)
		for idx in separatePathID:
			#1.center a ball on path node starting from goal
			n = np.shape(self.nodes)[0]
			radius = min(self.eta, self.gamma*np.sqrt(np.log(n)/n))
			pathNode = self.orphanedTree[idx, 0:2]
			distances, NNids = self.NeighbourID(pathNode, radius) 
			#2. search for possible connection from neightbor node 
			for nayID in NNids:
				branchCost = distances[nayID]
				#3. if connection is valid, reroot orpahned tree and let main tree adopt it
				nay = self.nodes[nayID, 0:2][0]
				# print("nay: {}".format(nay))
				if self.isBranch(nay, pathNode, branchCost):
					reconnectSuccess = True
					subtree = self.orphanedTree
					if self.orphanedTree[idx, -1] != -1:
						subtree = self.rerootAtID(idx, self.orphanedTree)
					self.nodes = self.adoptTree(nayID, subtree)
					print("*****Adoption via Reconnection Successful!******")
					costToGoal,goalID = self.minGoalID()
					solpath_ID = self.retrace(goalID)
					return reconnectSuccess,self.nodes[solpath_ID,0:2],solpath_ID
		return reconnectSuccess,None,None


	def regrow(self):
		print("Begin Regrow...")
		max_iterations = 5000
		num_iterations = 0
		goalFound = False
		while not goalFound:
			if num_iterations >= max_iterations:
				return None,None
			#2. Sample
			qrandom = samplepoints(self.xmin,self.ymin,self.xmax,self.ymax)
			#3. Find nearest node to qrandom
			qnear, qnearID = self.NearestNode(qrandom)
			qnew = steer(self.eta,qnear,qrandom)
		 
			if self.isBranch(qnear,qnew,np.linalg.norm(qnear - qnew)):
				num_iterations += 1
				#4. Find nearest neighbors within robot
				n = np.shape(self.nodes)[0] #number of nodes in self
				radius = min(self.eta,self.gamma * np.sqrt(np.log(n) / n))
				distances,NNids = self.NeighbourID(qnew,radius) 				
				#5. Choose qnew's best parent and insert qnew
				naysID = np.append(np.array([qnearID]),NNids)
				qparentID,qnewCost = self.chooseParent(qnew,naysID,distances)	
				qnewID = self.edge(int(qparentID),qnew,qnewCost)	
				#6. If qnew is near goal, store its id
				if np.linalg.norm(qnew - self.goal) < self.tolerance:
					goalFound = True
					#Append qnewID(goalID) to self.goalIDs list		
					self.addGoalID(int(qnewID))
				#7. Rewire within the robot's vicinity
				self.rewire(qnewID,naysID,distances)

				#8.Trim tree
				if np.shape(self.nodes)[0] > self.maxNumNodes:
					self.RemoveNode(qnewID,self.goal,goalFound)

				if goalFound:
					costToGoal,goalID = self.minGoalID()
					solpath_ID = self.retrace(goalID)
					return self.nodes[solpath_ID,0:2],solpath_ID

				else:
					separatePathID = np.flip(self.separatePathID)
					dist = np.linalg.norm(self.orphanedTree[separatePathID,0:2] - qnew, axis = 1)
					n = np.shape(self.nodes)[0] 
					radius = min(self.eta,self.gamma * np.sqrt(np.log(n) / n))
					poss_connectionIDs = separatePathID[dist <= radius]
					dist = dist[dist <= radius]
					
					for i,idx in enumerate(poss_connectionIDs):
						print("ATTEMPTING TO ADOPT ORPHANED TREE IN REGROW >>>>")
						pathNode = self.orphanedTree[idx,0:2]
						branchCost = dist[i]
						if self.isBranch(pathNode,qnew,branchCost):
							goalFound = True

							subtree = np.copy(self.orphanedTree)
							if self.orphanedTree[idx, -1] != -1:
								subtree = self.rerootAtID(idx,subtree)

							# 4. adopt subtree rooted at furthest node on separatePath at qnewID to main tree
							self.nodes = self.adoptTree(qnewID,subtree)
							print("			ADOPTION IN REGROW SUCCESSFUL>>>>>>>")
							costToGoal,goalID = self.minGoalID()
							solpath_ID = self.retrace(goalID)
							return self.nodes[solpath_ID,0:2],solpath_ID
		
		return None

	def nextSolNode(self,solPath,solPathID):
		self.pcurID = solPathID[1]
		#computes length of branch traversed
		dt = self.nodes[solPathID[1], 2] - self.nodes[solPathID[0], 2]
		return solPath[1:],solPathID[1:],dt

	def reset(self, inheritCost = True):
		#clears all nodes and seed new tree at self.pcur
		newroot = self.nodes[self.pcurID, 0:2]
		newrootCost = self.nodes[self.pcurID, 2]
		self.nodes = np.array([newroot[0],newroot[1],0,-1]).reshape(1,4)	
		if inheritCost is True:
			self.nodes = np.array([newroot[0],newroot[1], newrootCost,-1]).reshape(1,4)		
		self.goalIDs = np.array([]).astype(int) # list of near-goal nodeIDs
		self.update_q = [] # for cost propagation
		self.orphanedTree = np.array([0,0,0,0]).reshape(1,4)
		self.separatePathID = np.array([]) # IDs along path to goal in the orphaned tree

class Obstacle(object):
	
	def __init__(self,shape,param,mean_velocity,covar_vel=np.eye(2)*0.02,speed=1, \
			  boundaries=[0,0,30,30], goal_cor = [28,28,1]):

		if not ( ((shape=='rect') and (len(param)==4)) or \
				 ((shape=='circle') and (len(param)==3)) ):
			raise ValueError
		self.shape = shape
		self.params = param
		if shape == 'rect':
			self.position = np.array([param[0], param[1]])
			self.width = param[2]
			self.height = param[3]
		if shape == 'circle':
			self.position = np.array([param[0], param[1]])
			self.radius = param[2]
			
		#four walls of environment for bounces
		self.xmin = boundaries[0]
		self.ymin = boundaries[1]
		self.xmax = boundaries[2]
		self.ymax = boundaries[3]
		#robot radius for bounces
		self.robot_rad = 0.5
		#goal location and radius for bounces
		self.goal_x = goal_cor[0]
		self.goal_y = goal_cor[1]
		self.goal_r = goal_cor[2]
		#mean_velocity = mean of velocity mutltivariate Gaussian (updated at each t)
		#covar_vel = covariance of velocity multivariate Gaussian
		#speed = speed (limit) of obstacle
		self.mean_velocity = mean_velocity
		self.covar_vel = covar_vel
		self.speed = speed

	def isCollisionFree(self, x):
		#returns a boolean indicating whether obstacle is in collision
		if len(x.shape) == 1:
			x = x.reshape(1,2)

		if self.shape == 'rect':
			x_check = np.logical_and(x[:,0] >= self.position[0] - self.robot_rad,x[:,0] <= self.position[0] + self.width + self.robot_rad)
			y_check = np.logical_and(x[:,1] >= self.position[1] - self.robot_rad,x[:,1] <= self.position[1] + self.height + self.robot_rad)
			check = np.logical_and(x_check,y_check)		
			obs_check = check.any()
			return np.logical_not(obs_check)	
			
		else:
			temp = x - self.position
			dist = np.linalg.norm(temp,axis = 1)
			check = dist <= self.radius + self.robot_rad
			obs_check = check.any()
			return np.logical_not(obs_check)


	def toPatch(self, color = [0.1,0.2,0.7]):
		#returns patch object for plotting
		if self.shape == 'rect':
			return patches.Rectangle((self.position[0], self.position[1]), \
							self.width, self.height, \
								ec='k', lw=1.5, facecolor=color)

		return patches.Circle((self.position[0], self.position[1]), \
						self.radius, \
							ec='k', lw=1.5, facecolor=color)

	def moveObstacle(self,p_cur,dt):
		#updates dynamics and returns next timestep position
		# sample random velocity
		vel = np.random.multivariate_normal(self.mean_velocity, self.covar_vel)
		norm = np.linalg.norm(vel)
		if not norm==0:
			vel = self.speed*(vel/norm)
		# check for rebound and update + return obstacle position, velocity
		self.mean_velocity,self.position = self.doRebound(p_cur[0],p_cur[1],vel,dt)
		return self.position
	
	def doRebound(self,bot_x,bot_y,vel,dt):
		# temporary new position
		new = self.position + vel*dt

		# check for border rebound and udpate vel if necessary
		borderrebound = False
		if self.shape == 'rect':
			if new[0] + self.width > self.xmax:
				vel *= np.array([-1,1])
				borderrebound = True

			elif new[0] < self.xmin:
				vel *= np.array([-1,1])
				borderrebound = True

			if new[1] + self.height > self.ymax:
				vel *= np.array([1,-1])
				borderrebound = True

			elif new[1] < self.ymin:
				vel *= np.array([1,-1])
				borderrebound = True
				
		else: # self.shape=='circle'
			if new[0] > self.xmax - self.radius:
				vel *= np.array([-1,1])
				borderrebound = True
				
			elif new[0] < self.xmin + self.radius:
				vel *= np.array([-1,1])
				borderrebound = True
				
			if new[1] > self.ymax - self.radius:
				vel *= np.array([1,-1])
				borderrebound = True
				
			elif new[1] < self.ymin + self.radius:
				vel *= np.array([1,-1])
				borderrebound = True
				

		# check for goal rebound and update vel if necessary
		goalrebound = False
		goal = np.array([self.goal_x,self.goal_y])
		if self.shape == 'rect':
			center = new + np.array([self.width/2,self.height/2])
			diff = np.abs(center - goal)
			if ( diff[0] < self.goal_r + self.width/2 ) and \
					( new[1] + self.height > self.goal_y and \
					new[1] < self.goal_y ):
				vel *= np.array([-1,1])
				goalrebound = True
				
			elif ( diff[1] < self.goal_r + self.height/2 ) and \
					( new[0] + self.width > self.goal_x and \
					new[0] < self.goal_x ):
				vel *= np.array([1,-1])
				goalrebound = True
				
			elif ( np.linalg.norm(diff) < self.goal_r + np.linalg.norm([self.width/2,self.height/2]) ):
				vel *= np.array([-1,-1])
				goalrebound = True
				
		else: # self.shape=='circle'
			dist = np.linalg.norm(new - goal)
			if dist < (self.radius + self.goal_r):
				vel *= np.array([-1,-1])
				goalrebound = True
				

		# check for robot rebound and update vel if necessary
		robo_rebound = False
		robot = np.array([bot_x,bot_y])
		if self.shape == 'rect':
			center = new + np.array([self.width/2,self.height/2])
			diff = np.abs(center - robot)
			if ( diff[0] < self.robot_rad + self.width/2 ) and \
					( new[1] + self.height > bot_y and \
					new[1] < bot_y ):
				vel *= np.array([-1,1])
				robo_rebound = True
				
			elif ( diff[1] < self.robot_rad + self.height/2 ) and \
					( new[0] + self.width > bot_x and \
					new[0] < bot_x ):
				vel *= np.array([1,-1])
				robo_rebound = True
				
			elif ( np.linalg.norm(diff) < self.robot_rad + np.linalg.norm([self.width/2,self.height/2]) ):
				vel *= np.array([-1,-1])
				robo_rebound = True
				
		else: # self.shape=='circle'
			dist = np.linalg.norm(new - robot)
			if dist < (self.radius + self.robot_rad):
				vel *= np.array([-1,-1])
				robo_rebound = True
				
		norm = np.linalg.norm(vel)
		if not norm==0:
			vel = self.speed*(vel/norm)
		# if two rebounds happen, obstacle stops for this timestep
		if (robo_rebound or goalrebound) and borderrebound:
			vel *= vel*np.array([0,0])
		# if rebound necessary, compute position again
		if (borderrebound or goalrebound or robo_rebound):
			new = self.position + vel*dt
		return vel,new
	