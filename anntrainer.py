import numpy as np 
import time

class BackPropagationNetwork:

	layerCount = 0;
	shape  = None;
	weights = [];

	def __init__(self,layerSize):

		self.layerCount = len(layerSize) - 1;
		self.shape = layerSize

		self._layerInput = []
		self._layerOutput = []
		self._previousWeightDelta = []

		for (l1,l2) in zip(layerSize[:-1],layerSize[1:]):
			self.weights.append(np.random.normal(scale=0.1,size=(l2,l1+1)))
			self._previousWeightDelta.append(np.zeros((l2,l1+1)))

	def forwardProc(self,input):

		InCases = input.shape[0]

		self._layerInput = []
		self._layerOutput = []

		for index in range(self.layerCount):
			if index == 0:
				#print "weight" + str(self.weights[0])
				#print "vstack" + str(np.vstack([input.T,np.ones([1,InCases])]))
				layerInput = self.weights[0].dot(np.vstack([input.T,np.ones([1,InCases])]))
				#print "layerInput" + str(layerInput)
			else:
				layerInput = self.weights[index].dot(np.vstack([self._layerOutput[-1],np.ones([1,InCases])]))

			self._layerInput.append(layerInput)
			self._layerOutput.append(self.sgm(layerInput))

		return self._layerOutput[-1].T

	def train(self,input,target, trainingRate = 0.2, momentum = 0.5):

		delta = []
		InCases = input.shape[0]

		self.forwardProc(input)

		#Delta calculation
		for index in reversed(range(self.layerCount)):

			if index == self.layerCount - 1 :
				output_delta = self._layerOutput[index] - target.T
				error = np.sum(output_delta**2)
				delta.append(output_delta * self.sgm(self._layerInput[index],True))

			else:

				delta_pullback = self.weights[index+1].T.dot(delta[-1])
				delta.append(delta_pullback[:-1,:] * self.sgm(self._layerInput[index],True))

		#Weight Delta Calculation
		for index in range(self.layerCount):
			delta_index  = self.layerCount - 1 - index

			if index == 0:
				layerOutput  = np.vstack([input.T,np.ones([1,InCases])])
			else:
				layerOutput = np.vstack([self._layerOutput[index-1],np.ones([1,self._layerOutput[index-1].shape[1]])])

			currWeightDelta = np.sum(layerOutput[None,:,:].transpose(2,0,1) * delta[delta_index][None,:,:].transpose(2,1,0),axis = 0)

			weightDelta = trainingRate * currWeightDelta + momentum * self._previousWeightDelta[index]

			self.weights[index] -= weightDelta

			self._previousWeightDelta[index] = weightDelta

		return error

	def sgm(self,x,Derivative=False):
		if not Derivative:
			return 1/ (1+np.exp(-x))
		else:
			out = self.sgm(x)
			return out*(1-out)


if __name__ == "__main__":
	bpn = BackPropagationNetwork((260,25,25,28))
	

	# f1 = open("mfccData/alo_mfcc.npy", "rb")
	# f2 = open("mfccData/batden_mfcc.npy", "rb")
	# f3 = open("mfccData/batquat_mfcc.npy", "rb")
	# f4 = open("mfccData/tatden_mfcc.npy", "rb")
	# f5 = open("mfccData/tatquat_mfcc.npy", "rb")

	# inputArray1 = np.load(f1, allow_pickle=True, encoding="bytes")
	# inputArray2 = np.load(f2, allow_pickle=True, encoding="bytes")
	# inputArray3 = np.load(f3, allow_pickle=True, encoding="bytes")
	# inputArray4 = np.load(f4, allow_pickle=True, encoding="bytes")
	# inputArray5 = np.load(f5, allow_pickle=True, encoding="bytes")

	# inputArray = np.concatenate((inputArray1,inputArray2,inputArray3,inputArray4,inputArray5))

	# print(inputArray.shape)

	# t1 = np.array([[1,0,0,0,0] for _ in range(len(inputArray1))])
	# t2 = np.array([[0,1,0,0,0] for _ in range(len(inputArray2))])
	# t3 = np.array([[0,0,1,0,0] for _ in range(len(inputArray3))])
	# t4 = np.array([[0,0,0,1,0] for _ in range(len(inputArray4))])
	# t5 = np.array([[0,0,0,0,1] for _ in range(len(inputArray5))])

	# target = np.concatenate([t1,t2,t3,t4,t5])
	# print('target: ', target.shape)




	f1 = open("mfccData2/alo_mfcc.npy", "rb")
	f2 = open("mfccData2/batdenbancong_mfcc.npy", "rb")
	f3 = open("mfccData2/batdennhabep_mfcc.npy", "rb")
	f4 = open("mfccData2/batdenphongkhach_mfcc.npy", "rb")
	f5 = open("mfccData2/batdenphongngu_mfcc.npy", "rb")
	f6 = open("mfccData2/batdentoilet_mfcc.npy", "rb")
	f7 = open("mfccData2/batlovisong_mfcc.npy", "rb")
	f8 = open("mfccData2/batquatphongkhach_mfcc.npy", "rb")
	f9 = open("mfccData2/batquatphongngu_mfcc.npy", "rb")
	f10 = open("mfccData2/battiviphongkhach_mfcc.npy", "rb")
	f11 = open("mfccData2/battiviphongngu_mfcc.npy", "rb")
	f12 = open("mfccData2/dongcuanhabep_mfcc.npy", "rb")
	f13 = open("mfccData2/dongcuanhavesinh_mfcc.npy", "rb")
	f14 = open("mfccData2/dongcuaphongkhach_mfcc.npy", "rb")
	f15 = open("mfccData2/dongcuaphongngu_mfcc.npy", "rb")
	f16 = open("mfccData2/mocuanhabep_mfcc.npy", "rb")
	f17 = open("mfccData2/mocuanhavesinh_mfcc.npy", "rb")
	f18 = open("mfccData2/mocuaphongkhach_mfcc.npy", "rb")
	f19 = open("mfccData2/mocuaphongngu_mfcc.npy", "rb")
	f20 = open("mfccData2/tatdenbancong_mfcc.npy", "rb")
	f21 = open("mfccData2/tatdennhabep_mfcc.npy", "rb")
	f22 = open("mfccData2/tatdenphongkhach_mfcc.npy", "rb")
	f23 = open("mfccData2/tatdenphongngu_mfcc.npy", "rb")
	f24 = open("mfccData2/tatdentoilet_mfcc.npy", "rb")
	f25 = open("mfccData2/tatlovisong_mfcc.npy", "rb")
	f26 = open("mfccData2/tatquatphongkhach_mfcc.npy", "rb")
	f27 = open("mfccData2/tatquatphongngu_mfcc.npy", "rb")
	f28 = open("mfccData2/tattiviphongkhach_mfcc.npy", "rb")


	inputArray1 = np.load (f1, allow_pickle = True, encoding = "bytes")
	inputArray2 = np.load (f2, allow_pickle = True, encoding = "bytes")
	inputArray3 = np.load (f3, allow_pickle = True, encoding = "bytes")
	inputArray4 = np.load (f4, allow_pickle = True, encoding = "bytes")
	inputArray5 = np.load (f5, allow_pickle = True, encoding = "bytes")
	inputArray6 = np.load (f6, allow_pickle = True, encoding = "bytes")
	inputArray7 = np.load (f7, allow_pickle = True, encoding = "bytes")
	inputArray8 = np.load (f8, allow_pickle = True, encoding = "bytes")
	inputArray9 = np.load (f9, allow_pickle = True, encoding = "bytes")
	inputArray10 = np.load (f10, allow_pickle = True, encoding = "bytes")
	inputArray11 = np.load (f11, allow_pickle = True, encoding = "bytes")
	inputArray12 = np.load (f12, allow_pickle = True, encoding = "bytes")
	inputArray13 = np.load (f13, allow_pickle = True, encoding = "bytes")
	inputArray14 = np.load (f14, allow_pickle = True, encoding = "bytes")
	inputArray15 = np.load (f15, allow_pickle = True, encoding = "bytes")
	inputArray16 = np.load (f16, allow_pickle = True, encoding = "bytes")
	inputArray17 = np.load (f17, allow_pickle = True, encoding = "bytes")
	inputArray18 = np.load (f18, allow_pickle = True, encoding = "bytes")
	inputArray19 = np.load (f19, allow_pickle = True, encoding = "bytes")
	inputArray20 = np.load (f20, allow_pickle = True, encoding = "bytes")
	inputArray21 = np.load (f21, allow_pickle = True, encoding = "bytes")
	inputArray22 = np.load (f22, allow_pickle = True, encoding = "bytes")
	inputArray23 = np.load (f23, allow_pickle = True, encoding = "bytes")
	inputArray24 = np.load (f24, allow_pickle = True, encoding = "bytes")
	inputArray25 = np.load (f25, allow_pickle = True, encoding = "bytes")
	inputArray26 = np.load (f26, allow_pickle = True, encoding = "bytes")
	inputArray27 = np.load (f27, allow_pickle = True, encoding = "bytes")
	inputArray28 = np.load (f28, allow_pickle = True, encoding = "bytes")


	inputArray = np.concatenate((inputArray1, inputArray2, inputArray3, inputArray4, inputArray5, inputArray6, inputArray7, inputArray8, inputArray9, inputArray10, inputArray11, inputArray12, inputArray13, inputArray14, inputArray15, inputArray16, inputArray17, inputArray18, inputArray19, inputArray20, inputArray21, inputArray22, inputArray23, inputArray24, inputArray25, inputArray26, inputArray27, inputArray28))

	print('input: ', inputArray.shape)

	''' t1 = np.array([[1,0,0,0,0] for _ in range(len(inputArray1))])
	t2 = np.array([[0,1,0,0,0] for _ in range(len(inputArray2))])
	t3 = np.array([[0,0,1,0,0] for _ in range(len(inputArray3))])
	t4 = np.array([[0,0,0,1,0] for _ in range(len(inputArray4))])
	t5 = np.array([[0,0,0,0,1] for _ in range(len(inputArray5))]) '''

	t1 = np.array ([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ] for _ in range(len(inputArray1 ))])
	t2 = np.array ([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ] for _ in range(len(inputArray2 ))])
	t3 = np.array ([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ] for _ in range(len(inputArray3 ))])
	t4 = np.array ([[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ] for _ in range(len(inputArray4 ))])
	t5 = np.array ([[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ] for _ in range(len(inputArray5 ))])
	t6 = np.array ([[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ] for _ in range(len(inputArray6 ))])
	t7 = np.array ([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ] for _ in range(len(inputArray7 ))])
	t8 = np.array ([[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ] for _ in range(len(inputArray8 ))])
	t9 = np.array ([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ] for _ in range(len(inputArray9 ))])
	t10 = np.array ([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ] for _ in range(len(inputArray10 ))])
	t11 = np.array ([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ] for _ in range(len(inputArray11 ))])
	t12 = np.array ([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ] for _ in range(len(inputArray12 ))])
	t13 = np.array ([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ] for _ in range(len(inputArray13 ))])
	t14 = np.array ([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ] for _ in range(len(inputArray14 ))])
	t15 = np.array ([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ] for _ in range(len(inputArray15 ))])
	t16 = np.array ([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ] for _ in range(len(inputArray16 ))])
	t17 = np.array ([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ] for _ in range(len(inputArray17 ))])
	t18 = np.array ([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ] for _ in range(len(inputArray18 ))])
	t19 = np.array ([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 ] for _ in range(len(inputArray19 ))])
	t20 = np.array ([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0 ] for _ in range(len(inputArray20 ))])
	t21 = np.array ([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0 ] for _ in range(len(inputArray21 ))])
	t22 = np.array ([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 ] for _ in range(len(inputArray22 ))])
	t23 = np.array ([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0 ] for _ in range(len(inputArray23 ))])
	t24 = np.array ([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0 ] for _ in range(len(inputArray24 ))])
	t25 = np.array ([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0 ] for _ in range(len(inputArray25 ))])
	t26 = np.array ([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0 ] for _ in range(len(inputArray26 ))])
	t27 = np.array ([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0 ] for _ in range(len(inputArray27 ))])
	t28 = np.array ([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 ] for _ in range(len(inputArray28 ))])


	target = np.concatenate([t1 , t2 , t3 , t4 , t5 , t6 , t7 , t8 , t9 , t10 , t11 , t12 , t13 , t14 , t15 , t16 , t17 , t18 , t19 , t20 , t21 , t22 , t23 , t24 , t25 , t26 , t27 , t28])
	print('target: ', target.shape)

	lnMax = 1000000
	lnErr = 1e-5

	startTime = time.time()

	#Train Loop
	for i in range(lnMax-1):
		err = bpn.train(inputArray,target,momentum = 0.3)
		# print("Iteration {0} \tError: {1:0.6f}".format(i,err))
		if i % 1000 == 0:
			print("Iteration {0} \tError: {1:0.6f}".format(i,err))
		if err <= lnErr:
			print("Minimum error reached at iteration {0}".format(i))
			break

	endTime = time.time()

	with open("network/" + "vowel_network_words_30"+ ".npy", 'wb') as outfile:
  		np.save(outfile,bpn.weights)

  	

	lvOutput = bpn.forwardProc(inputArray)
	print("Output {0}".format(lvOutput))

	print("Time Elapsed: " + str(endTime - startTime) + " seconds")
	print("Total Iteration {0} \t Total Error: {1:0.6f}".format(i,err))
	