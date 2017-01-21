require 'torch'
require 'nn'
require 'LSTM'
require 'cunn'
require 'hdf5'
local utils = require 'utils'

local checkpoint = torch.load('trainModel_final.t7')
local model = checkpoint.model
model:cuda()
local criterion = nn.ClassNLLCriterion():cuda()

utils.printTime("Initializing DataLoader")
batchLoader=require 'DataLoader'

local loader = batchLoader.create(100)

function test()
	split='test'
	collectgarbage()
	utils.printTime("Starting testing ")

	local evalData = {}
	evalData.predictions={}

	evalData.loss = 0 -- sum of losses
	evalData.numBatches = 0 -- number of batches run
	local batch = loader:getTestSet()
	batchSize=batch.data:size(1)
	print("test dataset length is ",batchSize)
	print("batch.data:size() is ",batch.data:size())
	print("batch.label:size() is ",batch.label:size())
	batch.data=batch.data:cuda()
	batch.label=batch.label:cuda()
	for i=1,batchSize do
		local prediction = model:forward(batch.data[i])
		local confidence,indice = torch.sort(prediction,true)
		maxprediction=indice[1]-1
		print(maxprediction)
		table.insert( evalData.predictions , maxprediction )
	end
	--print("the tensor of predictions is : ",evalData.predictions:size())
	-- local predicts=torch.cat(evalData.predictions)
	local trueLabels=batch.label
	local preLength=#evalData.predictions
	print("predicts size is ",preLength)
	print("labels size is ",trueLabels:size())
	local predictLabels=torch.Tensor(preLength)
	for i=1,preLength do
		predictLabels[i]=evalData.predictions[i]
	end
	-- save results to file
	local myFile=hdf5.open('predictValue.h5','w')
	myFile:write('predicts',predictLabels:float())
	myFile:write('labels',trueLabels:float())
	myFile:close()
	-- then exec the score.py
	os.execute('python score.py')
end

test()