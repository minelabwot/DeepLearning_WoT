require 'torch'
require 'nn'
require 'cunn'
require 'optim'
require 'rnn'
require 'hdf5'

local utils = require 'utils'
--和输入数据格式务必保持一致
seqLength=60

numClass=17
slidingWindowLength=24
sensorChannel=113

batchSize=128

local cnn = {}
cnn.inputChannel=1
cnn.numHiddenUnits=64
cnn.kernel1=1
cnn.kernel2=5
cnn.outputs=1024

lstmHiddenUnits1=64
lstmHiddenUnits2=256
lstmHiddenUnits3=128

-- input(60,1,24,113),output(60,)

local model=nn.Sequential()

model:add(nn.SpatialConvolution(1,8,1,5))--(60,8,20,113)
model:add(nn.ReLU())
model:add(nn.SpatialConvolution(8,8,1,5))--(60,8,16,113)
model:add(nn.ReLU())
model:add(nn.Transpose({2,3}))--(60,16,8,113)
model:add(nn.View(seqLength,16,-1))--(60,16,904)
model:add(nn.Sequencer(nn.LSTM(904,128)))
model:add(nn.Sequencer(nn.LSTM(128,64)))--60,16,64

model:add(nn.View(60,-1))--60*1024
model:add(nn.Sequencer(nn.LSTM(1024,512)))--60*512
model:add(nn.Sequencer(nn.LSTM(512,128)))--60*128

model:add(nn.Linear(128,numClass))
model:add(nn.LogSoftMax())

model=model:cuda()

criterion=nn.ClassNLLCriterion()

batchLoader=require 'DataLoader'

local loader = batchLoader.create(batchSize)

local numEpochs=200

-- local checkpoint = torch.load('retrainModel_60_35.t7')
-- local model = checkpoint.model
-- model=model:cuda()

function train(model)
  utils.printTime("Starting training for %d epochs" % {numEpochs})

  local trainLossHistory = {}
  local valLossHistory = {}
  local checkpointEvery = 3
  local lrDecayEvery = 5
  local lrDecayFactor = 0.5
  local printEvery = 1

  local config = {learningRate = 1e-3}
  local params, gradParams = model:getParameters()

  for i = 1, numEpochs do
    collectgarbage()
    utils.printTime("Starting training for the %d th epoch" % {i})
    local epochLoss = {}

    -- if i % lrDecayEvery == 0 then
    --   local oldLearningRate = config.learningRate
    --   config = {learningRate = oldLearningRate * lrDecayFactor}
    -- end

    local batch = loader:nextBatch('train')
    batch_poch=0
    while batch ~= nil do
      print('get batch data...')
      print(batch_poch)
      batch_poch=batch_poch+1
      batch.data = batch.data:cuda()
      batch.label = batch.label:cuda()
      criterion=criterion:cuda()

      local function feval(x)
        collectgarbage()

        if x ~= params then
          params:copy(x)
        end

        gradParams:zero()

        local loss = 0

        for j=1,batchSize do

          local trueLabel = batch.label[j]
          local modelOut = model:forward(batch.data[j])

          local modelLoss = criterion:forward(modelOut, trueLabel)
          loss=loss+modelLoss
          local gradOutput = criterion:backward(modelOut, trueLabel)
          model:backward(batch.data[j], gradOutput)

        end
        
        loss=loss/batchSize
        gradParams:div(batchSize)
        return loss, gradParams
      end

      local _, loss = optim.adam(feval, params, config)
      table.insert(epochLoss, loss[1])
      print("batch_loss is :")
      print(loss)

      batch = loader:nextBatch('train')
      --batch=nil
    end
    --when batch is nil,the above loop end,so cal the epochloss
    local epochLoss = torch.mean(torch.Tensor(epochLoss))
    table.insert(trainLossHistory, epochLoss)

    -- Print the epoch loss
    if (printEvery > 0 and i % printEvery == 0) then
      utils.printTime("Epoch %d training loss: %f" % {i, epochLoss})
    end

    -- Save a checkpoint of the model, its opt parameters, the training loss history, and the validation loss history
    if (checkpointEvery > 0 and i % checkpointEvery == 0) or i == numEpochs then
      -- to know the validation loss,so we can know the train if have completed
      --and after we can need to earlyStop depends on the valLoss not increasing any more
        
      local filename
      if i == numEpochs then
        filename = '%s_60_%s.t7' % {'LRCL','final'}
      else
        filename = '%s_60_%d.t7' % {'LRCL', i}
      end

      -- Make sure the output directory exists before we try to write it
      paths.mkdir(paths.dirname(filename))  

      -- local valLoss = validate()
      -- utils.printTime("Epoch %d validation loss: %f" % {i, valLoss})
      -- table.insert(valLossHistory, valLoss)    

      -- local checkpoint = {
      --   trainLossHistory = trainLossHistory,
      --   valLossHistory = valLossHistory
      -- }

      local checkpoint = {
        trainLossHistory=trainLossHistory
      }

      -- Cast model to float so it can be used on CPU
      model:float()
      checkpoint.model = model
      torch.save(filename, checkpoint)

      -- Cast model back so that it can continue to be used
      model:cuda()
      params, gradParams = model:getParameters()
      utils.printTime("Saved checkpoint model and opt at %s" % filename)
      collectgarbage()
    end
  end

  utils.printTime("Finished training and begin testing")

end

function validate()
  split='val'
  utils.printTime("Starting testing on the %s split" % {split})

  local evalData = {}
   
  evalData.loss = 0 -- sum of losses
  evalData.numBatches = 0 -- total number of frames

  local valbatch = loader:nextBatch(split)

  while valbatch ~= nil do

    valbatch.data = valbatch.data:cuda()
    valbatch.label = valbatch.label:cuda()
    model=model:cuda()
    criterion=criterion:cuda()

    local function feval()

      local loss = 0

      for k=1,batchSize do
        local modelOut = model:forward(valbatch.data[k])
        local modelLoss = criterion:forward(modelOut, valbatch.label[k])
        loss=loss+modelLoss
      end
      
      return loss

    end

    evalData.loss = evalData.loss +feval()
    evalData.numBatches = evalData.numBatches + 1
    valbatch = loader:nextBatch(split)

  end

  return evalData.loss / (evalData.numBatches*batchSize)
end
-- can be a single file

-- function test()

--   local testData=loader:getTestSet()
--   prelist={}
--   for i=1,testData.data:size(1) do
--     local prediction = model:forward(testData.data[i])
--     table.insert(prelist,prediction)

--   end
  
--   -- just save the predicitons to file,and the use python to cal the scores
--   local myFile = hdf5.open('torchResult.h5', 'w')
--   myFile:write('predictions', prelist)
--   myFile:write('labels', testData.label)
--   myFile:close()

-- end

train(model)
print("\n-----train have done----\n")

