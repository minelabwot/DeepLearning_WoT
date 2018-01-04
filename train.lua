require 'torch'
require 'nn'
require 'cunn'
require 'optim'
require 'rnn'
require 'hdf5'
require 'gnuplot'

local utils = require 'utils'

numClass=7
slidingWindowLength=24
sensorChannel=114
batchSize=128

local numEpochs=300

local criterionWeight=torch.Tensor({20.575,26.523, 29.634,1572.25,156.573,5.199,1.469})

local cnn = {}
cnn.inputChannel=1
cnn.numHiddenUnits=64
cnn.kernel1=1
cnn.kernel2=5
cnn.outputs=1024

lstmHiddenUnits1=64
lstmHiddenUnits2=256
lstmHiddenUnits3=128

-- input(20,1,24,114),output(20,)

local model=nn.Sequential()

model:add(nn.SpatialConvolution(1,8,1,5))--(20,8,20,114)
model:add(nn.SpatialBatchNormalization(8))
model:add(nn.ReLU())
model:add(nn.SpatialDropout(0.3))
model:add(nn.SpatialBatchNormalization(8))
model:add(nn.SpatialConvolution(8,8,1,5))--(20,8,16,114)
model:add(nn.ReLU())
model:add(nn.SpatialDropout(0.3))
model:add(nn.Transpose({2,3}))--(20,16,8,114)
model:add(nn.View(20,16,-1))--(20,16,912)
model:add(nn.Sequencer(nn.FastLSTM(912,256)))
model:add(nn.Sequencer(nn.FastLSTM(256,64)))--20,16,64

model:add(nn.View(20,-1))--20*1024
model:add(nn.Sequencer(nn.FastLSTM(1024,512)))--20*512
model:add(nn.Sequencer(nn.FastLSTM(512,128)))--20*128

model:add(nn.Linear(128,numClass))
model:add(nn.LogSoftMax())

model=model:cuda()

criterion=nn.ClassNLLCriterion(criterionWeight)

batchLoader=require 'DataLoader'

local loader = batchLoader.create(batchSize)

function train(model)
  utils.printTime("Starting training for %d epochs" % {numEpochs})

  local trainLossHistory = {}
  local trainAccHistory = {}
  local valLossHistory = {}
  local valAccHistory = {}
  local epochHistory = {}
  local checkpointEvery = 20
  local printEvery = 1

  local config = {learningRate = 5e-3}
  local params, gradParams = model:getParameters()
  local LeastLoss = 100
  local bestEpoch = 0
  local waitTimes=0

  for i = 1, numEpochs do
    collectgarbage()
    utils.printTime("Starting training for the %d th epoch" % {i})
    local epochLoss = {}
    local train_acc_num = 0
    local train_dataSize =0

    local batch = loader:nextBatch('train')

    while batch ~= nil do
      print('get batch data...')
      train_dataSize = train_dataSize+ batch.data:size(1)
      batch.data = batch.data:cuda()
      batch.label = batch.label:cuda()
      criterion=criterion:cuda()

      local function feval(x)
        collectgarbage()

        if x ~= params then
          params:copy(x)
        end

        gradParams:zero()

        local train_batchLoss = {}

        for j=1,batchSize do

          local trueLabel = batch.label[j]
          local modelOut = model:forward(batch.data[j])
          local confidence,indice = torch.sort(modelOut,true)
          if indice[-1][1] == trueLabel[-1] then
            train_acc_num = train_acc_num+1
          end
          local modelLoss = criterion:forward(modelOut, trueLabel)
          table.insert(train_batchLoss,modelLoss)
          local gradOutput = criterion:backward(modelOut, trueLabel)
          model:backward(batch.data[j], gradOutput)

        end

        loss=torch.mean(torch.Tensor(train_batchLoss))
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

    local train_acc = train_acc_num/train_dataSize
    table.insert(trainAccHistory,train_acc)  

    table.insert(epochHistory,i)  
    --when batch is nil,the above loop end,so cal the epochloss
    local epochLoss = torch.mean(torch.Tensor(epochLoss))
    table.insert(trainLossHistory, epochLoss)

    local validLoss, valAcc = validate(model)
    table.insert(valLossHistory, validLoss) 
    table.insert(valAccHistory,valAcc) 

    -- Print the epoch matrics
    if (printEvery > 0 and i % printEvery == 0) then
      utils.printTime("Epoch %d training loss: %f" % {i, epochLoss})
      utils.printTime("Epoch %d validation loss: %f" % {i, validLoss})
      utils.printTime("Epoch %d training acc is %s" % {i,train_acc})
      utils.printTime("Epoch %d validation acc: %f" % {i, valAcc})
    end

    -- Save a checkpoint of the model, its opt parameters, the training loss history, and the validation loss history
    if (checkpointEvery > 0 and i % checkpointEvery == 0) or i == numEpochs then
      -- to know the validation loss,so we can know the train if have completed
      --and after we can need to earlyStop depends on the valLoss not increasing any more
        
      local filename
      if i == numEpochs then
        filename = '%s_%s.t7' % {'model_LRCL_classifier_'..slidingWindowLength..'_date_'..os.date("%m_%d"),'final'}
      else
        filename = '%s_%d.t7' % {'model_LRCL_classifier_'..slidingWindowLength..'_date_'..os.date("%m_%d"), i}
      end

      -- Make sure the output directory exists before we try to write it
      paths.mkdir(paths.dirname(filename))    

      local checkpoint = {
        trainLossHistory = trainLossHistory,
        valLossHistory = valLossHistory
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

    if validLoss < LeastLoss then
      LeastLoss, bestEpoch = validLoss, i
      local checkpoint = {
        trainLossHistory = trainLossHistory,
        valLossHistory = valLossHistory,
        model = model
      }
      torch.save("best_model_LRCL_classifier_"..slidingWindowLength..".t7", checkpoint)
      print(string.format("New maxima : %f @ %f", LeastLoss, bestEpoch))
      waitTimes = 0
    else
      waitTimes = waitTimes + 1
      if waitTimes >30 then break end
    end

  end

  -- draw the comparion of trainLossHistory and valLossHistory
  local cgTrainLoss=torch.Tensor(trainLossHistory)
  local cgValLoss=torch.Tensor(valLossHistory)
  local cgTrainAcc=torch.Tensor(trainAccHistory)
  local cgValAcc=torch.Tensor(valAccHistory)
  local cgEpoch=torch.Tensor(epochHistory)
  gnuplot.title('CG loss minimisation over time')
  gnuplot.pngfigure('trainValLossAcc_win'..slidingWindowLength..'.png')
  gnuplot.plot({'trainLoss',cgEpoch,cgTrainLoss,'-'},{'valLoss',cgEpoch,cgValLoss,'-'},
  {'trainAcc',cgEpoch,cgTrainAcc,'-'},{'valAcc',cgEpoch,cgValAcc,'-'})
  gnuplot.xlabel('epoch')
  gnuplot.ylabel('matrics')
  gnuplot.plotflush()

  utils.printTime("Finished training and begin testing")

end

function validate(model)
  local split='val'
  utils.printTime("Starting testing on the %s split" % {split})
  model:evaluate()
  local valLoss = {}
  local numAcc=0
  local acc=0
  local dataSize = 0
  local batch = loader:nextBatch(split)

  while batch~=nil do 

    batchSize=batch.data:size(1)
    dataSize=dataSize+batchSize

    batch.data=batch.data:cuda()
    batch.label=batch.label:cuda()

    for i=1,batchSize do
      local prediction = model:forward(batch.data[i])
      local confidence,indice = torch.sort(prediction,true)
      local pre_value=indice[-1][1]
      local realLabel = batch.label[i][-1]
      local val_loss = criterion:forward(prediction,batch.label[i])
      if pre_value==realLabel then
        numAcc=numAcc+1
      end
      table.insert(valLoss, val_loss)
    end

    batch = loader:nextBatch(split)

  end
  local epochValLoss = torch.mean(torch.Tensor(valLoss))
  print('epochValLoss ',epochValLoss)
  print('the arrurate number is ',numAcc)
  acc=numAcc/dataSize
  print('the accuracy is ',acc)

  model:training()
  return epochValLoss,acc
end

function test(model)
  local split='test'
  utils.printTime("Starting testing on the %s split" % {split})
  local evalData = {}
  evalData.predictions={}
  evalData.scores={}
  evalData.labels={}
  local numAcc=0
  local dataSize = 0
  evalData.loss = 0 -- sum of losses
  evalData.numBatches = 0 -- number of batches run
  local batch = loader:nextBatch(split)
  model:evaluate()
  model:cuda()
  while batch~=nil do 

    batchSize=batch.data:size(1)
    dataSize=dataSize+batchSize

    batch.data=batch.data:cuda()
    batch.label=batch.label:cuda()

    for i=1,batchSize do
      local prediction = model:forward(batch.data[i])
      local confidence,indice = torch.sort(prediction,true)
      local score= confidence[-1][1]
      local pre_value=indice[-1][1]
      local realLabel = batch.label[i][-1]
      if pre_value==realLabel then
        numAcc=numAcc+1
      end
      table.insert(evalData.scores,score)
      table.insert( evalData.predictions , pre_value )
      table.insert( evalData.labels, realLabel)
    end

    batch = loader:nextBatch(split)

  end
  model:training()
  print('the arrurate number is ',numAcc)
  print('the accuracy is ',(numAcc/dataSize))
  --print("the tensor of predictions is : ",evalData.predictions:size())
  -- local predicts=torch.cat(evalData.predictions)

  local preLength = #evalData.predictions
  local labelLength = #evalData.labels
  print("predicts size is ",preLength)

  local predictLabels=torch.Tensor(preLength)
  local trueLabels = torch.Tensor(preLength)
  local predScores=torch.Tensor(preLength)
  for i=1,preLength do
    predScores[i]=evalData.scores[i]
    predictLabels[i]=evalData.predictions[i]
    trueLabels[i]=evalData.labels[i]
  end
  -- save results to file
  filename='predict_LRCL_classifier_'..slidingWindowLength..'_date_'..os.date("%m_%d")..'_'..os.time()..'.h5'
  local myFile=hdf5.open(filename,'w')
  myFile:write('predicts',predictLabels:float())
  myFile:write('labels',trueLabels:float())
  myFile:write('scores',predScores:float())
  myFile:close()
  -- then exec the score.py
  os.execute('python calculate_LRCL.py '..filename..' LRCL'..slidingWindowLength)
end

train(model)
print("\n-----train have done----\n")
test(model)

local ck=torch.load('best_model_LRCL_classifier_'..slidingWindowLength..'.t7')
local bestModel=ck.model
test(bestModel)

