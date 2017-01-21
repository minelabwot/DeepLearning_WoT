require 'torch'
require 'hdf5'
local DataLoader = {}
DataLoader.__index = DataLoader

function DataLoader.create(batch_size)
    -- split_fractions is e.g. {0.9, 0.05, 0.05}

    local self = {}
    setmetatable(self, DataLoader)
    
    self.batch_size = batch_size
    self.splits={
        train={},
        val={},
        test={}
    }

    for split, _ in pairs(self.splits) do
        self.splits[split].batch_idx = 1
        local myFile = hdf5.open(split..'.h5','r')
        local dim = myFile:read('data'):dataspaceSize()
        myFile:close()

        print("the size of ".. split.." is ",dim)
        self.splits[split].sample_size = dim[1]
        print("the example size of "..split.." is ",self.splits[split].sample_size)
    end
    

    print(string.format('data have read once for the size.'))

    collectgarbage()
    return self
end


function DataLoader:nextBatch(split)
    assert(split == 'train' or split == 'test' or split == 'val')
    local dataset=self.splits[split]
    --batch has done in preprocssing
    local start=dataset.batch_idx
    local start_end=start+self.batch_size-1
    
    if(start%10==0)
        print("batch_index is ",start)
    end
    -- set the deadline
    if(start_end>=dataset.sample_size) then
        self.splits[split].batch_idx=1
        return nil
    else

        local myFile=hdf5.open(split..'.h5','r')
        local myFileData={data=myFile:read('data'):partial({start,start_end},{1,30},{1,1},{1,24},{1,113}),label=myFile:read('label'):partial({start,start_end})}
        myFile:close()

        self.splits[split].batch_idx=start_end+1

        activityData={}
        activityLabel={}

        -- for i=1,self.batch_size do
        --     local mydata=myFileData.data[i]
        --     -- local mylabel=myFileData.label[i]
        --     table.insert(activityData,mydata)
        --     -- table.insert(activityLabel,mylabel)
        -- end
        -- table can't call method 'size()'
        -- print('activityData shape is ',#activityData)
        --activityData[i] is torch.FloatTensor of size 30x1x24x113
        -- but cat() method expected DoubleTensor in tensor array
        --local aaa = torch.cat(activityData,1):type('torch.DoubleTensor')
        ---print("aaa is ",aaa),torch.DoubleTensor of size 3840x1x24x113
        -- local bbb = torch.cat(myFileData.data,1):type('torch.DoubleTensor')
        -- print(bbb)

        local batch={
            data=myFileData.data,
            label=myFileData.label
            -- label=torch.cat(activityLabel,1),
        }

        setmetatable(batch, 
          {__index = function(t, k) 
                          return {t.data[k], t.label[k]} 
                      end}
        );

        function batch:size() 
          return self.label:size(1)
        end

        return batch
    end
end

function DataLoader:getTestSet()
    local myTestFile=hdf5.open('test.h5','r')
    local testSet = {data=myTestFile:read('data'):all(),label=myTestFile:read('label'):all()}
    myTestFile:close()
    print("data size is ",testSet.data:size())
    print("label size is ",testSet.label:size())
    return testSet
end

return DataLoader

