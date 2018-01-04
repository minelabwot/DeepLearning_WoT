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
    
    -- set the deadline
    if(start_end>=dataset.sample_size) then
        self.splits[split].batch_idx=1
        return nil
    else

        local myFile=hdf5.open(split..'.h5','r')
        local myFileData={data=myFile:read('data'):partial({start,start_end},{1,20},{1,1},{1,24},{1,113}),label=myFile:read('label'):partial({start,start_end})}
        myFile:close()

        self.splits[split].batch_idx=start_end+1

        activityData={}
        activityLabel={}

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


return DataLoader

