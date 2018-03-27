require 'torch'
require 'hdf5'
local DataLoader = {}
DataLoader.__index = DataLoader

function DataLoader.create(batch_size,windowLen)

    local self = {}
    setmetatable(self, DataLoader)
    
    self.batch_size = batch_size
    self.windowLen=windowLen
    self.splits={
        train={},
        val={},
        test={}
    }

    for split, _ in pairs(self.splits) do
        print("the set is ",split)
        self.splits[split].batch_idx = 1
        local myFile
        if(split=='train') then
            myFile = hdf5.open(split..'_LRCL_informed_window'..self.windowLen..'_seed1.h5','r')
        else
            myFile = hdf5.open(split..'_LRCL_informed_window'..self.windowLen..'.h5','r')
        end
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

function DataLoader:getLableDim( split )

    print("the set is ",split)
    local myFile = hdf5.open(split..'_LRCL_informed_window'..self.windowLen..'_seed1.h5','r')
    local dim = myFile:read('label'):dataspaceSize()
    myFile:close()

    return dim
end

function DataLoader:nextBatch(split,seed)
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

        local myFileData = nil
        local myFile=nil
        if(split=='train') then
            myFile=hdf5.open(split..'_LRCL_informed_window'..self.windowLen..'_seed'..seed..'.h5','r')
        else            
            myFile=hdf5.open(split..'_LRCL_informed_window'..self.windowLen..'.h5','r')
        end
        myFileData={data=myFile:read('data'):partial({start,start_end},{1,20},{1,1},{1,self.windowLen},{1,48}),label=myFile:read('label'):partial({start,start_end},{1,20})}

        myFile:close()

        self.splits[split].batch_idx=start_end+1

        activityData={}
        activityLabel={}

        local batch={
            data=myFileData.data,
            label=myFileData.label
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

