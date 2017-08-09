--[[
    Copyright (c) 2015-present, Facebook, Inc.
    All rights reserved.

    This source code is licensed under the BSD-style license found in the
    LICENSE file in the root directory of this source tree. An additional grant
    of patent rights can be found in the PATENTS file in the same directory.
]]--

require 'image'
tds=require 'tds'
require 'lmdb'
ffi = require 'ffi'

-- This file contains the data-loading logic and details.
-- It is run by each data-loader thread.
------------------------------------------
-------- COMMON CACHES and PATHS
-- classes = {'bedroom', 'bridge', 'church_outdoor', 'classroom', 'conference_room',
--           'dining_room', 'kitchen', 'living_room', 'restaurant', 'tower'}
classes = {'bedroom'}
table.sort(classes)
print('Classes:')
for k,v in pairs(classes) do
   print(k, v)
end

-- Check for existence of opt.data
opt.data = os.getenv('DATA_ROOT') or os.getenv('HOME') .. '/local/lsun'
if not os.execute('cd ' .. opt.data) then
    error(("could not chdir to '%s'"):format(opt.data))
end

trainPath = paths.concat(opt.data, 'train')
valPath   = paths.concat(opt.data, 'val')

-----------------------------------------------------------------------------------------
if not os.execute('cd ' .. opt.data) then
    error(("could not chdir to '%s'"):format(opt.data))
end

local loadSize   = {3, opt.loadSize}
local sampleSize = {3, opt.fineSize}

local function loadImage(blob)
   local input = image.decompress(blob, 3, 'float')
   -- find the smaller dimension, and resize it to loadSize[2] (while keeping aspect ratio)
   local iW = input:size(3)
   local iH = input:size(2)
   if iW < iH then
      input = image.scale(input, loadSize[2], loadSize[2] * iH / iW)
   else
      input = image.scale(input, loadSize[2] * iW / iH, loadSize[2])
   end
   return input
end

--------------------------------------------------------------------------------
-- Hooks that are used for each image that is loaded

-- function to load the image, jitter it appropriately (random crops etc.)
local trainHook = function(path)
   collectgarbage()
   local input = loadImage(path)
   local iW = input:size(3)
   local iH = input:size(2)

   -- do random crop
   local oW = sampleSize[2];
   local oH = sampleSize[2]
   local h1 = math.ceil(torch.uniform(1e-2, iH-oH))
   local w1 = math.ceil(torch.uniform(1e-2, iW-oW))
   local out = image.crop(input, w1, h1, w1 + oW, h1 + oH)
   assert(out:size(2) == oW)
   assert(out:size(3) == oH)
   -- do hflip with probability 0.5
   if torch.uniform() > 0.5 then out = image.hflip(out); end
   out:mul(2):add(-1) -- make it [0, 1] -> [-1, 1]
   return out
end

--------------------------------------
-- trainLoader
print('initializing train loader')
trainLoader = {}
trainLoader.classes = classes
trainLoader.indices = {}
trainLoader.db = {}
trainLoader.db_reader = {}
for i=1,#classes do
   print('initializing: ', classes[i])
   trainLoader.indices[i] = torch.load(paths.concat(trainPath, classes[i]
                                                        .. '_train_lmdb_hashes_chartensor.t7'))
   trainLoader.db[i] = lmdb.env{Path=paths.concat(trainPath, classes[i] .. '_train_lmdb'),
                                RDONLY=true, NOLOCK=true, NOTLS=true, NOSYNC=true, NOMETASYNC=true,
                               MaxReaders=20, MaxDBs=20}
   trainLoader.db[i]:open()
   trainLoader.db_reader[i] = trainLoader.db[i]:txn(true)
end

local function getData(self, key, binary)
    self.mdb_key = lmdb.MDB_val(self.mdb_key, key, true)
    self.mdb_data = self.mdb_data or ffi.new('MDB_val[1]')
    if lmdb.errcheck('mdb_get', self.mdb_txn[0],
                     self.mdb_dbi[0], self.mdb_key,self.mdb_data) == lmdb.C.MDB_NOTFOUND then
        return nil
    else
        return lmdb.from_MDB_val(self.mdb_data, false, binary)
    end
end

function trainLoader:sample(quantity)
   local data = torch.Tensor(quantity, sampleSize[1], sampleSize[2], sampleSize[2])
   local label = torch.Tensor(quantity)
   for i=1, quantity do
      local class = torch.random(1, #self.classes)
      local index = torch.random(1, self.indices[class]:size(1))
      local hash = ffi.string(trainLoader.indices[class][index]:data(), trainLoader.indices[class]:size(2))
      local imgblob = getData(self.db_reader[class], hash, true)
      local out = trainHook(imgblob)
      data[i]:copy(out)
      label[i] = class
   end
   collectgarbage(); collectgarbage()
   return data, label
end

function trainLoader:size()
    if self._size then return self._size end
    local size = 0
    for i=1,#self.indices do
        size = size + self.indices[i]:size(1)
    end
    self._size = size
   return size
end
