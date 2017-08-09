require 'lmdb'
require 'image'
tds=require 'tds'
ffi = require 'ffi'

list = {'bedroom'}

-- list = {'bedroom', 'bridge', 'church_outdoor', 'classroom', 'conference_room',
--         'dining_room', 'kitchen', 'living_room', 'restaurant', 'tower'}

root = os.getenv('DATA_ROOT') or os.getenv('HOME') .. '/local/lsun'

for i=1,#list do
   local name = list[i] .. '_train_lmdb'
   print('opening lmdb database: ', name)
   db = lmdb.env{Path=paths.concat(root, name), RDONLY=true}
   db:open()
   reader = db:txn(true)
   cursor = reader:cursor()
   hsh = tds.hash()

   count = 1
   local cont = true
   while cont do
      local key,data = cursor:get()
      hsh[count] = key
      print('Reading: ', count, '   Key:', key)
      count = count + 1
      if not cursor:next() then
          cont = false
      end
   end

   hsh2 = torch.CharTensor(#hsh, #hsh[1])
   for i=1,#hsh do ffi.copy(hsh2[i]:data(), hsh[i], #hsh[1]) end

   local indexfile = paths.concat(root, name .. '_hashes_chartensor.t7')
   torch.save(indexfile, hsh2)
   print('wrote index file at: ', indexfile .. ' with ' .. count .. ' keys')
end

print("you're all set")
