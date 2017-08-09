require 'image'
require 'nn'
require 'qt'
local optnet = require 'optnet'
torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(1)
opt = {
    batchSize = 64,        -- number of samples to produce
    noisetype = 'normal',  -- type of noise distribution (uniform / normal).
    net = '',              -- path to the generator network
    imsize = 1,            -- used to produce larger images. 1 = 64px. 2 = 80px, 3 = 96px, ...
    noisemode = 'random',  -- random / line / linefull1d / linefull
    name = 'generation1',  -- name of the file saved
    gpu = 1,               -- gpu mode. 0 = CPU, 1 = GPU
    nz = 100,              
}
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)
if opt.display == 0 then opt.display = false end

assert(net ~= '', 'provide a generator model')

net = torch.load(opt.net)

-- for older models, there was nn.View on the top
-- which is unnecessary, and hinders convolutional generations.
if torch.type(net:get(1)) == 'nn.View' then
    net:remove(1)
end

print(net)

local sample_input = torch.randn(2,100,1,1)
if opt.gpu > 0 then
    require 'cunn'
    require 'cudnn'
    net:cuda()
    cudnn.convert(net, cudnn)
    sample_input = sample_input:cuda()
else
   sample_input = sample_input:float()
   net:float()
end

net:evaluate()

-- a function to setup double-buffering across the network.
-- this drastically reduces the memory needed to generate samples
optnet.optimizeMemory(net, sample_input)

local function regenerate()
   noise = torch.Tensor(opt.batchSize, opt.nz, opt.imsize, opt.imsize)
   noise:normal(0, 1)
   images = net:forward(noise)
   images:add(1):mul(0.5)
   for i=1, images:size(1) do
      images[i] = image.drawText(images[i], tostring(i), 3, 3, {color={255, 0, 0}, size=2})
   end
end

-- A - B + C
local A = {}
local B = {}
local C = {}

print("We will be doing vector arithmetic A - B + C")

local win

local function choose(name, tab)
   -- choose images
   print("Choose three images for " .. name)

   for i = 1, 3 do
      print('Choose image number. Enter -1 to regenerate new images: ')
      local choice = -1
      while choice == -1 do
	 regenerate()
	 win = image.display(images, nil, nil, nil, nil, win, nil, nil, true)
	 choice=tonumber(io.read())
      end
      print("Chosen image number " .. choice .. " for " .. name)
      table.insert(tab, noise[choice]:clone())
   end

   print("Images for " .. name .. " have been chosen")
end
choose("A", A)
choose("B", B)
choose("C", C)

if win then win.window:setHidden(true) end

print("Generating A - B + C")

local Aavg = (A[1] + A[2] + A[3]) / 3
local Bavg = (B[1] + B[2] + B[3]) / 3
local Cavg = (C[1] + C[2] + C[3]) / 3

local final_noise = Aavg - Bavg + Cavg

-- final display

-- place noise vectors in mini-batch
noise[1]:copy(A[1])
noise[8]:copy(A[2])
noise[15]:copy(A[3])
noise[22]:copy(Aavg)

noise[3]:copy(B[1])
noise[10]:copy(B[2])
noise[17]:copy(B[3])
noise[24]:copy(Bavg)

noise[5]:copy(C[1])
noise[12]:copy(C[2])
noise[19]:copy(C[3])
noise[26]:copy(Cavg)

noise[28]:copy(final_noise)

-- generate images
images = net:forward(noise)
images:add(1):mul(0.5)

-- insert + / - / = symbols
images[23]:fill(0)
images[25]:fill(0)
images[27]:fill(0)
images[23] = image.drawText(images[23], "-", 3, 3, {size=10, color={255, 0, 0}})
images[25] = image.drawText(images[25], "+", 3, 3, {size=10, color={255, 0, 0}})
images[27] = image.drawText(images[27], "=", 3, 3, {size=10, color={255, 0, 0}})

-- fill black to dummy boxes
for i=0,2 do
   images[2+7*i]:fill(0)
   images[4+7*i]:fill(0)
   images[6+7*i]:fill(0)
   images[7+7*i]:fill(0)
end

final_image = image.toDisplayTensor({input=images:narrow(1,1,28), nrow = 7, scaleeach=true})
image.save('arithmetic.png', final_image)
print("image saved to arithmetic.png")
image.display(final_image)
