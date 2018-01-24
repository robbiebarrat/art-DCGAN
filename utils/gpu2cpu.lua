require "torch"
require 'cutorch'
require 'cunn'
require 'cudnn'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Convert a GPU checkpoint to CPU checkpoint.')
cmd:text()
cmd:text('Options')
cmd:argument('-model','GPU model checkpoint to convert')
cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')
cmd:text()

-- parse input params
local opt = cmd:parse(arg)

torch.setdefaulttensortype('torch.FloatTensor')
cutorch.setDevice(opt.gpuid + 1)

local checkpoint = torch.load(opt.model)
cudnn.convert(checkpoint, nn)
checkpoint = checkpoint:float()

local savefile = paths.dirname(opt.model)..'/'.. paths.basename(opt.model, ".t7")..'_cpu.t7'
torch.save(savefile, checkpoint)
print('saved ' .. savefile)
