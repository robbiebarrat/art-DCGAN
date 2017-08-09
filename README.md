# Cool-DCGAN
Modified version of [Soumith Chintala's torch implementation](https://github.com/soumith/dcgan.torch) of [DCGAN](https://arxiv.org/pdf/1511.06434.pdf). 

## The most notable changes are:
* Doubled image size - now 128x128 instead of 64x64 (added a layer in both networks)

* Ability to resume training from checkpoints (simply pass -netG=[path_to_network], and -netD=[path_to_network])

* Included a horribly simple python script that will keep your checkpoint folder empty - it is meant for leaving running when you're training a GAN for a while, because sometimes I would come back to my computer in the morning after an overnight GAN session and have a hard drive with zero free bytes of disk space and a crashed GAN...

* The inclusion of multiple pre-trained GAN's (.t7 files) that can generate various types of images, including 128x128 landscape oil paintings, 128x128 nude oil paintings, and others highlighted below.

## Usage:
The usage is identical to Soumith's - with the exception of loading from a checkpoint.

### Train a GAN on a folder of images

#### Start Training ####
`
DATA_ROOT=myimages dataset=folder ndf=50 ngf=150 th main.lua
`

You can adjust ndf (number of filters in discriminator's first layer) and ngf (number of filters in generator's first layer) freely, although it's reccomended that the generator has ~2x the filters as the discriminator to prevent the discriminator from beating the generator out, since the generator has a much much harder job.

Keep in mind, you can also pass these arguments when training:
```
batchSize=64              -- Batchsize - didn't get very good results with this over 128...
noise=normal, uniform     -- pass ONE Of these. It seems like normal works a lot better, though.
nz=100                    -- number of dimensions for Z
nThreads=1                -- number of data loading threads
gpu=1                     -- gpu to use
name=experiment1          -- just to make sure you don't overwrite anything cool, change the checkpoint filenames with this
```


#### Resume from checkpoint ####
`
DATA_ROOT=myimages dataset=folder netD=checkpoints/your_discriminator_net.t7 netG=your_driscriminator_net.t7 th main.lua
`
Passing ndf and ngf will have no effect here - as the networks are loaded from the checkpoints.

#### Generate images with a pre-trained net ####
`
net=your_generator_net.t7 th generate.lua
`
Very straightforward... I hope. Keep in mind; you can also pass these when generating images:
```
batchSize=36                      -- How many images to generate - keep a multiple of six for unpadded output.
imsize=1                          -- How large the image(s) should be (not in pixels!)
noisemode=normal, line, linefull  -- pass ONE of these. If you pass line, pass batchSize > 1 and imsize = 1, too.
name=generation1                  -- just to make sure you don't overwrite anything cool, change the filename with this
```
####There are more passable arguments on [the unmodified network's page](https://github.com/soumith/dcgan.torch#all-training-options) - I think I included the more important ones here though####


## Pre-trained networks ##
Due to the nature of github, and the 100+ MB nature of the pre-trained networks, you'll have to click a link to get the pre-trained models, but it's worth it. Below are some of them and examples of what they can generate.


#### I'll add this in a second... ####
