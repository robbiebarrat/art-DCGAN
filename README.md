<p align="right">
<i>contact: robbiebarrat (at) gmail (dot) com</i>
</p>

# art-DCGAN
Modified version of [Soumith Chintala's torch implementation](https://github.com/soumith/dcgan.torch) of [DCGAN](https://arxiv.org/pdf/1511.06434.pdf) with a focus on generating artworks.




# Examples / Pre-trained networks
Due to the nature of github, and the 100+ MB nature of the pre-trained networks, you'll have to click a link to get the pre-trained models, but it's worth it. Below are some of them and examples of what they can generate.

### Abstract Landscape GAN
![Batch of Abstract Landscapes](https://raw.githubusercontent.com/robbiebarrat/art-DCGAN/master/abstractlandscapes.png)
#### Download the weights!
There is no download for abstract landscapes, yet. Scroll to the bottom to find out how to train your own from the regular landscapes network.

### Landscape GAN
![Batch of Landscapes](https://raw.githubusercontent.com/robbiebarrat/art-DCGAN/master/images/landscapenet_waifu2x.png)
#### Download the weights!
##### [Generator](https://drive.google.com/open?id=0B-_m9VM1w1bKUFBmV09VOWlmNG8) [(CPU)](https://drive.google.com/open?id=1HqoHuH1cefPN1A9i6j_CjbCuoQy_hpG1)

##### [Discriminator](https://drive.google.com/open?id=0B-_m9VM1w1bKaC1MRkNiMHp0VHM) [(CPU)](https://drive.google.com/open?id=1wz0Ke0TL8J9x2AkfpYTBII0UfO_OPVgH)

### Nude-Portrait GAN
![Batch of Nude-Portraits](https://raw.githubusercontent.com/robbiebarrat/art-DCGAN/master/images/nudenet_waifu2x.png)
#### Download the weights!
##### [Generator](https://drive.google.com/open?id=0B-_m9VM1w1bKdFJkdUFlNFRGRVE) [(CPU)](https://drive.google.com/open?id=1WfGaeEMNgMp355J194NnfRr8TqHSlvVd)

##### [Discriminator](https://drive.google.com/open?id=0B-_m9VM1w1bKUjdrckNQeGZqME0) [(CPU)](https://drive.google.com/open?id=1bOLqlIsXrOYYxfQnwgNQloCy4D0p7bBe)


### Portrait GAN
![Batch of Portraits](https://raw.githubusercontent.com/robbiebarrat/art-DCGAN/master/images/portraitnet_waifu2x.png)
#### Download the weights
##### [Generator](https://drive.google.com/open?id=0B-_m9VM1w1bKUXhmazg2eVF0bTA) [(CPU)](https://drive.google.com/open?id=1Ul9poUdKBSdyb6IAJlSXBP8zVOVpzHBT)

##### [Discriminator](https://drive.google.com/open?id=0B-_m9VM1w1bKMVh4S21BNlhzNEE) [(CPU)](https://drive.google.com/open?id=1KJMUW0sOZ3CRjshCEsJPd76DN4GT6Otv)

## The most notable changes are:
* Doubled image size - now 128x128 instead of 64x64 (added a layer in both networks)

* Ability to resume training from checkpoints (simply pass -netG=[path_to_network], and -netD=[path_to_network]). While this is convenient, it also allows for experimentation with training on one set of images, and then later in training shifting to another set of images. This allows you to train a landscape network, and then shift to abstract for a very short duration to get abstract landscapes (see example below in the "resume from checkpoint" section) - acting like a sort of style transfer for GANs.

* Included a horribly simple python script that will keep your checkpoint folder empty - it is meant for leaving running when you're training a GAN for a while, because sometimes I would come back to my computer in the morning after an overnight GAN session and have a hard drive with zero free bytes of disk space and a crashed GAN...

* Added a python 3 script (utils/genre-scraper.py) that allows easy image-scraping from wikiart into the format the GAN can draw from.

* Added a script (utils/gpu2cpu.lua) that converts checkpoints trained on a gpu to models that can be used by a cpu.

* The inclusion of multiple pre-trained GAN's (.t7 files) that can generate various types of images, including 128x128 landscape oil paintings, 128x128 nude oil paintings, and others highlighted below.

## Usage:
### Prerequisites:
See [INSTALL.md](INSTALL.md)

### General Usage:
The usage is identical to Soumith's - with the exception of loading from a checkpoint, and the fact that an artwork scraper is included with this project.

### Scraping Images from Wikiart
`genre-scraper.py` will allow you to scrape artworks from wikiart based on their genres. The usage is quite simple.
In `genre-scraper.py` there is a variable called `genre_to_scrape` - simply change that to any of the genre's listed on [this page](https://www.wikiart.org/en/paintings-by-genre/), or to any of the values in the huge list of comments right after `genre_to_scrape` is defined.

Run the program with python3 and a folder with the name of your genre will be created, with a subdirectory "images/" containing all of the jpgs. Point your GAN to the directory with the name of your genre (so if I did landscapes, i'd just change `genre_to_scrape` to "landscape", and then run my GAN with DATA_ROOT=landscape)

### Train a GAN on a folder of images

#### Start Training
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


#### Resume from checkpoint
`
DATA_ROOT=myimages dataset=folder netD=checkpoints/your_discriminator_net.t7 netG=your_driscriminator_net.t7 th main.lua
`
Passing ndf and ngf will have no effect here - as the networks are loaded from the checkpoints. Resuming from the checkpoint and training on different data can have very interesting effects. Below, a GAN trained on generating landscapes is trained on abstract art for half of an epoch.

![difference](https://raw.githubusercontent.com/robbiebarrat/art-DCGAN/master/images/difference.png)


#### Generate images with a pre-trained net
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



### Coming soon
* Pre-trained networks: flower paintings, cityscapes (comment in the open issue if you have suggestions!)
* creating animated gifs of walks in the space of samples
* a gui for this whole thing...
