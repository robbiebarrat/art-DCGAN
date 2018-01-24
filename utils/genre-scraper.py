# For a list of stuff you can scrape - go to https://www.wikiart.org/en/paintings-by-genre/ and look at the categories there. Usually you'll need about 1000 images for a good GAN.

import os
import urllib
import itertools


import bs4
from bs4 import BeautifulSoup

import multiprocessing
from multiprocessing.dummy import Pool

# how many pages you want to scrape, if unsure, just put 1000 that'll get all of them.
pages = 1000

# Change this value! 

genre_to_scrape = "landscape"

# genre_to_scrape can be any one of the below values.
"""
portrait
landscape
genre-painting
abstract
religious-painting
cityscape
sketch-and-study
figurative
illustration
still-life
design
nude-painting-nu
mythological-painting
marina
animal-painting
flower-painting
self-portrait
installation
photo
allegorical-painting
history-painting
"""
# at this point - there might not be enough images for good results - but if you'd like to mix and match the images you're training and pull from multiple genres, go ahead
"""
interior
literary-painting
poster
caricature
battle-painting
wildlife-painting
cloudscape
miniature
veduta
yakusha-e
calligraphy
graffiti
tessellation
capriccio
advertisement
bird-and-flower-painting
performance
bijinga
pastorale
trompe-loeil
vanitas
shan-shui
tapestry
mosaic
quadratura
panorama
architecture
"""



# get list of all links to paintings of the specified genre
def get_painting_list(count, genre=genre_to_scrape):
	try:
		url = "https://www.wikiart.org/en/paintings-by-genre/"+ genre+ "/" + str(count)
		soup =  BeautifulSoup(urllib.request.urlopen(url), "lxml")
		complete = 0
		url_list = []
		for item in str(soup.findAll()).split():
			if item == "data" or complete == 1:
				complete = 1
				if "}];" in item:
					break
				if "https" in item:
					link = "http" + item[6:-2]
					url_list.append(link)
					count += 1
		return url_list
	except Exception as e:
		#print(e)
		#print("Couldn't find page " + str(count) + " ... if you're seeing this at the beginning you're fine.")	
		pass


def downloader(link, genre):
	
	item,file = link	
	name=file.split('/')
	name_to_save_as = ''
	
	if len(name) == 5:
		name_to_save_as = genre + "/" + "images/" + name[4] + "+" + name[5].split('.')[0] +".jpg"
	if len(name) == 6:
		name_to_save_as = genre + "/" + "images/" + name[4].split('.')[0]+".jpg"
	if len(name) == 7:
		name_to_save_as = genre + "/" + "images/" + name[5] + "+" + name[6].split('.')[0] +".jpg"

	print(str(item) + " --- " + str(name_to_save_as))

	try:
	        urllib.request.urlretrieve(file,name_to_save_as)
	except Exception as e:
		print(e)
		print("failed downloading " + str(name_to_save_as))	

		

def main(genre): 
	pool_of_threads = Pool(multiprocessing.cpu_count() - 1) # lets hope you have more than 1 cpu core...

	numbers = list(range(1,pages))

	old_results = pool_of_threads.starmap( get_painting_list, zip( numbers, itertools.repeat(genre)) ) 
	
	pool_of_threads.close()
	pool_of_threads.join()

	results = []

	for item in old_results:
		if item:
			for x in item:
				results.append(x)

	pool_of_threads = Pool(multiprocessing.cpu_count() - 1)
	pool_of_threads.starmap(downloader, zip(enumerate(results), itertools.repeat(genre) ) )
	pool_of_threads.close	
	pool_of_threads.close()


if not os.path.exists("./" + genre_to_scrape):
	os.mkdir(genre_to_scrape)

if not os.path.exists("./" + genre_to_scrape + "/images"):
	os.mkdir(genre_to_scrape + "/images/")

print("Building a list of all the paintings to download.. This may take a few minutes.")
main(genre_to_scrape)




































	

