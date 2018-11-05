# Updated/fixed version from Gene Kogan's "machine learning for artists" collection - ml4a.github.io

import time
import os
import re
import random
import argparse
import urllib2
import itertools
import bs4
from bs4 import BeautifulSoup
import multiprocessing
from multiprocessing.dummy import Pool

genre_list = ['portrait', 'landscape', 'genre-painting', 'abstract', 'religious-painting', 
              'cityscape', 'sketch-and-study', 'figurative', 'illustration', 'still-life', 
              'design', 'nude-painting-nu', 'mythological-painting', 'marina', 'animal-painting', 
              'flower-painting', 'self-portrait', 'installation', 'photo', 'allegorical-painting', 
              'history-painting', 'interior', 'literary-painting', 'poster', 'caricature', 
              'battle-painting', 'wildlife-painting', 'cloudscape', 'miniature', 'veduta', 
              'yakusha-e', 'calligraphy', 'graffiti', 'tessellation', 'capriccio', 'advertisement', 
              'bird-and-flower-painting', 'performance', 'bijinga', 'pastorale', 'trompe-loeil', 
              'vanitas', 'shan-shui', 'tapestry', 'mosaic', 'quadratura', 'panorama', 'architecture']

style_list = ['impressionism', 'realism', 'romanticism', 'expressionism', 
            'post-impressionism', 'surrealism', 'art-nouveau', 'baroque', 
            'symbolism', 'abstract-expressionism', 'na-ve-art-primitivism', 
            'neoclassicism', 'cubism', 'rococo', 'northern-renaissance', 
            'pop-art', 'minimalism', 'abstract-art', 'art-informel', 'ukiyo-e', 
            'conceptual-art', 'color-field-painting', 'high-renaissance',
            'mannerism-late-renaissance', 'neo-expressionism', 'early-renaissance', 
            'magic-realism', 'academicism', 'op-art', 'lyrical-abstraction', 
            'contemporary-realism', 'art-deco', 'fauvism', 'concretism', 
            'ink-and-wash-painting', 'post-minimalism', 'social-realism', 
            'hard-edge-painting', 'neo-romanticism', 'tachisme', 'pointillism', 
            'socialist-realism', 'neo-pop-art']

parser = argparse.ArgumentParser()
parser.add_argument("--genre", help="which genre to scrape", choices=genre_list, default=None)
parser.add_argument("--style", help="which style to scrape", choices=style_list, default=None)
parser.add_argument("--num_pages", type=int, help="number of pages to scrape (leave blank to download all of them)", default=1000)
parser.add_argument("--output_dir", help="where to put output files")

num_downloaded = 0
num_images = 0




def get_painting_list(count, typep, searchword):
    try:
        time.sleep(3.0*random.random())  # random sleep to decrease concurrence of requests
        url = "https://www.wikiart.org/en/paintings-by-%s/%s/%d"%(typep, searchword, count)
        soup = BeautifulSoup(urllib.request.urlopen(url), "lxml")
        regex = r'https?://uploads[0-9]+[^/\s]+/\S+\.jpg'
        url_list = re.findall(regex, str(soup.html()))
        count += len(url_list)
        return url_list
    except Exception as e:
        print('failed to scrape %s'%url, e)


def downloader(link, genre, output_dir):
    global num_downloaded, num_images
    item, file = link
    filepath = file.split('/')
    #savepath = '%s/%s/%d_%s' % (output_dir, genre, item, filepath[-1])
    savepath = '%s/%s/%s' % (output_dir, genre, filepath[-1])    
    try:
        time.sleep(0.2)  # try not to get a 403
        urllib.request.urlretrieve(file, savepath)
        num_downloaded += 1
        if num_downloaded % 100 == 0:
            print('downloaded number %d / %d...' % (num_downloaded, num_images))
    except Exception as e:
        print("failed downloading " + str(file), e) 


def main(typep, searchword, num_pages, output_dir):
    global num_images
    print('gathering links to images... this may take a few minutes')
    threadpool = Pool(multiprocessing.cpu_count()-1)
    numbers = list(range(1, num_pages))
    wikiart_pages = threadpool.starmap(get_painting_list, zip(numbers, itertools.repeat(typep), itertools.repeat(searchword))) 
    threadpool.close()
    threadpool.join()

    pages = [page for page in wikiart_pages if page ]
    items = [item for sublist in pages for item in sublist]
    items = list(set(items))  # get rid of duplicates
    num_images = len(items)
    
    if not os.path.isdir('%s/%s'%(output_dir, searchword)):
        os.mkdir('%s/%s'%(output_dir, searchword))
    
    print('attempting to download %d images'%num_images)
    threadpool = Pool(multiprocessing.cpu_count()-1)
    threadpool.starmap(downloader, zip(enumerate(items), itertools.repeat(searchword), itertools.repeat(output_dir)))
    threadpool.close    
    threadpool.close()


if __name__ == '__main__':
    args = parser.parse_args()
    searchword, typep = (args.genre, 'genre') if args.genre is not None else (args.style, 'style')
    num_pages = args.num_pages
    output_dir = args.output_dir
    main(typep, searchword, num_pages, output_dir)
