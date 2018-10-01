
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import time
from scipy.misc import imread
from scipy.misc import imresize
import scipy.misc as misc 
import matplotlib.image as mpimg
import os
from scipy.ndimage import filters
import urllib
import urllib2



def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
    '''From:
    http://code.activestate.com/recipes/473878-timeout-function-using-threading/'''
    import threading
    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = None

        def run(self):
            try:
                self.result = func(*args, **kwargs)
            except:
                self.result = default

    it = InterruptableThread()
    it.start()
    it.join(timeout_duration)
    if it.isAlive():
        return False
    else:
        return it.result

testfile = urllib.URLopener()            


#Note: you need to create the uncropped folder first in order 
#for this to work

def rgb2gray(rgb):
    '''Return the grayscale version of the RGB image rgb as a 2D numpy array
    whose range is 0..1
    Arguments:
    rgb -- an RGB image, represented as a numpy array of size n x m x 3. The
    range of the values is 0..255
    '''
    
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray/255

    

def download_save_im(act, tra, val, tes):
    """
    download the images from URLs to the uncropped folder 
    and save the cropped images to the cropped folder.
    """
         
    if not os.path.isdir("uncropped"):
        os.makedirs("uncropped")
    if not os.path.isdir("cropped"):
        os.makedirs("cropped")  
        
    for a in act:
        name = a.split()[1].lower()
        i = 0
        cropped_i = 0
        for line in open("facescrub_actors.txt"): #get pictures of actors
            if a in line:
                filename = name+str(i)+'.'+line.split()[4].split('.')[-1]
                crop = line.split()[5].split(",")
                dim = [int(x) for x in crop]
                #A version without timeout (uncomment in case you need to 
                #unsupress exceptions, which timeout() does)
                #testfile.retrieve(line.split()[4], "uncropped/"+filename)
                #timeout is used to stop downloading images which take too long to download
                timeout(testfile.retrieve, (line.split()[4], "uncropped/"+filename), {}, 55)
                if not os.path.isfile("uncropped/"+filename):
                    continue
                else:
                    try:
                        image = imread("uncropped/"+filename)
                        if (len(image.shape) > 2):
                            cropped = image[dim[1]:dim[3]+1, dim[0]:dim[2]+1]
                            gray = rgb2gray(cropped)
                        else:
                            gray = image[dim[1]:dim[3]+1, dim[0]:dim[2]+1]
                        resized = misc.imresize(gray, (32, 32))
                        misc.imsave("cropped/" + filename, resized)
                        if cropped_i < 70:
                            tra[filename] = [name, "M"]
                        elif 70 <= cropped_i < 80:
                            val[filename] = [name, "M"]
                        elif 80 <= cropped_i < 90:
                            tes[filename] = [name, "M"]
                        cropped_i += 1
                    except:
                        print(filename + "cannot be read")
                
                print(filename)
                i += 1
                
        for line in open("facescrub_actresses.txt"): #get pictures of actresses
            if a in line:
                filename = name+str(i)+'.'+line.split()[4].split('.')[-1]
                crop = line.split()[5].split(",")
                dim = [int(x) for x in crop]            
                #A version without timeout (uncomment in case you need to 
                #unsupress exceptions, which timeout() does)
                #testfile.retrieve(line.split()[4], "uncropped/"+filename)
                #timeout is used to stop downloading images which take too long to download
                timeout(testfile.retrieve, (line.split()[4], "uncropped/"+filename), {}, 55)
                if not os.path.isfile("uncropped/"+filename):
                    continue
                else:
                    try:
                        image = imread("uncropped/"+filename)
                        if (len(image.shape) > 2):
                            cropped = image[dim[1]:dim[3]+1, dim[0]:dim[2]+1]
                            gray = rgb2gray(cropped)
                        else:
                            gray = image[dim[1]:dim[3]+1, dim[0]:dim[2]+1]
                        resized = misc.imresize(gray, (32, 32))
                        misc.imsave("cropped/" + filename, resized)
                        if cropped_i < 70:
                            tra[filename] = [name, "F"]
                        elif 70 <= cropped_i < 80:
                            val[filename] = [name, "F"]
                        elif 80 <= cropped_i < 90:
                            tes[filename] = [name, "F"]
                        cropped_i += 1
                    except:
                        pass                     
                print(filename)
                i += 1  
                
def download_save_im_part5(act, tes):
    """
    Used in part 5 for the six actors not in act. Only ten images per each actor are saved.
    """
        
    if not os.path.isdir("uncropped5"):
        os.makedirs("uncropped5")
    if not os.path.isdir("cropped5"):
        os.makedirs("cropped5")     
        
    for a in act:
        name = a.split()[1].lower()
        i = 0
        cropped_i = 0
        for line in open("facescrub_actors.txt"): #get pictures of actors
            if a in line:
                filename = name+str(i)+'.'+line.split()[4].split('.')[-1]
                crop = line.split()[5].split(",")
                dim = [int(x) for x in crop]
                #A version without timeout (uncomment in case you need to 
                #unsupress exceptions, which timeout() does)
                #testfile.retrieve(line.split()[4], "uncropped/"+filename)
                #timeout is used to stop downloading images which take too long to download
                timeout(testfile.retrieve, (line.split()[4], "uncropped5/"+filename), {}, 30)
                if not os.path.isfile("uncropped5/"+filename):
                    continue
                else:
                    try:
                        image = imread("uncropped5/"+filename)
                        if (len(image.shape) > 2):
                            cropped = image[dim[1]:dim[3]+1, dim[0]:dim[2]+1]
                            gray = rgb2gray(cropped)
                        else:
                            gray = image[dim[1]:dim[3]+1, dim[0]:dim[2]+1]
                        resized = misc.imresize(gray, (32, 32))
                        #fname = filename.split(".")[0]
                        misc.imsave("cropped5/" + filename, resized)
                        if cropped_i < 10:
                            tes[filename] = [name, "M"]
                        else:
                            break
                        cropped_i += 1
                    except:
                        print(filename + "cannot be read")
                
                print(filename)
                i += 1
                
        for line in open("facescrub_actresses.txt"): #get pictures of actresses
            if a in line:
                filename = name+str(i)+'.'+line.split()[4].split('.')[-1]
                crop = line.split()[5].split(",")
                dim = [int(x) for x in crop]            
                #A version without timeout (uncomment in case you need to 
                #unsupress exceptions, which timeout() does)
                #testfile.retrieve(line.split()[4], "uncropped/"+filename)
                #timeout is used to stop downloading images which take too long to download
                timeout(testfile.retrieve, (line.split()[4], "uncropped5/"+filename), {}, 30)
                if not os.path.isfile("uncropped5/"+filename):
                    continue
                else:
                    try:
                        image = imread("uncropped5/"+filename)
                        if (len(image.shape) > 2):
                            cropped = image[dim[1]:dim[3]+1, dim[0]:dim[2]+1]
                            gray = rgb2gray(cropped)
                        else:
                            gray = image[dim[1]:dim[3]+1, dim[0]:dim[2]+1]
                        resized = misc.imresize(gray, (32, 32))
                        misc.imsave("cropped5/" + filename, resized)
                        if cropped_i < 10:
                            tes[filename] = [name, "F"]
                        else:
                            break
                        cropped_i += 1
                    except:
                        pass                     
                print(filename)
                i += 1
    

if __name__ == "__main__":
    
    act = ['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']
        
    download_save_im(act, training, validation, test)