import astropy
from astropy.io import fits
import numpy as np 
import matplotlib.pyplot as plt
import math
from math import sin, cos, sqrt, pi
import IPython
import warnings
from time import time
import glob 


def coords_convert(img, x, y):
    """
    Convert given (x,y) in pixels to +VE longitude and latitude. 

    IMPT: x goes from 0 to 1600, y goes from 0 to 2800. (0,0) at lower left corner of img.

    """
    assert x <= 1600 and x >= 0, 'Error; x value out of bounds.'
    assert y <= 2800 and y >= 0, 'Error; y value out of bounds.'
    assert type(x) == int, 'Error; x not an integer.'
    assert type(y) == int, 'Error; y not an integer.'

    hdulist = fits.open(img) #open image

    #convert coordinates with header info
    lon_step, lat_step = hdulist[0].header['LON_STEP'], hdulist[0].header['LAT_STEP']
    lon_left, lat_bot = hdulist[0].header['LON_LEFT'], hdulist[0].header['LAT_BOT']
    lon, lat = lon_left + lon_step * x, lat_bot + lat_step * y
    
    #mod 360 for determined coords. lon will not be over 360 here.
    if lon < 0:
        lon += 360
    return [lon, lat]

def lon_justify(lon): 
    '''
    This function will justify the longitude system between 0 and 360     
    '''

    if lon < 0: lon += 360 
    if lon > 360: lon -=360

    return lon

def lon_interval(lon,lon_left,lon_right): 
    '''
    This function will compute if a longitude falls in between a longitude range 
    Returns: Does it fall inbetween, Does it wrap around   
    '''

    # check if lon_left is larger than the lon_right 
    # Example: lon_left 250, lon_right 100 lon_interval(170,250,100)

    # Round to two decimal points 
    lon     = np.around(lon,2)
    lon_left = np.around(lon_left,2) 
    lon_right = np.around(lon_right,2) 

    if lon_left > lon_right: 
        if (lon <= lon_left) and (lon>=lon_right):
            return True, False
        else: 
            return False, False 

    # Example: lon_left 10, lon_right 300 lon_interval(10,300,320) 

    if lon_left < lon_right: 
        if (lon <= lon_left) or (lon>=lon_right):
            return True, True 
        else: 
            return False, False 

            
def inverse_coords(img, lon, lat):
    """
    Convert given longitude and latitude to (x,y) in pixels.

    IMPT: x goes from 0 (1st pixel) to 1600 (1601th pixel), y goes from 0 (1st pixel) to 2800 (2801th pixel). (0,0) at lower left corner of img.

    """

    # Make sure that longitude range is within -180 to 180 
    lon = lon_justify(lon)

    hdulist = fits.open(img) #open image

    #header info
    lon_step, lat_step = hdulist[0].header['LON_STEP'], hdulist[0].header['LAT_STEP']
    #lon_left, lon_right = round(hdulist[0].header['LON_LEFT'], 2), round(hdulist[0].header['LON_RIGH'], 2)
    lon_left, lon_right = lon_justify(hdulist[0].header['LON_LEFT']), lon_justify(hdulist[0].header['LON_RIGH'])
    lat_bot, lat_top = hdulist[0].header['LAT_BOT'], hdulist[0].header['LAT_TOP']

    # Check if the longitude falls within the region 
    if not lon_interval(lon,lon_justify(lon_left),lon_justify(lon_right))[0]:
        print(f'Longitude { lon}, LoLe {lon_justify(lon_left)}, LoRi {lon_justify(lon_right)} ')
        warnings.warn('Longitude falls outside of the image range')
        return [np.nan, np.nan]

    # shift the values by 360 degrees if longitude right 
    img_lons = np.arange(lon_left,lon_right+lon_step, lon_step)
    img_lons_dist = abs(img_lons - lon)
    x = np.argmin(img_lons_dist)

    #img_lats = np.linspace(lat_bot, lat_top + lat_step, 2801, endpoint=False)
    img_lats = np.arange(lat_bot,lat_top + lat_step, lat_step)
    img_lats_dist = abs(img_lats - lat)
    y = np.argmin(img_lats_dist)

    return [x, y]






#OVERLAP 

#def overlap_img(img1, img2):
    """
    Determine the overlapping POSITIVE longitude range between an image pair. img1 and img2 are strings. 
    This assumes that the lengths of the longitude ranges for all trim images are the same. 

    --- +VE, +VE (TRIVIAL)
    --- -VE, -VE -> MOD BOTH 
    --- PARTIAL, PARTIAL -> MOD BOTH
    --- -VE, PARTIAL -> MOD BOTH
    --- +VE, -VE -> MOD BOTH IF SEPARATION IS < 80, MOD -VE IF > 280 DEGREES. OTHERWISE RETURN FALSE.
    --- +VE, PARTIAL -> MOD BOTH IF SEPARATION IS < 80, MOD PARTIAL IF > 280. OTHERWISE RETURN FALSE. 

    """
    hdulist1, hdulist2 = fits.open(img1), fits.open(img2) #open images
    lon_left_1, lon_left_2 = hdulist1[0].header['LON_LEFT'], hdulist2[0].header['LON_LEFT'] 
    lon_right_1, lon_right_2 = hdulist1[0].header['LON_RIGH'], hdulist2[0].header['LON_RIGH'] #get lon ranges

    #cases
    if lon_left_1 < 0 and lon_left_2 < 0:
        lon_left_1, lon_left_2, lon_right_1, lon_right_2 = lon_left_1 + 360, lon_left_2 + 360, lon_right_1 + 360, lon_right_2 + 360
    
    elif (lon_left_1 > 0 and lon_right_1 < 0) and (lon_left_2 > 0 and lon_right_2 < 0):
        lon_left_1, lon_left_2, lon_right_1, lon_right_2 = lon_left_1 + 360, lon_left_2 + 360, lon_right_1 + 360, lon_right_2 + 360
    
    elif lon_left_1 < 0 and (lon_left_2 > 0 and lon_right_2 < 0):
        lon_left_1, lon_left_2, lon_right_1, lon_right_2 = lon_left_1 + 360, lon_left_2 + 360, lon_right_1 + 360, lon_right_2 + 360
    elif lon_left_2 < 0 and (lon_left_1 > 0 and lon_right_1 < 0):
        lon_left_1, lon_left_2, lon_right_1, lon_right_2 = lon_left_1 + 360, lon_left_2 + 360, lon_right_1 + 360, lon_right_2 + 360

    elif lon_right_1 > 0 and lon_left_2 < 0:
        if abs(lon_right_1 - lon_right_2) < 80:
            lon_left_1, lon_left_2, lon_right_1, lon_right_2 = lon_left_1 + 360, lon_left_2 + 360, lon_right_1 + 360, lon_right_2 + 360
        elif abs(lon_right_1 - lon_right_2) > 280:
            lon_left_2, lon_right_2 = lon_left_2 + 360, lon_right_2 + 360
        else:
            return False
    elif lon_right_2 > 0 and lon_left_1 < 0:
        if abs(lon_right_1 - lon_right_2) < 80:
            lon_left_1, lon_left_2, lon_right_1, lon_right_2 = lon_left_1 + 360, lon_left_2 + 360, lon_right_1 + 360, lon_right_2 + 360
        elif abs(lon_right_1 - lon_right_2) > 280:
            lon_left_1, lon_right_1 = lon_left_1 + 360, lon_right_1 + 360
        else:
            return False
    
    elif lon_right_1 > 0 and (lon_left_2 > 0 and lon_right_2 < 0):
        if abs(lon_right_1 - lon_right_2) < 80:
            lon_left_1, lon_left_2, lon_right_1, lon_right_2 = lon_left_1 + 360, lon_left_2 + 360, lon_right_1 + 360, lon_right_2 + 360
        elif abs(lon_right_1 - lon_right_2) > 280:
            lon_left_2, lon_right_2 = lon_left_2 + 360, lon_right_2 + 360
        else:
            return False
    elif lon_right_2 > 0 and (lon_left_1 > 0 and lon_right_1 < 0):
        if abs(lon_right_1 - lon_right_2) < 80:
            lon_left_1, lon_left_2, lon_right_1, lon_right_2 = lon_left_1 + 360, lon_left_2 + 360, lon_right_1 + 360, lon_right_2 + 360
        elif abs(lon_right_1 - lon_right_2) > 280:
            lon_left_1, lon_right_1 = lon_left_1 + 360, lon_right_1 + 360
        else:
            return False
    
    #determine overlap
    if lon_left_1 > lon_left_2:
        if lon_left_2 > lon_right_1:
            if lon_right_1 > lon_right_2:
                overlap = [lon_left_2, lon_right_1]
            else:
                overlap = [lon_left_2, lon_right_2]
        else:
            overlap = False
    elif lon_left_2 > lon_left_1:
        if lon_left_1 > lon_right_2:
            if lon_right_2 > lon_right_1:
                overlap = [lon_left_1, lon_right_2]
            else:
                overlap = [lon_left_1, lon_right_1]
        else:
            overlap = False
    else:
        overlap = [lon_left_1, lon_right_1]

    #mod back if possible
    return overlap

def overlap_slice(slice1, slice2):
    """
    Return (at least partially) POSITIVE longitude overlap between two slices in DEGREES. 
    Assumes both slices have the same length - namely 80 degrees. 

    --- +VE, +VE (TRIVIAL)
    --- -VE, -VE -> MOD BOTH 
    --- PARTIAL, PARTIAL -> MOD BOTH
    --- -VE, PARTIAL -> MOD BOTH
    --- +VE, -VE -> MOD BOTH IF SEPARATION IS < 80, MOD -VE IF > 280 DEGREES. OTHERWISE RETURN FALSE.
    --- +VE, PARTIAL -> MOD BOTH IF SEPARATION IS < 80, MOD PARTIAL IF > 280. OTHERWISE RETURN FALSE.
    Separation = abs(lon_right_1 - lon_right_2) = abs(lon_left_1 - lon_left_2)

    """
    lon_left_1, lon_left_2 = slice1[0], slice2[0]
    lon_right_1, lon_right_2 = slice1[len(slice1) - 1], slice2[len(slice2) - 1]
    #print(lon_left_1, lon_right_1, lon_left_2, lon_right_2)

    #cases 2 and 3
    if (lon_left_1 < 0 and lon_left_2 < 0) or ((lon_left_1 > 0 and lon_right_1 < 0) and (lon_left_2 > 0 and lon_right_2 < 0)):
        lon_left_1, lon_left_2, lon_right_1, lon_right_2 = lon_left_1 + 360, lon_left_2 + 360, lon_right_1 + 360, lon_right_2 + 360
    
    #case 4
    elif (lon_left_1 < 0 and (lon_left_2 > 0 and lon_right_2 < 0)) or (lon_left_2 < 0 and (lon_left_1 > 0 and lon_right_1 < 0)):
        lon_left_1, lon_left_2, lon_right_1, lon_right_2 = lon_left_1 + 360, lon_left_2 + 360, lon_right_1 + 360, lon_right_2 + 360
        
    #case 5
    elif lon_right_1 > 0 and lon_left_2 < 0:
        if abs(lon_right_1 - lon_right_2) < 80:
            lon_left_1, lon_left_2, lon_right_1, lon_right_2 = lon_left_1 + 360, lon_left_2 + 360, lon_right_1 + 360, lon_right_2 + 360
        elif abs(lon_right_1 - lon_right_2) > 280:
            lon_left_2, lon_right_2 = lon_left_2 + 360, lon_right_2 + 360
        else:
            return False
    elif lon_right_2 > 0 and lon_left_1 < 0:
        if abs(lon_right_1 - lon_right_2) < 80:
            lon_left_1, lon_left_2, lon_right_1, lon_right_2 = lon_left_1 + 360, lon_left_2 + 360, lon_right_1 + 360, lon_right_2 + 360
        elif abs(lon_right_1 - lon_right_2) > 280:
            lon_left_1, lon_right_1 = lon_left_1 + 360, lon_right_1 + 360
        else:
            return False
    
    #case 6
    elif lon_right_1 > 0 and (lon_left_2 > 0 and lon_right_2 < 0):
        if abs(lon_right_1 - lon_right_2) < 80:
            lon_left_1, lon_left_2, lon_right_1, lon_right_2 = lon_left_1 + 360, lon_left_2 + 360, lon_right_1 + 360, lon_right_2 + 360
        elif abs(lon_right_1 - lon_right_2) > 280:
            lon_left_2, lon_right_2 = lon_left_2 + 360, lon_right_2 + 360
        else:
            return False
    elif lon_right_2 > 0 and (lon_left_1 > 0 and lon_right_1 < 0):
        if abs(lon_right_1 - lon_right_2) < 80:
            lon_left_1, lon_left_2, lon_right_1, lon_right_2 = lon_left_1 + 360, lon_left_2 + 360, lon_right_1 + 360, lon_right_2 + 360
        elif abs(lon_right_1 - lon_right_2) > 280:
            lon_left_1, lon_right_1 = lon_left_1 + 360, lon_right_1 + 360
        else:
            return False
    
    #determine overlap. this section actually also works for slices that are different lengths. 
    if lon_left_1 > lon_left_2:
        if lon_left_2 > lon_right_1:
            if lon_right_1 > lon_right_2:
                overlap = [lon_left_2, lon_right_1]
            else:
                overlap = [lon_left_2, lon_right_2]
        else:
            overlap = False
    elif lon_left_2 > lon_left_1:
        if lon_left_1 > lon_right_2:
            if lon_right_2 > lon_right_1:
                overlap = [lon_left_1, lon_right_2]
            else:
                overlap = [lon_left_1, lon_right_1]
        else:
            overlap = False
    else:
        overlap = [lon_left_1, lon_right_1] #in this case both slices are the same

    #print(overlap)
    return overlap



def overlap_all(y, v, path2data='./'):
    """
    Determine all image pairs that have overlapping longitude ranges at a certain latitude (pix) after 
    advecting a certain velocity (m/s). 
    
    path2data = '/Users/chris/GDrive-UCB/Berkeley/Research/Jupiter/ZonalWind/Images/'
    
    im_pair = overlap_all(1000,50,path2data = path2data) 

    """
    #Create an array of all images and an empty 2D array of overlapping images 


    images, overlapping_images = [], []
    images = glob.glob(path2data + '*.fits')


    #Loop through all possible image pairs
    for i in images[:len(images)-1]:
        x = images.index(i) + 1
        images2 = images[x:]
        for j in images2:
            #Advect one img to the other, and determine whether they overlap
            hdulist2 = fits.open(j)
            lon_left_2, lon_right_2 = hdulist2[0].header['LON_LEFT'], hdulist2[0].header['LON_RIGH'] 
            lon_range_1, lon_range_2 = advection(i, j, y, v)[0], np.linspace(round(lon_left_2, 2), round(lon_right_2, 2) - 0.05, 1601, endpoint=False)          
            if not overlap_slice(lon_range_1, lon_range_2):
                continue
            overlapping_images.append([i, j]) 

    #Return result
    return overlapping_images
     
#TIME

def time_difference(img1, img2):
    """
    Determine (time of img2) - (time of img1). Can be negative. 

    """
    hdulist1, hdulist2 = fits.open(img1), fits.open(img2) #open images

    date1, date2 = int(hdulist1[0].header['DATE-OBS'][8:]), int(hdulist2[0].header['DATE-OBS'][8:])
    t1, t2 = hdulist1[0].header['TIME-OBS'], hdulist2[0].header['TIME-OBS']
    hours1, mins1, secs1 = int(t1[0:2]), int(t1[3:5]), int(t1[6:8])
    hours2, mins2, secs2 = int(t2[0:2]), int(t2[3:5]), int(t2[6:8]) #get dates / times, convert to int  

    #account for all possible cases
    if date1 > date2:
        diff1 = (24 + hours1) * 3600 + mins1 * 60 + secs1 #compute time since 00:00:00 on earlier date in secs
        diff2 = hours2 * 3600 + mins2 * 60 + secs2 #repeat 
    
    elif date2 > date1: 
        diff2 = (24 + hours2) * 3600 + mins2 * 60 + secs2 #compute time since 00:00:00 on earlier date in secs
        diff1 = hours1 * 3600 + mins1 * 60 + secs1 #repeat 
    
    else: #repeat for final case where dates are the same
        diff1 = (hours1 * 3600) + (mins1 * 60) + secs1
        diff2 = (hours2 * 3600) + (mins2 * 60) + secs2
        
    return diff2 - diff1 #return time difference in seconds

#def earlier(img1, img2):
    """
    Determine which of two images has an EARLIER observation time. 

    """
    x, y = img1[10], img2[10]
    if img1[11].isdigit():
        x += img1[11]
    if img2[11].isdigit():
        y += img2[11]
    if int(x) < int(y):
        return img1
    else:
        return img2   #filenames are of the form corrected_x.fits, with increasing x meaning later times

#ADVECTION / CORRELATION / OTHER

def advection(img1, img2, y, v): 
    """
    Advect a longitude range of IMG1 by v * time_difference(img1, img2). 
    If img1 is earlier than img2, the correct direction is rightward (delta_lon > 0). And vice versa. 
    
    img1 and img2 are filenames (strings), y is in pixels, and v is in m/s.

    """
    R_eq, R_po = 71492e3, 66854e+3 #Jovian equatorial and polar radii in meters

    lat = coords_convert(img1, 1000, y)[1] #convert y to latitude (img, x value don't matter for trim)
    lat *= (pi / 180) #convert to radians
    
    time_diff = time_difference(img1, img2)
    delta_x = time_diff * v #get DISPLACEMENT (i.e. could be -ve) travelled for v in meters

    R_lat = sqrt(((R_eq ** 2) * (cos(lat) ** 2)) + ((R_po ** 2) * (sin(lat) ** 2)))
    delta_lon = (delta_x * R_lat) / (R_eq * R_eq * cos(lat)) #compute longitude shift for v (radians) - could be -ve
    delta_lon *= (180 / pi) #convert to degrees
    #print(delta_lon)

    hdulist = fits.open(img1) #open img1
    lon_left, lon_right, lon_step = np.around(hdulist[0].header['LON_LEFT'],2), np.around(hdulist[0].header['LON_RIGH'],2), np.around(hdulist[0].header['LON_STEP'],2)

    #lon_range_init = np.linspace(round(lon_left, 2), round(lon_right, 2) + lon_step, 1601, endpoint=False)
    lon_range_init = np.linspace(lon_left, lon_right,hdulist[0].header['NAXIS1'] ) #get other slice (DEGREES LON)

    lon_range_shifted = lon_range_init - delta_lon #creating array of longitudes, shifting by delta_lon

    return [lon_range_shifted, delta_lon]

def correlation_image_pair(y, v, img1, img2):
    """
    Determine correlation at a given latitude y (pixels) at some velocity v (m/s) for ALL images. 
    (actual corr function will sum over all y in a bin, then all bins for all overlapping images - order doesn't
    matter)

    """

    hdulist1, hdulist2 = fits.open(img1), fits.open(img2) #open images
    lat = coords_convert(img1, 0, y)[1] #get latitude in degrees from y (image and x value don't matter)
 
    lon_left_1, lon_right_1, lon_step = np.around(hdulist1[0].header['LON_LEFT'],2), np.around(hdulist1[0].header['LON_RIGH'],2), np.around(hdulist1[0].header['LON_STEP'],2)
    lon_left_2 , lon_right_2, lon_step = np.around(hdulist2[0].header['LON_LEFT'],2), np.around(hdulist2[0].header['LON_RIGH'],2), np.around(hdulist2[0].header['LON_STEP'],2)

    # Advect the slice according to the velocity 
    slice1_lon, delta_lon = advection(img1, img2, y, v) #get SHIFTED SLICE of IMG1 SHIFTING TO IMG2 (DEGREES LON)
    #note slice2_shifted is NOT an advected array; the naming is simply to match that of slice1_shifted.
    #slice2_lon = np.arange(lon_left_2, lon_right_2 + lon_step, lon_step) #get other slice (DEGREES LON)
    slice2_lon = np.linspace(lon_left_2, lon_right_2, hdulist2[0].header['NAXIS1'] ) #get other slice (DEGREES LON)

    # Obtain the overlap region between the two slices 
    overlap = overlap_slice(slice1_lon, slice2_lon)

    if not overlap or np.abs(np.diff(overlap))<10:
        return np.nan #end the function body if no overlap is detected after advection

    # Interpolate both arrays on the same grid 
    N = (np.arange((overlap[0]/lon_step + 1)*lon_step, (overlap[-1]/lon_step - 1)*lon_step,lon_step))

    #get corresponding brightness values from pixel overlap range. data array is size 2801 x 1601; each value corresponds to a pixel.

    slice1 = hdulist1[0].data[y, :]
    slice2 = hdulist2[0].data[y, :]

    # # Nan filter for debugging on untrimmed files 
    idx = slice1==slice1
    slice1 = slice1[idx]
    slice1_lon = slice1_lon[idx]
    idy = slice2==slice2
    slice2 = slice2[idy]
    slice2_lon = slice2_lon[idy]

    # Interpolate onto overlap grid 

    # Interpolate all values after wrap 
    if np.any(slice1_lon<0):
        slice1_lon += 360 
    if np.any(slice2_lon<0):
        slice2_lon += 360 

    slice1_brightness = np.flip(np.interp(np.flip(N), np.flip(slice1_lon), np.flip(slice1)))
    slice2_brightness = np.flip(np.interp(np.flip(N), np.flip(slice2_lon), np.flip(slice2)))
        
    # Compute the correlation 
    n1_brightness, n2_brightness = len(slice1_brightness), len(slice2_brightness)
    assert n1_brightness == n2_brightness, 'Error - brightness slices different lengths.'

    sum1, sum2 = np.sum(slice1_brightness), np.sum(slice2_brightness)
    slice1_brightness_avg, slice2_brightness_avg = sum1, sum2  #compute average brightness value of each array

    product_avg = np.dot(slice1_brightness,slice2_brightness)

    slice1_brightness_sq_avg, slice2_brightness_sq_avg = np.sum(slice1_brightness**2), np.sum(slice2_brightness**2), 

    corr_numerator = product_avg - (slice1_brightness_avg * slice2_brightness_avg/n1_brightness)
    corr_denominator_1 = slice1_brightness_sq_avg - (((slice1_brightness_avg) ** 2)/n1_brightness)
    corr_denominator_2 = slice2_brightness_sq_avg - (((slice2_brightness_avg) ** 2)/n1_brightness)
    corr = corr_numerator / ((corr_denominator_1 * corr_denominator_2) ** 0.5) #calculate correlation


    return corr #return final correlation


def correlation(y, v):
    """
    Determine correlation at a given latitude y (pixels) at some velocity v (m/s) for ALL images. 
    (actual corr function will sum over all y in a bin, then all bins for all overlapping images - order doesn't
    matter)

    """
    # path2data = '/Users/chris/GDrive-UCB/Berkeley/Research/Jupiter/ZonalWind/Images/'
    # image1 = '190626_631_1300_reg_trim.fits' 
    # image2 = '190626_631_1340_reg_trim.fits'


    path2data = './'
    image1 = 'corrected_1.fits'
    image2 = 'corrected_2.fits'

    # overlapping_images = overlap_all(y, v)
    overlapping_images = [path2data + image1, path2data + image2]
    # overlapping_images = overlap_simple(y) 
    total_corr = 0
    
    #plt.figure()  
    for i in range(len(overlapping_images)-1): #loop thru overlapping image pairs at given lat and v. 
      
        img1, img2 = overlapping_images[i], overlapping_images[i+1]
        #print(img1, img2)

        hdulist1, hdulist2 = fits.open(img1), fits.open(img2) #open images
        lat = coords_convert(img1, 0, y)[1] #get latitude in degrees from y (image and x value don't matter)

        # Advect the slice according to the velocity 
        lon_left_1, lon_right_1, lon_step = lon_justify(hdulist1[0].header['LON_LEFT']), lon_justify(hdulist1[0].header['LON_RIGH']), hdulist1[0].header['LON_STEP']
        lon_left_2, lon_right_2, lon_step = lon_justify(hdulist2[0].header['LON_LEFT']), lon_justify(hdulist2[0].header['LON_RIGH']), hdulist2[0].header['LON_STEP']
        slice1_shifted , delta_lon = advection(img1, img2, y, v) #get SHIFTED SLICE of IMG1 SHIFTING TO IMG2 (DEGREES LON)
        #slice2_shifted = np.linspace(round(lon_left_2, 2), round(lon_right_2, 2) + lon_step, 1601, endpoint=False) #get other slice (DEGREES LON)
        slice2_shifted = np.arange(lon_justify(lon_left_2), lon_justify(lon_right_2), lon_step) #get other slice (DEGREES LON)
        #note slice2_shifted is NOT an advected array; the naming is simply to match that of slice1_shifted.
        # Obtain the overlap region between the two slices 


        print(v,delta_lon)
        # If delta_lon is negative, the slice is advected outside of the overlap region  
        # if delta_lon < 0: 
        #     overlap = overlap_slice(slice1_shifted +  delta_lon, slice2_shifted)
        # else: 
        #     overlap = overlap_slice(slice1_shifted - delta_lon, slice2_shifted) #get overlapping longitude boundaries (array)

        overlap = overlap_slice(slice1_shifted, slice2_shifted)

        # if not overlap:
        #     return 0 #end the function body if no overlap is detected after advection

        # Faster method for computing overlapping slice 
        # N = np.around(np.arange(np.around(overlap[0]/lon_step)*lon_step,overlap[1],lon_step),2)
        N = np.arange((overlap[0]/lon_step + 1)*lon_step,overlap[1],lon_step)
 
        slice1_shifted_pixels = np.arange(inverse_coords(img1, N[0] +  delta_lon, lat)[0], inverse_coords(img1, N[-1] +  delta_lon, lat)[0]  ,1,)
        slice2_shifted_pixels = np.arange(inverse_coords(img2, N[0] , lat)[0], inverse_coords(img2, N[-1] , lat)[0] , 1)

        # Debug 
        '''
        img2 = 'corrected_2.fits'  
        lat = -20 
        overlap = [302.80720387703735, 246.40000999998182]
        lon_step = -0.05 
        '''

        #sanity check
        assert len(slice1_shifted_pixels) == len(slice2_shifted_pixels), 'Error - overlap pixel ranges different lengths.'
        
        #get corresponding brightness values from pixel overlap range. data array is size 2801 x 1601; each value corresponds to a pixel.
        slice1_brightness = hdulist1[0].data[y, slice1_shifted_pixels[0]:slice1_shifted_pixels[len(slice1_shifted_pixels) - 1]]
        slice2_brightness = hdulist2[0].data[y, slice2_shifted_pixels[0]:slice2_shifted_pixels[len(slice2_shifted_pixels) - 1]]


        # Plot the slices 
 
        # plt.plot(slice1_brightness) 
        # plt.plot(slice2_brightness,label=f'v {v}')
        # plt.legend() 

        # Nan filter for debugging
        # idx = slice1_brightness==slice1_brightness
        # slice1_brightness = slice1_brightness[idx]
        # slice2_brightness = slice2_brightness[idx]
        # idy = slice2_brightness==slice2_brightness
        # slice1_brightness = slice1_brightness[idy]
        # slice2_brightness = slice2_brightness[idy]



        n1_brightness, n2_brightness = len(slice1_brightness), len(slice2_brightness)
        assert n1_brightness == n2_brightness, 'Error - brightness slices different lengths.'


        sum1, sum2 = np.sum(slice1_brightness), np.sum(slice2_brightness)
        slice1_brightness_avg, slice2_brightness_avg = sum1, sum2  #compute average brightness value of each array

        product_avg = np.dot(slice1_brightness,slice2_brightness)

        slice1_brightness_sq_avg, slice2_brightness_sq_avg = np.sum(slice1_brightness**2), np.sum(slice2_brightness**2), 


        # corr_numerator = product_avg - (n1_brightness * slice1_brightness_avg * slice2_brightness_avg)
        # corr_denominator_1 = slice1_brightness_sq_avg - (n1_brightness * ((slice1_brightness_avg) ** 2))
        # corr_denominator_2 = slice2_brightness_sq_avg - (n2_brightness * ((slice2_brightness_avg) ** 2))
        # corr = corr_numerator / ((corr_denominator_1 * corr_denominator_2) ** 0.5) #calculate correlation

        corr_numerator = product_avg - (slice1_brightness_avg * slice2_brightness_avg/n1_brightness)
        corr_denominator_1 = slice1_brightness_sq_avg - (((slice1_brightness_avg) ** 2)/n1_brightness)
        corr_denominator_2 = slice2_brightness_sq_avg - (((slice2_brightness_avg) ** 2)/n1_brightness)
        corr = corr_numerator / ((corr_denominator_1 * corr_denominator_2) ** 0.5) #calculate correlation

        print(corr) 

        total_corr += corr        

    return total_corr #return final correlation

def readZWP(plotting=False): 
    '''
    Read in the zonal wind profile from Josh's 2019 paper 

    
    ------------
    Return:
    latitude Planetographic  (deg) 
    E-W Wind speed (m/s) 
    '''



    path2wp = 'ZWP_j2016_PJ03.txt'

    A = np.loadtxt(path2wp) 
    if plotting: 
        fig, axs = plt.subplots(1, 1,figsize=(8,4))
        axs.plot(A[:,1],A[:,0],label='JT - ZWP')
        axs.set_ylabel('Latitude (deg)')
        axs.set_xlabel('Velocity (m/s)')
        axs.set_ylim([-60,60])

    return A[:,0],A[:,1] 

def v_max(y,path2data=None,plotting=True, vstep = 51):
    """
    Return velocity with maximum correlation in m/s at a particular latitude y (pix) for two images. Also displays a graph.

    """
    # CM Debugging
    if path2data is None:  
        path2data = './'

    # Obtain image pairs at velocity 0 only 
    im_pairs = overlap_all(y,0, path2data = path2data)

    


    vel_array = np.linspace(-500, 500, vstep)
    correlations = np.zeros((len(im_pairs),vstep))

    for i in range(len(im_pairs)):
        for j in range(len(vel_array)):
            correlations[i,j] = correlation_image_pair(y, vel_array[j], im_pairs[i][0], im_pairs[i][1])
        if plotting: 
            plt.plot(vel_array, correlations)
            #plt.ylim([1,min(correlations)])
            plt.ylabel('correlation')
            plt.xlabel('Velocity')
            plt.show()

    # Remove inf and nans 
    for i in range(len(im_pairs)): 
        if np.any(~np.isfinite(correlations[i,:] )): 
            correlations[i,:] = np.nan 

    return vel_array[np.argmax(np.nansum(correlations,axis=0))]


# Test the code


path2data = './'
#path2data = '/Users/chris/GDrive-UCB/Berkeley/Research/Jupiter/ZonalWind/Images/'

images = glob.glob(path2data+'*.fits')
image1 = images[0]

hdulist = fits.open(image1) 
# hdulist = fits.open('corrected_12.fits')
lat_bot, lat_top, lat_step = hdulist[0].header['LAT_BOT'], hdulist[0].header['LAT_TOP'], hdulist[0].header['LAT_STEP']
latitude = np.linspace(lat_bot,lat_top,int((lat_top-lat_bot)/lat_step) + 1)


lat = []
v_corr = [] 
for y in range(1600,2000,10):
    v = v_max(y,plotting=False)
    v_corr.append(v)

    print(f'Latitude {latitude[y]:2.2f} Velocity',v_corr[-1])

    lat.append(y) 




lat_zwp, zwp = readZWP() 
fig, axs = plt.subplots(1, 1,figsize=(8,4))
axs.plot(zwp,lat_zwp,label='JT - ZWP')
axs.plot(v_corr,latitude[lat],label='DP')
axs.set_ylabel('Latitude (deg)')
axs.set_xlabel('Velocity (m/s)')
axs.set_ylim([-60,60])
plt.show()