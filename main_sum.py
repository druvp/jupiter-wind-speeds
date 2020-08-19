import astropy
from astropy.io import fits
import numpy as np 
import matplotlib.pyplot as plt
import math
from math import sin, cos, sqrt, pi

#COORDS

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

def inverse_coords(img, lon, lat):
    """
    Convert given longitude and latitude to (x,y) in pixels.

    IMPT: x goes from 0 (1st pixel) to 1600 (1601th pixel), y goes from 0 (1st pixel) to 2800 (2801th pixel). (0,0) at lower left corner of img.

    """
    hdulist = fits.open(img) #open image

    #header info
    lon_step, lat_step = hdulist[0].header['LON_STEP'], hdulist[0].header['LAT_STEP']
    lon_left, lon_right = round(hdulist[0].header['LON_LEFT'], 2), round(hdulist[0].header['LON_RIGH'], 2)
    lat_bot, lat_top = hdulist[0].header['LAT_BOT'], hdulist[0].header['LAT_TOP']
    
    #account for the fact that the user may be inputting a longitude outside the range of the image
    if lon > lon_left:
        while lon > lon_left:
            lon = round(lon - 360, 2)
        if lon < lon_right:
            return 'Error; inputted longitude not in longitude range of ' + img 
    elif lon < lon_right:
        while lon < lon_right:
            lon = round(lon + 360, 2)
        if lon > lon_left:
            return 'Error; inputted longitude not in longitude range of ' + img
    
    #get values
    img_lons = np.linspace(lon_left, lon_right + lon_step, 1601, endpoint=False)
    img_lons_dist = abs(img_lons - lon)
    x = np.argmin(img_lons_dist)

    img_lats = np.linspace(lat_bot, lat_top + lat_step, 2801, endpoint=False)
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

def overlap_all(y, v):
    """
    Determine all image pairs that have overlapping longitude ranges at a certain latitude (pix) after 
    advecting a certain velocity (m/s). 

    """
    #Create an array of all images and an empty 2D array of overlapping images
    images, overlapping_images = [], []
    for i in range(1, 22):
        images.append('corrected_' + str(i) + '.fits')

    #Loop through all possible image pairs
    for i in images[:20]:
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
    R_eq, R_po = 7.15e+7, 6.69e+7 #Jovian equatorial and polar radii in meters

    lat = coords_convert(img1, 1000, y)[1] #convert y to latitude (img, x value don't matter for trim)
    lat *= (pi / 180) #convert to radians
    
    time_diff = time_difference(img1, img2)
    delta_x = time_diff * v #get DISPLACEMENT (i.e. could be -ve) travelled for v in meters

    R_lat = sqrt(((R_eq ** 2) * (cos(lat) ** 2)) + ((R_po ** 2) * (sin(lat) ** 2)))
    delta_lon = (delta_x * R_lat) / (R_eq * R_eq * cos(lat)) #compute longitude shift for v (radians) - could be -ve
    delta_lon *= (180 / pi) #convert to degrees
    #print(delta_lon)

    hdulist = fits.open(img1) #open img1
    lon_left, lon_right, lon_step = hdulist[0].header['LON_LEFT'], hdulist[0].header['LON_RIGH'], hdulist[0].header['LON_STEP']

    lon_range_init = np.linspace(round(lon_left, 2), round(lon_right, 2) + lon_step, 1601, endpoint=False)
    lon_range_shifted = lon_range_init - delta_lon #creating array of longitudes, shifting by delta_lon

    return [lon_range_shifted, delta_lon]

def correlation(y, v):
    """
    Determine correlation at a given latitude y (pixels) at some velocity v (m/s) for ALL images. 
    (actual corr function will sum over all y in a bin, then all bins for all overlapping images - order doesn't
    matter)

    """
    overlapping_images = overlap_all(y, v)

    total_corr = 0

    for i in overlapping_images: #loop thru overlapping image pairs at given lat and v. 
        img1, img2 = i[0], i[1]
        print(img1, img2)
    
        hdulist1, hdulist2 = fits.open(img1), fits.open(img2) #open images
        lat = coords_convert(img1, 0, y)[1] #get latitude in degrees from y (image and x value don't matter)

        lon_left_2, lon_right_2, lon_step = hdulist2[0].header['LON_LEFT'], hdulist2[0].header['LON_RIGH'], hdulist2[0].header['LON_STEP']
        slice1_shifted = advection(img1, img2, y, v)[0] #get SHIFTED SLICE of IMG1 SHIFTING TO IMG2 (DEGREES LON)
        slice2_shifted = np.linspace(round(lon_left_2, 2), round(lon_right_2, 2) + lon_step, 1601, endpoint=False) #get other slice (DEGREES LON)
                    #note slice2_shifted is NOT an advected array; the naming is simply to match that of slice1_shifted.
        delta_lon = advection(img1, img2, y, v)[1] #get shift in longitude for img1 in DEGREES
        
        #print('slice1_shifted is', slice1_shifted)
        #print('slice2_shifted is', slice2_shifted)

        overlap = overlap_slice(slice1_shifted, slice2_shifted) #get overlapping longitude boundaries (array)
        if not overlap:
            return 0 #end the function body if no overlap is detected after advection
        
        #print('overlap is' , overlap[0], overlap[1])

        #convert overlap range from DEGREES lon to lon PIXELS. 
        slice1_shifted_pixels, slice2_shifted_pixels = [], []
                #red_overlap_1, red_overlap_2 = round(round(overlap[0] / 0.05) * 0.05, 2), round(round(overlap[1] / 0.05) * 0.05, 2) 
        N = np.linspace(overlap[0], overlap[1] + lon_step, int((overlap[1] - overlap[0]) / lon_step) + 1, endpoint=False)
            #this produces an overlap array with step length ~-0.05. 
        for i in N:
            #print(i)
            reverse_advect_i = round(i + delta_lon, 2)
            j = inverse_coords(img1, reverse_advect_i, lat)[0]
            if j in slice1_shifted_pixels:
                continue
            slice1_shifted_pixels.append(j)
        #print('slice1 done')
        for i in N:
            #print(i)
            j = inverse_coords(img2, i, lat)[0]
            if j in slice2_shifted_pixels:
                continue
            slice2_shifted_pixels.append(j)
        #print('slice2 done')
        #print(slice1_shifted_pixels, slice2_shifted_pixels)

        '''slice1_shifted_pixels = []
        ind1_0 =  np.where(slice1_shifted <= overlap[0])[0][0]
        ind1_1 =  np.where(slice1_shifted <= overlap[1])[0][0]

        ind2_0 = np.where(slice2_shifted >= overlap[0])[0][-1]
        ind2_1 = np.where(slice2_shifted >= overlap[1])[0][-1]

        print(ind1_0, ind1_1, ind2_0, ind2_1)
        j0 = inverse_coords(img1, overlap[0], lat)[0] #gets first longitude pixel of img1 in the overlap
        j = inverse_coords(img1, overlap[1], lat)[0] #gets last longitude pixel of img1 in the overlap
        slice1_shifted_pixels.append([0, y]) #append first [lon, lat] of img1's overlap
        slice1_shifted_pixels.append([j - j0, y]) #append last [lon, lat] of img1's overlap
        #instead of appending j0 and j, we append 0 and j - j0, zeroing the pixel range. 
        for i in overlap:
            j = inverse_coords(img1, i, lat)
            slice1_shifted_pixels.append(j)

        #repeat for the other non-shifted image, but without zeroing the pixel range.
        slice2_shifted_pixels = []
        for k in overlap:
            j = inverse_coords(img2, k, lat)
            slice2_shifted_pixels.append(j)'''

        #sanity check
        assert len(slice1_shifted_pixels) == len(slice2_shifted_pixels), 'Error - overlap pixel ranges different lengths.'
        
        #get corresponding brightness values from pixel overlap range. data array is size 2801 x 1601; each value corresponds to a pixel.
        slice1_brightness = hdulist1[0].data[y, slice1_shifted_pixels[0]:slice1_shifted_pixels[len(slice1_shifted_pixels) - 1]]
        slice2_brightness = hdulist2[0].data[y, slice2_shifted_pixels[0]:slice2_shifted_pixels[len(slice2_shifted_pixels) - 1]]

        n1_brightness, n2_brightness = len(slice1_brightness), len(slice2_brightness)
        assert n1_brightness == n2_brightness, 'Error - brightness slices different lengths.'

        sum1, sum2 = sum(slice1_brightness), sum(slice2_brightness)
        slice1_brightness_avg, slice2_brightness_avg = sum1 / n1_brightness, sum2 / n2_brightness #compute average brightness value of each array

        product_sum = 0
        for k in range(0, n1_brightness):
            product_sum += (slice1_brightness[k] * slice2_brightness[k])
        product_avg = product_sum / n1_brightness #compute average of products of arrays

        square_sum_1, square_sum_2 = 0, 0
        for k in range(0, n1_brightness):
            square_sum_1 += ((slice1_brightness[k]) ** 2)
            square_sum_2 += ((slice2_brightness[k]) ** 2)
        slice1_brightness_sq_avg, slice2_brightness_sq_avg = square_sum_1 / n1_brightness, square_sum_2 / n2_brightness #compute average of squares
        
        corr_numerator = product_avg - (n1_brightness * slice1_brightness_avg * slice2_brightness_avg)
        corr_denominator_1 = slice1_brightness_sq_avg - (n1_brightness * ((slice1_brightness_avg) ** 2))
        corr_denominator_2 = slice2_brightness_sq_avg - (n2_brightness * ((slice2_brightness_avg) ** 2))
        corr = corr_numerator / ((corr_denominator_1 * corr_denominator_2) ** 0.5) #calculate correlation

        total_corr += corr

    return total_corr #return final correlation

def v_max(y):
    """
    Return velocity with maximum correlation in m/s at a particular latitude y (pix) for two images. Also displays a graph.

    """
    vel_array = np.linspace(-300, 300, 71)
    correlations = []
    for a in vel_array:
        print('a is', a)
        corr = correlation(y, a)
        assert corr <= -1, 'Correlation is greater than -1.'
        correlations.append(corr)
    plt.plot(vel_array, correlations)
    plt.show()
    return vel_array[np.argmax(correlations)]

print(v_max(1424)) 