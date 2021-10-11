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

#BASIC

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

#OVERLAP 

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
    overlapping_images = []
    images = glob.glob(path2data + '*.fits')

    #Loop through all possible image pairs
    for i in images[:len(images)-1]:
        x = images.index(i) + 1
        images2 = images[x:]
        for j in images2:
            #Filter out pairs w/ time difference of less than 5 hrs or more than 15 hrs
            if abs(time_difference(i, j)) < 18000 or abs(time_difference(i, j)) > 54000:
                continue
            #Advect one img to the other, and determine whether they overlap
            hdulist2 = fits.open(j)
            lon_left_2, lon_right_2 = hdulist2[0].header['LON_LEFT'], hdulist2[0].header['LON_RIGH'] 
            lon_range_1, lon_range_2 = advection(i, j, y, v)[0], np.linspace(round(lon_left_2, 2), round(lon_right_2, 2) - 0.05, 1601, endpoint=False)          
            if not overlap_slice(lon_range_1, lon_range_2):
                continue
            overlapping_images.append([i, j]) 

    #Return result
    return overlapping_images
     
#TIME AND ADVECTION

def time_difference(img1, img2):
    """
    Determine (time of img2) - (time of img1) in seconds. Can be negative. 

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
    lon_left, lon_right, lon_step = np.around(hdulist[0].header['LON_LEFT'], 2), np.around(hdulist[0].header['LON_RIGH'], 2), np.around(hdulist[0].header['LON_STEP'], 2)

    lon_range_init = np.linspace(lon_left, lon_right, hdulist[0].header['NAXIS1']) #get other slice (DEGREES LON)

    lon_range_shifted = lon_range_init - delta_lon #creating array of longitudes, shifting by delta_lon

    return [lon_range_shifted, delta_lon]

#CORRELATION

def averaging_correlation_img_pair(y, v, img1, img2, N=5):
    """
    Generate a matrix that has advected rows for an image pair, then calculate correlation for each row and sum over.

    N is the number of rows on each side of the inputted latitude to average over.

    """  
    #sanity check
    assert type(N) == int, 'N must be an integer.'
    
    #get information, and compute how many rows and columns the matrices should have
    hdulist1, hdulist2 = fits.open(img1), fits.open(img2)
    rows, columns = 2*N + 1, hdulist1[0].header['NAXIS1']
    lon_step = hdulist1[0].header['LON_STEP']

    #create a matrix of advected longitudes; each row is the advection of img1's longitude array to img2, differing by their latitude value
    advected_matrix = np.zeros((rows, columns))
    for i in range(rows):
        advected_matrix[i,:] = advection(img1, img2, y+N-i, v)[0]

    #create a comparison matrix for img2's longitudes
    comparison_matrix = np.zeros((rows, columns))
    lon_left_2, lon_right_2 = np.around(hdulist2[0].header['LON_LEFT'], 2), np.around(hdulist2[0].header['LON_RIGH'], 2)
    for i in range(rows):
        comparison_matrix[i,:] = np.linspace(lon_left_2, lon_right_2, hdulist2[0].header['NAXIS1'])
    
    #compute the overlap for each row and add it to a 2N+1 x 2 matrix
    overlaps = np.zeros((rows, 2))
    for i in range(rows):
        overlap = overlap_slice(advected_matrix[i,:], comparison_matrix[i,:])
        #print(advected_matrix[i,:], comparison_matrix[i,:])
        #if not overlap or np.abs(np.diff(overlap)) < 10:
           # overlap = np.array([np.nan, np.nan])
        overlaps[i,:] = overlap
    
    '''for each overlap row: get data pixels, filter nan values, interpolate, determine brightness values, compute correlation. 
    Also need to only look at pixels where BOTH mask arrays are 1 (0 = mask, 1 = no mask).
    '''
    total_correlation = 0
    for i in range(rows):
        #get data pixel values
        overlap = overlaps[i, :]
        N = np.arange((overlap[0]/lon_step + 1) * lon_step, (overlap[-1]/lon_step - 1) * lon_step, lon_step) #get full overlap range (excluding endpoints)
        slice1, slice2 = hdulist1[0].data[y - i + rows//2, :], hdulist2[0].data[y - i + rows//2, :] #get corresponding data entries (NOT PIXELS) from img1, img2
        mask1, mask2 = hdulist1[1].data[y - i + rows//2, :], hdulist2[1].data[y - i + rows//2, :] #get corresponding data entries (NOT PIXELS) from mask1, mask2
        '''nan filter for debugging on untrimmed files. arrays above must be NumPy arrays for idx and idy to work. order of idx/idy shouldn't matter.
        masks themselves shouldn't have nan values, but filter them with idx/y anyway to make them same length when interpolating. most images don't have nan 
        values which is why the code has only failed in some instances but not others.'''
        idx = slice1==slice1 #gives positions of nans of the ORIGINAL slice1 array
        slice1 = slice1[idx]
        slice2 = slice2[idx]
        mask1 = mask1[idx]
        mask2 = mask2[idx]
        idy = slice2==slice2 #gives positions of nans of the MODIFIED slice2 array - array left over after removing positions of slice1 nans
        slice1 = slice1[idy]
        slice2 = slice2[idy]
        mask1 = mask1[idy]
        mask2 = mask2[idy]
        #get longitudes for interpolation (also w/ nan filter). ORDER MATTERS - must do idx first. See comments on idx/idy lines for why.
        slice1_lon = advected_matrix[i][idx] 
        slice1_lon = advected_matrix[i][idy] 
        slice2_lon = comparison_matrix[i][idx] 
        slice2_lon = comparison_matrix[i][idy]
        assert(len(mask1) == len(mask2) == len(slice1) == len(slice2) == len(slice1_lon) == len(slice2_lon), 'Error - slices different lengths.')
        #interpolation requires continous arrays, avoid wrap 360 to 0.
        if np.any(slice1_lon < 0):
            slice1_lon += 360 
        if np.any(slice2_lon < 0):
            slice2_lon += 360 
        #interpolate the slices onto the overlap grid to get final brightness values
        slice1_brightness = np.flip(np.interp(np.flip(N), np.flip(slice1_lon), np.flip(slice1)))
        slice2_brightness = np.flip(np.interp(np.flip(N), np.flip(slice2_lon), np.flip(slice2)))
        #filter out to get pixels where masks for both images have value of 1
        slice1_masking = np.flip(np.interp(np.flip(N), np.flip(slice1_lon), np.flip(mask1)))
        slice2_masking = np.flip(np.interp(np.flip(N), np.flip(slice2_lon), np.flip(mask2)))
        mask_filter = np.where((slice1_masking==1) & (slice2_masking==1)) #list of pixel values / indices for which masks are 1
        #get final brightness values after filtering out masked pixels
        slice1_masked, slice2_masked = slice1_brightness[mask_filter], slice2_brightness[mask_filter]
                
        #compute correlation from resulting arrays
        correlation = row_correlation(slice1_masked, slice2_masked)
        total_correlation += correlation
    
    #return the final correlation for the image pair
    final_correlation = total_correlation / rows
    return final_correlation

def row_correlation(arr1, arr2):
    """
    Compute the correlation for two 1D or 2D arrays of arbitrary shape by averaging over rows (uses same formula as correlation_image_pair).

    IMPT: array arguments must be NumPy arrays, not general Python ones. Rows correspond to lats, columns correspond to lons.
    Correlation should be a value between -1 and 1. 

    """
    #ensure that the matrices have the same shape, and compute number of rows
    assert np.shape(arr1) == np.shape(arr2), 'Error - arrays need to be same shape.'
    #print(np.shape(arr1), np.shape(arr2))
    dimension = len(np.shape(arr1))
    if dimension == 1:
        rows = 1
    elif dimension == 2:
        rows = np.shape(arr1)[0]

    #compute row by row correlation and take the average
    total_corr = 0
    if dimension == 1:
        row1, row2 = arr1, arr2
        N = len(row1)
        sum1, sum2 = np.sum(row1), np.sum(row2)
        product_sum = np.dot(row1, row2)
        sq_sum1, sq_sum2 = np.sum(row1 ** 2), np.sum(row2 ** 2)
        try:
            total_corr = (product_sum - sum1*sum2/N) / sqrt((sq_sum1 - (sum1**2)/N) * (sq_sum2 - (sum2**2)/N))
        except ValueError: #in case of a math domain error, exclude the row
            total_corr = 0
    elif dimension == 2:
        for i in range(rows):
            row1, row2 = arr1[i,:], arr2[i,:]
            N = len(row1)
            sum1, sum2 = np.sum(row1), np.sum(row2)
            product_sum = np.dot(row1, row2)
            sq_sum1, sq_sum2 = np.sum(row1 ** 2), np.sum(row2 ** 2)
            try:
                row_corr = (product_sum - sum1*sum2/N) / sqrt((sq_sum1 - (sum1**2)/N) * (sq_sum2 - (sum2**2)/N))
            except ValueError: #in case of a math domain error, exclude the row
                row_corr = 0
            total_corr += row_corr
        
    #return average correlation
    return total_corr / rows

def readZWP(plotting=False): 
    '''
    Read in the zonal wind profile from Josh's 2019 paper 

    ------------
    Return:
    latitude Planetographic (deg) 
    E-W Wind speed (m/s) 
    '''
    path2data = './'
    path2wp = path2data + 'ZWP_j2019_PJ19.txt'

    A = np.loadtxt(path2wp) 
    if plotting: 
        fig, axs = plt.subplots(1, 1, figsize=(8,4))
        axs.plot(A[:,1],A[:,0],label='JT - ZWP')
        axs.set_ylabel('Latitude (deg)')
        axs.set_xlabel('Velocity (m/s)')
        axs.set_ylim([-60,60])

    return A[:,0],A[:,1] 

def v_max(y, path2data=None, plotting=False, vstep=37):
    """
    Return velocity with maximum correlation in m/s at a particular latitude y (pix) for two images.

    vstep is the NUMBER of velocities being tested, not the actual step size despite its name. 

    Plotting code below is intended to show correlation vs velocity for each image pair. 

    """
    #set up path environment 
    if path2data is None:  
        path2data = './'

    #obtain image pairs at velocity 0 only - saves time when computing
    im_pairs = overlap_all(y, 0, path2data=path2data)
    
    #intitialize the arrays to be saved in - correlations is 2D, with a correlation slot for each pair at each velocity
    vel_array = np.linspace(-180, 180, vstep)
    correlations = np.zeros((len(im_pairs), vstep))

    #compute correlations for each slot
    for i in range(len(im_pairs)):
        print(im_pairs[i][0], im_pairs[i][1])
        for j in range(len(vel_array)):
            correlations[i,j] = averaging_correlation_img_pair(y, vel_array[j], im_pairs[i][0], im_pairs[i][1]) #account for averaging here!
        if plotting: 
            plt.plot(vel_array, correlations)
            #plt.ylim([1,min(correlations)])
            plt.ylabel('correlation')
            plt.xlabel('Velocity')
            plt.show()

    # Remove inf by converting to nans 
    for i in range(len(im_pairs)): 
        if np.any(~np.isfinite(correlations[i,:])): 
            correlations[i,:] = np.nan 

    #sum over all image pair correlations for each velocity; return the velocity with the highest correlation
    return vel_array[np.argmax(np.nansum(correlations, axis=0))]

#RUNNING CODE FOR A CERTAIN LATITUDE RANGE

#Set path of images
path2data = './'

#Get an array of latitudes from -70 to 70 degrees in increments of +0.05 to refer to when printing out latitudes below.
images = glob.glob(path2data + '*.fits')
image1 = images[0]
hdulist = fits.open(image1) 
lat_bot, lat_top, lat_step = hdulist[0].header['LAT_BOT'], hdulist[0].header['LAT_TOP'], hdulist[0].header['LAT_STEP']
latitude = np.linspace(lat_bot, lat_top, int((lat_top - lat_bot)/lat_step) + 1)

#Generate an array of latitudes (pixels) and best velocities (m/s). 
lat = []
v_corr = [] 
for y in range(500, 900, 10):
    t0 = time()
    try:
        v = v_max(y, path2data=path2data, plotting=False, vstep=361)
    except:
        print('Error at', y, 'pix latitude.')
    v_corr.append(v)
    lat.append(y) 
    print((time() - t0)/60)
    print(f'Latitude {latitude[y]:2.2f} Velocity',v_corr[-1])

#Plot results along with currently accepted ZWP to compare. 
lat_zwp, zwp = readZWP() 
fig, axs = plt.subplots(1, 1,figsize=(8,4))
axs.plot(zwp,lat_zwp,label='JT - ZWP')
axs.plot(v_corr,latitude[lat],label='DP')
axs.set_ylabel('Latitude (deg)')
axs.set_xlabel('Velocity (m/s)')
axs.set_ylim([-60,60])
plt.show()

#Save results as text file.
lat_title = ['Latitude (deg)']
vel_title = ['Velocities (m/s)']
np.savetxt('/Users/Druv/Documents/Berkeley/Career/URAP_Jupiter/results/2020a/final/output.txt', np.transpose([lat_title, vel_title]), fmt="%s", delimiter="   ")
with open('/Users/Druv/Documents/Berkeley/Career/URAP_Jupiter/results/2020a/final/output.txt', 'ab') as file: 
    np.savetxt(file, np.transpose([latitude[lat], v_corr]), fmt='%.2e', delimiter='          ')