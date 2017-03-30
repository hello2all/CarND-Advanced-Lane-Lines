import glob

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from collections import deque

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Calculate directional gradient
    img = np.copy(img)
    if orient == 'x':
        # Sobel x
        sobel = cv2.Sobel(img, cv2.CV_64F, 1, 0) # Take the derivative in x
    elif orient == 'y':
        # Sobel y
        sobel = cv2.Sobel(img, cv2.CV_64F, 0, 1) # Take the derivative in y
    else:
        raise NameError('Please specify gradient orientation, x or y')
    # Absolute derivative to accentuate lines away from horizontal
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    # Threshold gradient
    grad_binary = np.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    return grad_binary

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    mag_binary = np.zeros_like(gradmag)
    mag_binary[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
    return mag_binary

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    dir_binary =  np.zeros_like(absgraddir)
    dir_binary[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    return dir_binary

def color_threshold(img, h_thresh=(0, 255), s_thresh=(0, 255), v_thresh=(0, 255)):
    img = np.copy(img)
    # Convert to HSV color space and separate channel
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float)
    h_channel = hsv[:, :, 0]
    s_channel = hsv[:, :, 1]
    v_channel = hsv[:, :, 2]

    # Threshold color channel
    color_binary = np.zeros_like(s_channel)
    color_binary[((s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])) &
                ((h_channel >= h_thresh[0]) & (h_channel <= h_thresh[1])) &
                ((v_channel >= v_thresh[0]) & (v_channel <= v_thresh[1]))] = 1
    return color_binary

def cal_undistort(img, objpoints, imgpoints):
    # Use cv2.calibrateCamera() and cv2.undistort()
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (img.shape[1], img.shape[0]),None,None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist

def warp_image(img,src,dst,img_size):
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    Minv = cv2.getPerspectiveTransform(dst, src)

    return warped, M, Minv

def calibrate(path='./camera_cal/calibration*.jpg'):
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
    objpoints = []
    imgpoints = []
    # Make a list of calibration images
    images = glob.glob(path)
    # Step through the list and search for chessboard corners
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6),None)
        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
    
    return objpoints, imgpoints
def set_perspective(img_size=(720, 1280)):
    ht_window = np.uint(img_size[0]/1.5)
    hb_window = np.uint(img_size[0])
    c_window = np.uint(img_size[1]/2)
    ctl_window = c_window - .2*np.uint(img_size[1]/2)
    ctr_window = c_window + .2*np.uint(img_size[1]/2)
    cbl_window = c_window - 1*np.uint(img_size[1]/2)
    cbr_window = c_window + 1*np.uint(img_size[1]/2)
    src = np.float32([[cbl_window,hb_window],[cbr_window,hb_window],[ctr_window,ht_window],[ctl_window,ht_window]])
    dst = np.float32([[0,img_size[0]],[img_size[1],img_size[0]],[img_size[1],0],[0,0]])
    
    return src, dst

def gradient_pipe_line(image):
    img_g_mag = mag_thresh(image,3,(20,150))
    img_d_mag = dir_threshold(image,3,(.6,1.1))
    img_abs_x = abs_sobel_thresh(image,'x',5,(50,200))
    img_abs_y = abs_sobel_thresh(image,'y',5,(50,200))
    sobel_combined = np.zeros_like(img_d_mag)
    sobel_combined[((img_abs_x == 1) & (img_abs_y == 1)) | \
            ((img_g_mag == 1) & (img_d_mag == 1))] = 1
    return sobel_combined

def calc_radius_pos(binary_warped, leftx, lefty, rightx, righty):
    y_eval = np.max(binary_warped.shape[0] - 1)
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/920 # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radius of curvature in meters
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    # calculate vehicle position
    position = (right_fit_cr[2] - left_fit_cr[2])/2 + left_fit_cr[2] - binary_warped.shape[1]/2 * xm_per_pix

    return left_curverad, right_curverad, position

def extract_pixels_uninformed(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]/2:,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty

def extract_pixels_informed(binary_warped, left_fit, right_fit):
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty

def polyfit_pixels(leftx, lefty, rightx, righty):
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    return left_fit, right_fit

def overlay_lane_detection(image, binary_warped, Minv, left_fit, right_fit):
    # Create an image to draw the lines on
    color_warp = np.zeros_like(image).astype(np.uint8)

    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1],image.shape[0])) 
    # Combine the result with the original image
    overlay = cv2.addWeighted(image, 1, newwarp, 0.3, 0)
    return overlay

def overlay_curvature_pos(overlay, left_curverad, right_curverad, position):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(overlay, "left line radius: {0:.5g} m".format(left_curverad), (50,50), font, 1, (255,255,255),2,cv2.LINE_AA)
    cv2.putText(overlay, "right line radius: {0:.5g} m".format(right_curverad), (50,100), font, 1, (255,255,255),2,cv2.LINE_AA)

    if position > 0:
        rel_dir = "left"
    else:
        rel_dir = "right"

    cv2.putText(overlay, "Vehicle is {0:.2g}m {1} of center".format(np.absolute(position), rel_dir), (50,150), font, 1, (255,255,255),2,cv2.LINE_AA)

    return overlay

def warped_lane_binary(undist, src, dst):
    # Extract yellow binary
    yellow_binary = color_threshold(undist,
                        h_thresh=(0, 50),
                        s_thresh=(90, 255),
                        v_thresh=(0, 255))
    # Extract white binary
    white_binary = color_threshold(undist,
                        h_thresh=(0, 255),
                        s_thresh=(0, 30),
                        v_thresh=(200, 255))
    # Combine color binaries
    color_binary = cv2.bitwise_or(yellow_binary, white_binary)
    
    # Convert undistorted image to HLS
    hls = cv2.cvtColor(undist, cv2.COLOR_RGB2HLS)
    l = hls[:, :, 1]
    s = hls[:, :, 2]
    # Apply gradient pipe line to L and S channel
    gradient_combined = gradient_pipe_line(l) + gradient_pipe_line(s)
    # Apply Gaussian blur
    gradient_combined_blur = cv2.GaussianBlur(gradient_combined, (5, 5), 0)
    # Gradient Binary
    gradient_binary = np.zeros_like(gradient_combined_blur)
    gradient_binary[gradient_combined_blur > 0.5] = 1

    # combine color and gradient filter
    lane_combined = cv2.bitwise_or(color_binary, gradient_binary)
    binary_warped, M, Minv = warp_image(lane_combined, src, dst, (undist.shape[1], undist.shape[0]))

    return  binary_warped, M, Minv

class Pipe_line():
    def __init__(self, img_size=(720, 1280)):
        print("Calibrating Camera")
        self.objpoints, self.imgpoints = calibrate()
        print("Setting perspective")
        self.src, self.dst = set_perspective(img_size=img_size)
        print("Pipe line ready")
        # Init line objects
        self.left_line = Line()
        self.right_line = Line()

        self.left_line.current_fit = None
        self.right_line.current_fit = None

    def process(self, image):
        # undistort image
        undist = cal_undistort(image, self.objpoints, self.imgpoints)
        # get warped detected lanes
        binary_warped, M, Minv = warped_lane_binary(undist, self.src, self.dst)
        # extract lane pixels
        if (self.left_line.current_fit is None) | (self.right_line.current_fit is None):
            # uninformed search
            leftx, lefty, rightx, righty = extract_pixels_uninformed(binary_warped)
        else:
            # informed search (based on margin)
            leftx, lefty, rightx, righty = extract_pixels_informed(binary_warped, self.left_line.current_fit, self.right_line.current_fit)
        # calculate polyfit coefficients
        left_fit, right_fit = polyfit_pixels(leftx, lefty, rightx, righty)
        self.left_line.update_queue(left_fit)
        self.right_line.update_queue(right_fit)

        #calculate 
        
        # calculate curvature
        left_curverad, right_curverad, position = calc_radius_pos(binary_warped, leftx, lefty, rightx, righty)
        # overlay detected lane
        overlay = overlay_lane_detection(undist, binary_warped, Minv, self.left_line.best_fit, self.right_line.best_fit)
        # overlay curvature and position text
        overlay = overlay_curvature_pos(overlay, left_curverad, right_curverad, position)
        return overlay

class Line():
    def __init__(self):
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #polynomial coefficients queue
        self.fit_queue = deque([])
    # Update FIFO queue of recent values
    def update_queue(self, value, n=5):
        self.current_fit = value
        self.fit_queue.append(value)
        if len(self.fit_queue) > n:
            self.fit_queue.popleft()
        self.best_fit = np.average(self.fit_queue, axis=0)
