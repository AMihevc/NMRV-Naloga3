import numpy as np
import cv2
from utils.tracker import Tracker
from ex2_utils import get_patch
from ex3_utils import create_cosine_window, create_gauss_peak

class MosseSimple(Tracker):

    def name(self):
        return "mosse_simple"
    
    def make_fft_patch(self, image):
        # get the patch
        patch, crop_mask = get_patch(image, self.position, self.size)

        #convert to grayscale
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

        # multiply with the cosine window
        patch = np.multiply(self.cosine_window, patch)

        # convert to frequency domain
        patch = np.fft.fft2(patch)
        
        return patch
    
    def initialize(self, image, region): #initialize the tracker
        #parameters (majbe should be able to change them with arguments)
        self.enlargment_factor = 1.0
        self.alpha = 0.125
        self.sigma = 2.0
        self.lamda = 0.000001

        # define the region of interest
        self.position = (region[0] + region[2] / 2, region[1] + region[3] / 2)

        # store original size
        self.original_size = (round(region[2]), round(region[3]))

        # make sure the size is odd for the convolution
        if self.original_size[0] % 2 == 0:
            self.original_size = (self.original_size[0] + 1, self.original_size[1])
        if self.original_size[1] % 2 == 0:
            self.original_size = (self.original_size[0], self.original_size[1] + 1)

        # define the size with enlargement
        self.size = (round(region[2] * self.enlargment_factor), round(region[3] * self.enlargment_factor))
        # make sure the size is odd for the convolution
        if self.size[0] % 2 == 0:
            self.size = (self.size[0] + 1, self.size[1])
        if self.size[1] % 2 == 0:
            self.size = (self.size[0], self.size[1] + 1)

        # define the cosine window
        self.cosine_window = create_cosine_window(self.size)

        # define the gaussian peak
        self.gaussian_peak = create_gauss_peak(self.size, self.sigma)
        self.fft_gaussian_peak = np.fft.fft2(self.gaussian_peak)

        # get the patch
        self.patch = self.make_fft_patch(image)

        # define the filter
        self.filter = (self.fft_gaussian_peak * np.conj(self.patch)) / ((self.patch * np.conj(self.patch)) + self.lamda)

    
    def track(self, image): #track the object in the image
        # get the patch
        fft_patch = self.make_fft_patch(image)

        # calculate the response
        response = np.fft.ifft2(self.filter * fft_patch)

        # find the peak
        peak = np.unravel_index(np.argmax(response), response.shape) #is a tuple (x,y)


        # update the position
        self.position = (self.position[0] + (peak[0] - self.size[0] / 2), self.position[1] + (peak[1] - self.size[1] / 2)) # this is the center of the patch
        #TODO check if this is correct from slides you might not need to subtract the size

        # update the filter
        new_patch = self.make_fft_patch(image)
        self.filter = ((self.filter * (1 - self.alpha)) + ((self.fft_gaussian_peak * np.conj(self.patch)) / ((self.patch * np.conj(self.patch)) + self.lamda)) * self.alpha) #TODO check if this is correct and could be preatyer 

        # return the new position as list
        return [self.position[0] - self.original_size[0] / 2, self.position[1] - self.original_size[1] / 2, self.original_size[0], self.original_size[1]]





