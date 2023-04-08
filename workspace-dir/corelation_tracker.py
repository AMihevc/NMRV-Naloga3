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
        # print(patch.shape)
        # print(self.size)

        if (patch.shape[0] != self.size[1] or patch.shape[1] != self.size[0]):
            print("Error: patch size is not correct")
            print("Patch shape:", patch.shape)
            print("Expected size:", self.size)
            print("Position:", self.position)
            print("Image shape:", image.shape)
            #exit()

        #convert to grayscale
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

        # multiply with the cosine window
        patch = np.multiply(self.cosine_window, patch)

        # convert to frequency domain
        patch = np.fft.fft2(patch)
        
        return patch
    
    def make_filter(self, given_patch):
        return (self.fft_gaussian_peak * np.conj(given_patch)) / ((self.patch * np.conj(given_patch)) + self.lamda)
    
    def initialize(self, image, region): #initialize the tracker
        #parameters (majbe should be able to change them with arguments)
        self.enlargment_factor = 1.0
        self.alpha = 0.2
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
        # print("size:")
        # print(self.size)

        # define the cosine window
        self.cosine_window = create_cosine_window(self.size)
        # print("cosine window:")
        # print(self.cosine_window.shape)

        # define the gaussian peak
        self.gaussian_peak = create_gauss_peak(self.size, self.sigma)
        self.fft_gaussian_peak = np.fft.fft2(self.gaussian_peak)

        # get the patch
        self.patch = self.make_fft_patch(image)

        # define the filter
        self.filter = self.make_filter(self.patch)

    
    def track(self, image): #track the object in the image
        # get the patch
        fft_patch = self.make_fft_patch(image)

        # calculate the response
        response = np.fft.ifft2(self.filter * fft_patch)

        # find the peak
        peak_y, peak_x  = np.unravel_index(np.argmax(response), response.shape) #is a tuple (y,x)

        # update the position
        # self.position = (self.position[0] + (peak[0] - self.size[0] / 2), self.position[1] + (peak[1] - self.size[1] / 2)) # this is the center of the patch
        #Todo-Done check if this is correct from slides you might not need to subtract the size (it was not correct )
        # from slides: 
        if (peak_x > self.size[0] / 2):
            peak_x = peak_x - self.size[0]

        if (peak_y > self.size[1] / 2):
            peak_y = peak_y - self.size[1]

        self.position = (self.position[0] + peak_x, self.position[1] + peak_y)
        

        # update the filter
        new_patch = self.make_fft_patch(image)
        self.filter = (self.filter * (1 - self.alpha) + (self.make_filter(new_patch) * self.alpha)) 

        # return the new position as list
        return [self.position[0] - self.original_size[0] / 2, self.position[1] - self.original_size[1] / 2, self.original_size[0], self.original_size[1]]





