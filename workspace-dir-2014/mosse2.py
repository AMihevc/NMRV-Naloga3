import numpy as np
import cv2
from utils.tracker import Tracker
from ex2_utils import get_patch
from ex3_utils import create_cosine_window, create_gauss_peak

class MosseSimple2(Tracker):
    def name(self):
        return "mosse2"
    
    def make_fft_patch(self, image):
        
        # get the patch
        patch, crop_mask = get_patch(image, self.position, self.size)

        # for debugging purposes
        if (patch.shape[0] != self.size[1] or patch.shape[1] != self.size[0]):
            print("Error: patch size is not correct")
            print("Patch shape:", patch.shape)
            print("Expected size:", self.size)
            print("Position:", self.position)
            print("Image shape:", image.shape)

        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) # convert to grayscale
        patch = np.multiply(self.cosine_window, patch) # apply cosine window
        fouier_patch = np.fft.fft2(patch) # convert to frequency domain FROM SLIDES
        
        return fouier_patch
    
    def make_filter(self, given_patch):
        # this is the equation from instructions for filter H 
        return (self.fft_gaussian_peak * np.conj(given_patch)) / ((given_patch * np.conj(given_patch)) + self.lamda)
    
    def initialize(self, image, region): #initialize the tracker
        #parameters for the tracker
        self.enlargment_factor = 1.25
        self.alpha = 0.2
        self.sigma = 2.0
        self.lamda = 0.000001
        # TODO change this params to find best results

        # define the region of interest
        self.position = (region[0] + region[2] / 2, region[1] + region[3] / 2)

        # store original size for better display of results 
        self.original_size = (round(region[2]), round(region[3]))
        # make sure the size is odd 
        if self.original_size[0] % 2 == 0:
            self.original_size = (self.original_size[0] + 1, self.original_size[1])
        if self.original_size[1] % 2 == 0:
            self.original_size = (self.original_size[0], self.original_size[1] + 1)
       
        # define the size with enlargement use this to calculate
        self.size = (round(region[2] * self.enlargment_factor), round(region[3] * self.enlargment_factor))
        # make sure the size is odd f
        if self.size[0] % 2 == 0:
            self.size = (self.size[0] + 1, self.size[1])
        if self.size[1] % 2 == 0:
            self.size = (self.size[0], self.size[1] + 1)

        # define the cosine window
        self.cosine_window = create_cosine_window(self.size)

        # define the gaussian peak
        self.fft_gaussian_peak = np.fft.fft2(create_gauss_peak(self.size, self.sigma))

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
        peak_y, peak_x = np.unravel_index(response.argmax(), response.shape) #returns the index of the max value in the array

        # update the position
        # from slides (page 9): (so that the peak is top left corner of the patch because gaussian is inverted)
        if (peak_x > self.size[0] / 2):
            peak_x = peak_x - self.size[0]

        if (peak_y > self.size[1] / 2):
            peak_y = peak_y - self.size[1]

        self.position = (self.position[0] + peak_x, self.position[1] + peak_y) # update
    
        # update the filter
        new_filter = self.make_filter(self.make_fft_patch(image))
        self.filter = ((self.filter * (1 - self.alpha)) +  new_filter * self.alpha)

        # return the new position
        return [self.position[0] - self.original_size[0] / 2, self.position[1] - self.original_size[1] / 2, self.original_size[0], self.original_size[1]]
    

# class MosseSimple1(MosseSimple):
#     def __init__(self):
#         super().__init__()
#         self.enlargment_factor = 1.0
#         self.alpha = 0.15
#         self.sigma = 2.0
#         self.lamda = 0.000001

#     def name(self):
#         return f"mosse_tracker_{self.enlargment_factor}_{self.alpha}_{self.sigma}_{self.lamda}"
    
# class MosseSimple2(MosseSimple):
#     def __init__(self):
#         super().__init__()
#         self.enlargment_factor = 1.0
#         self.alpha = 0.15
#         self.sigma = 2.0
#         self.lamda = 0.000001

#     def name(self):
#         return f"mosse_tracker_{self.enlargment_factor}_{self.alpha}_{self.sigma}_{self.lamda}"
    
# class MosseSimple3(MosseSimple):
#     def __init__(self):
#         super().__init__()
#         self.enlargment_factor = 1.0
#         self.alpha = 0.15
#         self.sigma = 2.0
#         self.lamda = 0.000001

#     def name(self):
#         return f"mosse_tracker_{self.enlargment_factor}_{self.alpha}_{self.sigma}_{self.lamda}"
    

# class MosseSimple4(MosseSimple):
#     def __init__(self):
#         super().__init__()
#         self.enlargment_factor = 1.0
#         self.alpha = 0.15
#         self.sigma = 2.0
#         self.lamda = 0.000001

#     def name(self):
#         return f"mosse_tracker_{self.enlargment_factor}_{self.alpha}_{self.sigma}_{self.lamda}"