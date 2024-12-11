#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import nibabel as nib
import argparse

# In[2]:


def binarise_maps(input_image, threshold):
    """
    The function binarises an input image. It sets voxels with an intensity
    above the threshold to a value of 1 and voxels with an intensity below
    the threshold to a value of 0.

    input_image: the path to the image
    threshold: the threshold which will be used to binarise the img
    """
    img = nib.load(input_image)
    img = img.get_fdata()

    binary_image = np.zeros_like(img)

    binary_image[img>=threshold] = 1
    binary_image[img<threshold] = 0 
    return binary_image


# In[3]:


def dice_coefficient(binary_image_data_1, binary_image_data_2):
    """
    Calculates the Dice coefficient which quantifies the degree 
    of similarity between the two activation maps. The function
    takes 2 binary images as inputs and outputs the dice
    coefficient.
    """

    # Number of activated voxels in each image
    n_activations_1 = np.sum(binary_image_data_1)
    n_activations_2 = np.sum(binary_image_data_2)

    # Number of voxels that overlap i.e., the intersection
    overlap_image = binary_image_data_1 * binary_image_data_2
    intersection = np.sum(overlap_image)

    # Computing the Dice coefficient
    dice_coeff = (2*intersection)/(n_activations_1 + n_activations_2)
    return dice_coeff, n_activations_1, n_activations_2


# In[4]:


def jaccard_index(binary_image_data_1, binary_image_data_2):
    """
    Calculates the Jaccard index which quantifies the degree 
    of similarity between the two activation maps. The function
    takes 2 binary images as inputs and outputs the Jaccard index.
    """

    # Number of activated voxels in each image
    n_activations_1 = np.sum(binary_image_data_1)
    n_activations_2 = np.sum(binary_image_data_2)

    # Number of voxels that overlap i.e., the intersection
    overlap_image = binary_image_data_1 * binary_image_data_2
    intersection = np.sum(overlap_image)

    # Computing the Jaccard index
    jacc_index = (intersection)/(n_activations_1 + n_activations_2 - intersection)
    return jacc_index, n_activations_1, n_activations_2


# In[5]:


def overlap_coefficient(binary_image_data_1, binary_image_data_2):
    """
    Calculates the ratio of the intersection of the two binary images relative
    to the number of activated voxels in binary image containing the smallest
    number of activated voxels. This is called the overlap coefficient. The 
    VB index has a much larger number of voxels deemed significantly active.
    Hence, this metric may be used to quantify the proportion of the GLM's 
    activation map that was deemed significantly active by the VB index.
    """

    # Number of activated voxels in each image
    n_activations_1 = np.sum(binary_image_data_1)
    n_activations_2 = np.sum(binary_image_data_2)
    
    # The set which contains the smallest amount of activations
    if n_activations_1 < n_activations_2:
        smallest_subset = n_activations_1
    else:
        smallest_subset = n_activations_2

    # Number of voxels that overlap i.e., the intersection
    overlap_image = binary_image_data_1 * binary_image_data_2
    intersection = np.sum(overlap_image)

    # Computing the Overlap coefficient
    overlap_coeff = (intersection)/smallest_subset
    
    return overlap_coeff, n_activations_1, n_activations_2


# In[6]:

def main():
    parser = argparse.ArgumentParser(
        description=
        """Computes the Dice Coefficient, Jaccard index or the Overlap coefficient 
        to assess the spatial similarity of two input activation maps""",

        epilog=
        """  
        Example: Computing the Jaccard index of two activation maps thresholded to a user-specified degree
        python3 Jaccard_Dice_coefficient.py -sm1 $path_to_activation_map1$ -sm2 $path_to_activation_map2$ -thr1 5.2 -thr2 5.4
        Example 2: Computing the Overlap coefficient of two unthresholded (threshold=0) activation maps
        python3 Jaccard_Dice_coefficient.py -sm1 $path_to_activation_map1$ -sm2 $path_to_activation_map2$ -cf 1
        """,

        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument('-sm1', '--stat_map1', type=str, help='The path to the first statistical map', required=True)
    parser.add_argument('-sm2', '--stat_map2', type=str, help='The path to the second statistical map', required=True)
    parser.add_argument('-cf', '--coefficient', type=int, help='The similarity coefficient computed. Default (0) is the Jaccard index, (1) is the Dice coefficient and (2) is the Overlap coefficient. Pass 3 if all the coefficients are to be computed.', default=0)
    parser.add_argument('-thr1', '--threshold_statmap1', type=float, help='The threshold of the statistical map 1. The image is binarised using this threshold (default=0)', default=0)
    parser.add_argument('-thr2', '--threshold_statmap2', type=float, help='The threshold of the statistical map 2. The image is binarised using this threshold (default=0)', default=0)

    args = parser.parse_args()

    binary_image_1 = binarise_maps(args.stat_map1, args.threshold_statmap1)
    binary_image_2 = binarise_maps(args.stat_map2, args.threshold_statmap2)

    if args.coefficient==0:
        jc, ac1, ac2 = jaccard_index(binary_image_1, binary_image_2)
        print("Jaccard Index:", jc)
    elif args.coefficient==1:
        dc, ac1, ac2 = dice_coefficient(binary_image_1, binary_image_2)
        print("Dice coefficient:", dc)
    elif args.coefficient==2:
        oc, ac1, ac2 = overlap_coefficient(binary_image_1, binary_image_2)
        print("Overlap coefficient:", oc)
    elif args.coefficient==3:
        jc, ac1, ac2 = jaccard_index(binary_image_1, binary_image_2)
        dc, ac1, ac2 = dice_coefficient(binary_image_1, binary_image_2)
        oc, ac1, ac2 = overlap_coefficient(binary_image_1, binary_image_2)       
        print("Jaccard Index:", jc)
        print("Dice coefficient:", dc)
        print("Overlap coefficient:", oc)

if __name__ == '__main__':
    main()