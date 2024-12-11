#!/usr/bin/env python
# coding: utf-8

# # Statistical analysis of VB

# In[1]:


# Importing packages necessary for the computation
import numpy as np
import nibabel as nib
import os
import nilearn 
import pandas as pd
import nilearn
import argparse

from nilearn.plotting import plot_stat_map
from nilearn.glm import threshold_stats_img
from templateflow import api as tflow
from nilearn import image
from fsl.wrappers import fslmaths, fslstats

# In[2]:


def baseline(data_path, vv):
    """
    The function computes summary statistics of the input data including the median,
    average, etc. The input data should be passed as a path to a folder containing
    the VB data.
    """

    print("- Searching for VB data in the specified path")

    # Searching through the path and appending their paths in arrays
    files = os.listdir(data_path)
    vb_files = []
    
    for file in files:
        if "vbi-vol_normalised" in file:
            path_file = os.path.join(data_path, file)
            vb_files.append(path_file)
            
    if len(vb_files) > 0:
        print(f"- {len(vb_files)} VB files found... proceeding")
    else:
        print("- VB files not found... terminating analysis")
        exit()

    # Proceeding; computing the baseline i.e., the average of the median VB of all files
    print("- Computing summary statistics on the VB files...")
    
    median_array = []
    mean_array = []
    nth_percentile = []
    nine_fifth_perc = []
    
    for vb_file in vb_files:
        vb_img  = nib.load(vb_file)
        # Computing a mask for the VB file
        mask_img = fslmaths(vb_img).bin().run()
        median = fslstats(vb_img).k(mask_img).p(50).run()
        median_array.append(median)
        mean = fslstats(vb_img).M.run()
        mean_array.append(mean)

        #nth percentile
        nth = fslstats(vb_img).P(75).run()
        nth_percentile.append(nth)
        
        nin_fifth = fslstats(vb_img).P(95).run()
        nine_fifth_perc.append(nin_fifth)
        
    mean_med = np.mean(median_array)
    var_median = np.var(median_array)
    mean_vb = np.mean(mean_array) 
    var_vb = np.mean(mean_array)
    
    print(f"- The mean of the median VB index across all subjects: {mean_med}")
    print(f"- The variance of the median VB index across all subjects: {var_median}")
    print(f"- The mean VB index all subjects: {mean_vb}")
    print(f"- The variance of the mean VB index across all subjects: {var_vb}")

    mean_nth = np.mean(nth_percentile)
    print(f"- 75th Percentile VB index is {mean_nth}")
    
    mean_ninefifth = np.mean(nine_fifth_perc)
    print(f"- 95th Percentile VB index is {mean_ninefifth}")

    print("- Summary statistics computed")
    return mean_med, mean_vb, vb_files, mean_nth, mean_ninefifth


# In[3]:


def statistical_test(vb_paths, vv, label=None, sl_smooth=None, alpha=0.05, path_struct=None):
    """
    Takes the baseline computed by the first function and
    performs a one sample test using the baseline. This function
    loads the paths of the spatially normalised VB data
    specified in the vb_paths, and performs a one sample test using the 
    baseline
    """
    
    mean_med, mean_vb, vb_normal_files, mean_nth, mean_ninefifth = baseline(vb_paths, vv)

    # Has the user specified a structural image path? used in the plotting
    if path_struct:
        print("- Importing the structural image")
        mni_image = image.load_img(path_struct)
    else:
        # Importing the template MNI image from template flow; used in the statistical map plotting
        print("- Importing the template MNI image from template flow")
        mni_image = tflow.get('MNI152NLin6Asym', desc=None, resolution=1, suffix='T1w', extension='nii.gz')
        mni_image = image.load_img(mni_image)
        nib.save(mni_image, "MNI152NLin6Asym_template.nii.gz")

    # Searching for VB data within the path specified
    print("- Searching for VB data in the specified directory...")
    
    #dir_path, base_filename = os.path.split(vb_paths[0])
    files = os.listdir(vb_paths)
    vb_files = []
    imgs = []
    
    # Searching for VB-index files in the specified path 
    for file in files:
        if "vbi-vol_normalised" in file:
            path_file = os.path.join(vb_paths, file)
            vb_files.append(path_file)
            
            img = nib.load(path_file)
            img_np = img.get_fdata()
            
            # Removing the baseline from the data
            img_minus_bs = img_np - mean_med  
            img_minus_bs = nib.Nifti1Image(img_minus_bs, affine=img.affine, header=img.header)

            imgs.append(img_minus_bs)
    
    # Number of subjects to perform the test
    n_subjects = len(vb_files)
    print(f"- Performing group level analysis for the VB index on {n_subjects} subjects at {vv} resolution")

    # Constructing the second level design matrix
    design_matrix = pd.DataFrame([1] * n_subjects, columns=['intercept'])

    second_level_model = nilearn.glm.second_level.SecondLevelModel(mask_img=None,
                                            target_affine=None, target_shape=None,
                                            smoothing_fwhm=sl_smooth, memory=None,
                                            memory_level=1, verbose=0,
                                            n_jobs=1, minimize_memory=True)

    # The following function takes a second_level_input as a list of z-maps for the different subjects
    second_level_model = second_level_model.fit(imgs, design_matrix=design_matrix)

    # Computing the contrasts for the second level analysis: One sample testing with non-parametric multiple comparisons correction
    stat_map = second_level_model.compute_contrast(second_level_contrast='intercept', second_level_stat_type='t', output_type='stat')

    # Bonferroni correction
    thresholded_map, threshold = threshold_stats_img(stat_map, alpha=alpha, height_control="bonferroni", cluster_threshold=10, two_sided=False)

    nib.save(thresholded_map, f"{label}_bonf_map_{vv}_vb_smoothsl_{sl_smooth}.nii.gz")

    # Plotting the thresholded map
    print(f"- Bonferroni-corrected, p<{alpha} threshold: {threshold:.3f}")
    x = plot_stat_map(
        thresholded_map,
        bg_img = mni_image,
        threshold=threshold,
        display_mode="mosaic",
        cut_coords=5,
        black_bg=True,
        cmap='jet',
        title=f"task-rest_bonferroni={alpha}), threshold: {threshold:.3f}, clusters > 10 voxels",)
    x.savefig(f"{label}_smoothsl_{sl_smooth}_bonf_clustered_map_vb_{vv}mm.pdf")
    print("- Second-level computation done")
    return None

# In[4]:

def main():
    parser = argparse.ArgumentParser(
        description='Performs a one-sample t-test to assess which voxels have a VB index that is statistically larger (p<0.05) than the'
        'median VB index across all subjects. The script takes the path to the folder containing the VB data as input. To run, place the'
        'spatially normalised VB data inside a folder. Make sure that the VB data files has vb-vol_normalised in their name. Then as input'
        'provide the path to the folder. You may also specify a FWHM to smoothen the data via the -sm argument. The algorithm also computed'
        'summary statistics (e.g. average, median) on the VB index data contained in the input folder.',

        epilog="""Example, running the algorithm and smoothing the group VB data with a FWHM of 3mm:
        python3 vb_statistical_final.py -pvb $path_to_folder_containing_spatially_normalised_vb$ -vv 1.8 -sm 3 -lb smooth_3
        python3 vb_statistical_final.py --path_vb $path_to_folder_containing_spatially_normalised_vb$ --voxel_volume 1.8 --smoothing 3 --label smooth_3
        """,

        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument('-pvb', '--path_vb', type=str, help='The path to the folder containing the spatially normalised volumetric VB data. For the function to identify the VB files they need to contain "vbi-vol_normalised" in their file name.', required=True)
    parser.add_argument('-vv', '--voxel_volume', type=float, help='The volume of the voxels used', required=True)
    parser.add_argument('-sm', '--smoothing', type=float, help='The FWHM of the Gaussian smoothing operator: by default no smoothing is applied', default=None)
    parser.add_argument('-lb', '--label', type=str, help='Custom label to be appended to the output file name', default=None)
    parser.add_argument('-a', '--alpha_level', type=str, help='The statistical significance level used (alpha); 0.05 is the default.', default=0.05)
    parser.add_argument('-s', '--struct_img', type=str, help='The path to the structural image used as a background in the plotting.', default=None)

    args = parser.parse_args()

    statistical_test(args.path_vb, args.voxel_volume, args.label, args.smoothing, args.alpha_level, args.struct_img)

if __name__ == '__main__':
    main()