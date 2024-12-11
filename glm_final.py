#!/usr/bin/env python
# coding: utf-8

# # The General linear model

# In[1]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import nilearn
import nibabel as nib
import argparse
import ants

# nilearn subpackages
from nilearn.plotting import plot_design_matrix
from nilearn.glm.first_level import FirstLevelModel
from nilearn import image
from nilearn import image
from nilearn.plotting import plot_contrast_matrix
from nilearn.plotting import plot_stat_map
from nilearn.glm import threshold_stats_img
from templateflow import api as tflow

# In[2]:

def events_tsv_func(psychopy_log_file, voxel_size, subject, task, verbose):
    """
    The function determines the events.tsv file which is used to compute the
    design matrix of the GLM analysis. Note that the function is designed for
    a particular experiment.
    """

    print(f"- Computing the events.tsv file for subject {subject} at {voxel_size}mm^3 resolution")

    # The texts stored by the psychopy log file during task and rest periods
    index_text = "motor_text: text = '   tap your right index'"
    thumb_text = "motor_text: text = '     tap your right thumb'"
    rest_text = "Rest_text: autoDraw = True"
    keypress = "Keypress: s"

    # Reading the psychopy log file
    log_file=pd.read_csv(psychopy_log_file,sep='\t')

    # Creating empty lists
    events_list = []
    duration = []
    s_key_responses = []

    # Does the user want to analyse thumb, index separately?
    if task:
      # The experimental trials
      trials = ['thumb', 'index', 'rest']

      # Looping over each row of the psychopy log file
      for row in range(log_file.shape[0]):

        # Searching for rest, index and thumb events
        if log_file.iloc[row, 2] == rest_text:
            events_tsv_temp = pd.DataFrame({'onset': [log_file.iloc[row, 0]], 'duration': [1.0], 'trial_type': [trials[2]]})
            events_list.append(events_tsv_temp)

        elif log_file.iloc[row, 2] == index_text:
            events_tsv_temp = pd.DataFrame({'onset': [log_file.iloc[row, 0]], 'duration': [1.0], 'trial_type': [trials[1]]})
            events_list.append(events_tsv_temp)

        elif log_file.iloc[row, 2] == thumb_text:
            events_tsv_temp = pd.DataFrame({'onset': [log_file.iloc[row, 0]], 'duration': [1.0], 'trial_type': [trials[0]]})
            events_list.append(events_tsv_temp)

        # Searching for s-keyboard responses that indicate an MRI volume
        elif log_file.iloc[row, 2] == keypress:
            s_key_responses_temp = pd.DataFrame({'onset': [log_file.iloc[row, 0]]})
            s_key_responses.append(s_key_responses_temp)

    # By default thumb and index events are grouped
    else:
      trials = ['task', 'rest']

      # Looping over each row of the psychopy log file
      for row in range(log_file.shape[0]):

        # Searching for rest, index and thumb events
        if log_file.iloc[row, 2] == rest_text:
            events_tsv_temp = pd.DataFrame({'onset': [log_file.iloc[row, 0]], 'duration': [1.0], 'trial_type': [trials[1]]})
            events_list.append(events_tsv_temp)

        elif log_file.iloc[row, 2] == index_text:
            events_tsv_temp = pd.DataFrame({'onset': [log_file.iloc[row, 0]], 'duration': [1.0], 'trial_type': [trials[0]]})
            events_list.append(events_tsv_temp)

        elif log_file.iloc[row, 2] == thumb_text:
            events_tsv_temp = pd.DataFrame({'onset': [log_file.iloc[row, 0]], 'duration': [1.0], 'trial_type': [trials[0]]})
            events_list.append(events_tsv_temp)

        # Searching for s-keyboard responses that indicate an MRI volume
        elif log_file.iloc[row, 2] == keypress:
            s_key_responses_temp = pd.DataFrame({'onset': [log_file.iloc[row, 0]]})
            s_key_responses.append(s_key_responses_temp)

    # Concatenating the lists into a data frame
    events_tsv_df = pd.concat(events_list, ignore_index = True)
    # Modifying the onsets so that the onset time starts from 0s
    events_tsv_df['onset'] = events_tsv_df['onset'] - events_tsv_df['onset'][0]

    # Creating a list of s-key responses
    if len(s_key_responses) > 0:
      s_key_responses_df = pd.concat(s_key_responses, ignore_index = True)

      # Computing the durations of each trial event
      for row in range(events_tsv_df.shape[0]):
          if row < events_tsv_df.shape[0] - 1:
              events_tsv_df.iloc[row, 1] = events_tsv_df.iloc[row+1,0] - events_tsv_df.iloc[row,0]
          else:
              events_tsv_df.iloc[row, 1] = s_key_responses_df.iloc[-1,:] - events_tsv_df.iloc[row,0]

    # In some cases, the s-values were not stored, accounting for these here; (807.3491 - 86.0823) is the total scanning time from one of the runs
    else:
      for row in range(events_tsv_df.shape[0]):
          if row < events_tsv_df.shape[0] - 1:
              events_tsv_df.iloc[row, 1] = events_tsv_df.iloc[row+1,0] - events_tsv_df.iloc[row,0]
          else:
              events_tsv_df.iloc[row, 1] = (807.3491 - 86.0823) - events_tsv_df.iloc[row,0]

    if verbose:
      events_tsv_df.to_csv("_sub-{}_{}mm_events".format(subject, voxel_size), sep='\t', index=False, header=True)

    print("- events.tsv file computed")

    return events_tsv_df

# In[3]:

"""First-level GLM"""

def first_level_glm(func_path, struct_path, events_tsv, tr, voxel_size, subject, smooth, analysis_type, thr, contrast, task, verbose, fw):

  '''
  Defining a function that computes the GLM using nilearn and outputs the unthresholded statistical 
  t-maps used to compute the group level analysis for multiple subject analysis. In addition, the Bonferroni
  corrected individual statistical maps are also computed.
  '''

  print(f"- Running the first level GLM analysis on subject {subject} for {voxel_size}mm^3 resolution")

  # Loading the images
  func_image = image.load_img(func_path)
  struct_image = image.load_img(struct_path)

  # will the data be smoothed?
  if smooth:
    # The FWHM suggested by Weibull et al., (2008) are used
    if fw is not None:
      fwhm = fw
    elif voxel_size == 1.8:
      fwhm = 2.67*voxel_size
    elif voxel_size == 2:
      fwhm = 2*voxel_size
    elif voxel_size == 2.5:
      fwhm = 1.28*voxel_size
    # Smoothing the functional image
    func_image = nilearn.image.smooth_img(func_image, fwhm)
    print(f"- Data smoothened with a FWHM of {fwhm}")
    smooth_text = f'smoothed_{fwhm}'
  else:
    smooth_text = 'non_smoothed'

  # What contrasts are we computing?
  if contrast==0:
    task_text = 'task-rest' # default
  elif contrast==1:
    task_text = 'thumb-rest'
  elif contrast==2:
    task_text = 'index-rest'
  elif contrast==3:
    task_text = 'index-thumb'

  # The confounds parameters used for the confound regressors. The confounds file of fMRIprep should be placed in the same directory as the func file to be analysed
  confounds_params, sample_mask = nilearn.interfaces.fmriprep.load_confounds(func_path,
                                                                             strategy = ("motion", "global_signal", "compcor", "scrub", "high_pass"),
                                                                             motion="full", # the six basic translation/rotation parameters; trans_x, trans_y, trans_z, rot_x, rot_y, rot_z
                                                                             global_signal="basic", # the basic global signal
                                                                             fd_threshold = 0.5,  # Thresholds to scrub outlier volumes based on Satterwtwaighte et al., (2013) and Power (2012)
                                                                             std_dvars_threshold = 2,
                                                                             compcor="anat_combined", # anat_combined noise components determined by PCA
                                                                             n_compcor = 6,
                                                                             demean=True)

  print("- Number of scrubbed volumes;", func_image.shape[3] - sample_mask.shape[0])

  # Parameters of the first level GLM model
  fmri_glm = FirstLevelModel(
    t_r=tr,
    noise_model="ar1",
    standardize=False,
    hrf_model="glover",
    drift_model=None,
    mask_img=None,
    subject_label=subject)

  # Fitting the GLM
  fmri_glm = fmri_glm.fit(func_image, events=events_tsv, confounds=confounds_params, sample_masks=sample_mask)

  # If the user would like to separate thumb, index events
  if task:
    # Computing the contrasts
    if contrast == 1:
      # thumb-rest contrast
      condition = {"thumb": np.zeros(fmri_glm.design_matrices_[0].shape[1]), "rest": np.zeros(fmri_glm.design_matrices_[0].shape[1])}
      condition["thumb"][2] = 1
      condition["rest"][1] = -1
      contrast_vector = condition["thumb"] + condition["rest"]

    elif contrast == 2:
      # index-rest contrast
      condition = {"index": np.zeros(fmri_glm.design_matrices_[0].shape[1]), "rest": np.zeros(fmri_glm.design_matrices_[0].shape[1])}
      condition["index"][0] = 1
      condition["rest"][1] = -1
      contrast_vector = condition["index"] + condition["rest"]

    elif contrast == 3:
      # index-thumb contrast
      condition = {"index": np.zeros(fmri_glm.design_matrices_[0].shape[1]), "thumb": np.zeros(fmri_glm.design_matrices_[0].shape[1])}
      condition["index"][0] = 1
      condition["thumb"][2] = -1
      contrast_vector = condition["index"] + condition["thumb"]

    elif contrast == 4:
      # thumb/index-rest contrast
      condition = {"index": np.zeros(fmri_glm.design_matrices_[0].shape[1]), "thumb": np.zeros(fmri_glm.design_matrices_[0].shape[1]), "rest": np.zeros(fmri_glm.design_matrices_[0].shape[1])}
      condition["index"][0] = 1
      condition["thumb"][2] = 1
      condition["thumb"][1] = -1
      contrast_vector = condition["index"] + condition["thumb"] + condition['rest']

  # If the user does not want to separate thumb/index events: default
  else:
      condition = {"task": np.zeros(fmri_glm.design_matrices_[0].shape[1]), "rest": np.zeros(fmri_glm.design_matrices_[0].shape[1])}
      condition["task"][1] = 1
      condition["rest"][0] = -1
      contrast_vector = condition["task"] + condition["rest"]

  #Calculating the t-map of the contrasts
  statmap = fmri_glm.compute_contrast(contrast_vector, stat_type='t', output_type='stat')
  output_statmap = '{}_stat_map_sub-{}_{}mm_{}.nii.gz'.format(smooth_text, subject, voxel_size, task_text)
  nib.save(statmap, output_statmap)

  design_matrix = fmri_glm.design_matrices_[0]
  plot_contrast_matrix(contrast_vector, design_matrix=design_matrix)

  if analysis_type == 0:
    #Plotting the statistical map with the Bonferroni correction
    cleanmap, threshold = threshold_stats_img(statmap, alpha=thr, height_control="bonferroni", cluster_threshold=10, two_sided=False)
    #saving the entire file as a nifti
    output_path = '{}_bonferronni_map_sub-{}_{}mm_{}.nii.gz'.format(smooth_text, subject, voxel_size, task_text)
    nib.save(cleanmap, output_path)

    #plotting the map
    print(f"- Bonferroni-corrected, p<{thr} threshold: {threshold:.3f}")
    x = plot_stat_map(
        cleanmap,
        bg_img=struct_image,
        threshold=threshold,
        display_mode="mosaic",
        cut_coords=5,
        black_bg=True,
        cmap='jet',
        title=f"{task_text} (p<{thr}, Bonferroni-corrected), threshold: {threshold:.3f}",)
    x.savefig("{}_bonferronni_map_sub-{}_{}mm_{}.pdf".format(smooth_text, subject, voxel_size, task_text))

  if analysis_type == 1:
    #Plotting the statistical map using the false discovery rate method
    cleanmap, threshold = threshold_stats_img(statmap, alpha=thr, height_control="fdr", two_sided=False)
    #saving the entire file as a nifti
    output_path = '{}_fdr_map_sub-{}_{}mm_{}.nii.gz'.format(smooth_text, subject, voxel_size, task_text)
    nib.save(cleanmap, output_path)

    #plotting the map
    print(f"- False Discovery rate = {thr} threshold: {threshold:.3f}")
    x = plot_stat_map(
        cleanmap,
        bg_img=struct_image,
        threshold=threshold,
        display_mode="mosaic",
        cut_coords=5,
        black_bg=True,
        cmap='jet',
        title=f"{task_text} (fdr={thr}), threshold: {threshold:.3f}",)
    x.savefig("{}_fdr_map_sub-{}_{}mm_{}.pdf".format(smooth_text, subject, voxel_size, task_text))

  if analysis_type == 2:
    #Adding a cluster threshold of 10 to the FDR map
    cleanmap, threshold = threshold_stats_img(statmap, alpha=thr, height_control="fdr", cluster_threshold=10, two_sided=False)

    #saving the entire file as a nifti
    output_path = '{}_fdr_clustered_map_sub-{}_{}mm_{}.nii.gz'.format(smooth_text, subject, voxel_size, task_text)
    nib.save(cleanmap, output_path)

    #plotting the map
    print(f"- False Discovery rate = {thr} threshold: {threshold:.3f} with a cluster threshold of 10")
    x = plot_stat_map(
        cleanmap,
        bg_img=struct_image,
        threshold=threshold,
        display_mode="mosaic",
        cut_coords=5,
        black_bg=True,
        cmap='jet',
        title=f"{task_text} (fdr={thr}), threshold: {threshold:.3f}, clusters > 10 voxels",)
    x.savefig("{}_fdr_clustered_map_sub-{}_{}mm_{}.pdf".format(smooth_text, subject, voxel_size, task_text))

    print(f"- First level analysis on subject {subject} at {voxel_size}mm^3 done")

    # Verbose section: if verbose is activated, plots of the design matrix, expected responsed etc will be generated
    if verbose:
      # Visualising the design matrix, expected response
      design_matrix = fmri_glm.design_matrices_[0]
      fig, (ax1) = plt.subplots(figsize=(10, 6), nrows=1, ncols=1)
      plot_design_matrix(design_matrix, ax=ax1)
      plt.tight_layout()
      plt.savefig("{}_design_matrix_sub-{}_{}mm.png".format(smooth_text, subject, voxel_size), dpi=400)
      plt.close()

      if task:
        if contrast == 1:
          plt.figure(figsize=(10, 6))
          plt.plot(design_matrix["thumb"])
          plt.xlabel("Time (s)")
          plt.ylabel("BOLD signal")
          plt.title("Expected Response")
          plt.tight_layout()
          plt.savefig("{}_expected_response_thumb_sub-{}_{}mm.png".format(smooth_text, subject, voxel_size))
          plt.close()

        elif contrast == 2:
          plt.figure(figsize=(10, 6))
          plt.plot(design_matrix["index"])
          plt.xlabel("Time (s)")
          plt.ylabel("BOLD signal")
          plt.title("Expected Response")
          plt.tight_layout()
          plt.savefig("{}_expected_response_index_sub-{}_{}mm.png".format(smooth_text, subject, voxel_size))
          plt.close()

        elif contrast == 3:
          plt.figure(figsize=(10, 6))
          plt.plot(design_matrix["index"])
          plt.xlabel("Time (s)")
          plt.ylabel("BOLD signal")
          plt.title("Expected Response")
          plt.tight_layout()
          plt.savefig("{}_expected_response_index_sub-{}_{}mm.png".format(smooth_text, subject, voxel_size))
          plt.close()

          plt.figure(figsize=(10, 6))
          plt.plot(design_matrix["thumb"])
          plt.xlabel("Time (s)")
          plt.ylabel("BOLD signal")
          plt.title("Expected Response")
          plt.tight_layout()
          plt.savefig("{}_expected_response_thumb_sub-{}_{}mm.png".format(smooth_text, subject, voxel_size))
          plt.close()

      else:
        plt.figure(figsize=(10, 6))
        plt.plot(design_matrix["task"])
        plt.xlabel("Time (s)")
        plt.ylabel("BOLD signal")
        plt.title("Expected Response")
        plt.axvline(x = tr*10, color = 'b')
        plt.axvline(x = tr*2*10, color = 'b')
        plt.tight_layout()
        plt.xlim(0,19*3)
        plt.savefig("{}_expected_response_task_sub-{}_{}mm.png".format(smooth_text, subject, voxel_size))
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(design_matrix["rest"])
        plt.xlabel("Time (s)")
        plt.ylabel("BOLD signal")
        plt.title("Expected Response")
        plt.tight_layout()
        plt.xlim(0,19*3)
        plt.savefig("{}_expected_response_task_rest-{}_{}mm.png".format(smooth_text, subject, voxel_size))
        plt.close()

      #Plotting the contrast plot
      plot_contrast_matrix(contrast_vector, design_matrix=design_matrix)
      plt.savefig("{}_contrast_matrix_sub-{}_{}mm.png".format(smooth_text, subject, voxel_size))
      plt.close()

  return fmri_glm, statmap

# In[4]:

def second_level_glm(second_level_input, thr, mni_image, voxel_size, slsm):
  """
  Performs a second level GLM on the data for group-wise analysis. In order for
  a second level analysis to be performed, the subjects need to be processed in
  a common space e.g. MNI space.
  """

  # Fetching the MNI template from the template flow data base if no alternate template is specified
  if mni_image is None:
    mni_image = tflow.get('MNI152NLin6Asym', desc=None, resolution=1, suffix='T1w', extension='nii.gz')
    mni_image = image.load_img(mni_image)
    nib.save(mni_image, "MNI152NLin6Asym_template.nii.gz")

  # Number of subjects
  n_subjects = len(second_level_input)
  print(f"- Performing group level analysis on {n_subjects} subjects")

  # Constructing the second level design matrix
  design_matrix = pd.DataFrame([1] * n_subjects, columns=['intercept'])


  # Implementing the GLM for multiple subject fMRI data
  second_level_model = nilearn.glm.second_level.SecondLevelModel(mask_img=None,
                                            target_affine=None, target_shape=None,
                                            smoothing_fwhm=slsm, memory=None,
                                            memory_level=1, verbose=0,
                                            n_jobs=1, minimize_memory=True)

  # The following function takes a second_level_input as a list of r-maps for the different subjects
  second_level_model = second_level_model.fit(second_level_input, design_matrix=design_matrix)

  # Computing the contrasts for the second level analysis: One sample testing with non-parametric multiple comparisons correction
  statmap = second_level_model.compute_contrast(second_level_contrast='intercept',
                                              second_level_stat_type='t', output_type='stat')

  print(f"- Second level contrast computed, performing Bonferroni correction with an alpha of {thr}")

  # Thresholding the raw stat map using Bonferroni
  thresholded_map, threshold = threshold_stats_img(statmap, alpha=thr, height_control="bonferroni", cluster_threshold=10, two_sided=False)

  # Saving the entire Bonferroni thresholded stat map as a nifti file
  output_path = 'clustered_bonferroni_map_grouplevel_{}mm_task-rest.nii.gz'.format(voxel_size)
  nib.save(thresholded_map, output_path)

  # Plotting the thresholded map
  print(f"Bonferroni = {thr} threshold: {threshold:.3f} with a cluster threshold of 10")
  x = plot_stat_map(
      thresholded_map,
      bg_img = mni_image,
      threshold=threshold,
      display_mode="mosaic",
      cut_coords=5,
      black_bg=True,
      cmap='jet',
      title=f"task-rest_bonferroni={thr}), threshold: {threshold:.3f}, clusters > 10 voxels",)
  #x.close()
  x.savefig("bonferroni_clustered_map_grouplevel_{}mm_task-rest.pdf".format(voxel_size))

  print(f"- Second level computation done")

# In[5]:

def normalisation_mni(map, transform_path, reference_image_path):
  """
  The function normalises the t-map outputted in the first level model to the
  MNI space. This is achieved using the transformation object outputted by
  fMRIprep. The file name is given by T1w_target-MNI152NLin2009cAsym_warp.h5.
  """
  transform = transform_path
  fixed = ants.image_read(reference_image_path) # The reference MNI image
  # The transformation will be applied with the antspy package
  input_img = ants.image_read(map)
  transformed_img = ants.apply_transforms(fixed=fixed, moving=input_img, transformlist=transform, interpolator='linear')

  # Saving the transformed image as a nifti
  label_base = os.path.splitext(map)[0]
  label_base = os.path.splitext(label_base)[0]

  output = f"{label_base}_normalised_to_MNI.nii.gz"
  nib.save(transformed_img, output)

  return None

# In[6]:

def main():
    parser = argparse.ArgumentParser(
        description='The algorithm runs the GLM on fMRIprep pre-processed data.'
        'The first-level argument may be used to run a first level GLM analysis,'
        'while second-level argument may be used to run a group level GLM analysis.'
        'Finally the normalise argument may be used to spatially normalise the'
        'output (t-maps) of the first level analysis to the MNI template so that'
        'a group level analysis may be performed.',

        formatter_class=argparse.RawTextHelpFormatter
    )

    subparsers = parser.add_subparsers(dest='command', help='Subcommands')

    first_level = subparsers.add_parser('first-level', help='Performs first level'
    'analysis on the inputted functional data. Make sure that the path to the'
    'functional data also contains the confounds.tsv file outputted by fMRIprep'
    'in the same directory.'
    )

    first_level.add_argument('-pf', '--path_func', nargs='+', type=str, help='The path or list of paths of the func files registered to native space or T1', required=True)
    first_level.add_argument('-pe', '--path_events', nargs='+', type=str, help='The path of the psycho.log file of the subject/s', required=True)
    first_level.add_argument('-vv', '--voxel_volume', type=float, help='The volume of the voxels used', required=True)
    first_level.add_argument('-sub', '--subject', nargs='+', type=str, help='The labels/s of the subject/s', required=True)

    # Non compulsary arguments
    first_level.add_argument('-ps', '--path_struct', nargs='+', type=str, help='The path or list of paths of the T1w images of the subject/s. Used to generate plots', default='MNI152TEMPLATE')
    first_level.add_argument('-tr', '--repetition_time', type=float, help='The repetition time', default=1.84)
    first_level.add_argument('-sm', '--smooth', action='store_true', help='Do you want to smooth the data? leave empty if not', default=False)
    first_level.add_argument('-fw', '--fwhm', type=float, help='Custom FWHM of the smoothing operator, by default the FWHM of Weibull et al., (2008) are used', default=None)
    first_level.add_argument('-type', '--analysis_type', type=int, help='The type of hypothesis tests performed, 0 for Bonferroni, 1 for FDR and 2 for clustered FDR', default=2)
    first_level.add_argument('-thr', '--alpha_threshold', type=float, help='The statistical threshold alpha, default = 0.05', default=0.05)
    first_level.add_argument('-c', '--contrast', type=int, help='Which contrast would you like computed? 0: task-rest, 1: thumb-rest, 2: index-rest, 3: index-thumb. Default is 0', default=0)
    first_level.add_argument('-ts', '--task', action='store_true', help='Do you want to analyse thumb and index separately? by default, thumb, index events are grouped. If this is activated, contrast needs to be set to 1,2 or 3', default=False)
    first_level.add_argument('-ver', '--verbose', action='store_true', help='Outputs a number of plots related to GLM design', default=False)

    # Normalisation of the t-maps to a standard MNI template
    normalisation = subparsers.add_parser('normalise', help='Normalises the t-map outputted in the first level analysis to the standard MNI template')
    normalisation.add_argument('-s', '--s_maps', type=str, help='Path to the t-maps outputted during the first level analysis')
    normalisation.add_argument('-t', '--transform', type=str, help='Path to the transform outputted by fMRIprep in the /anat directory: e.g. T1w_target-MNI152NLin2009cAsym_warp.h5')
    normalisation.add_argument('-rf', '--reference_img', type=str, help='Path to the T1w image normalised to MNI found in the /anat directory')

    # Second level (group level) analysis arguments - NOT REQUIRED TO RUN THE ALGORITHM
    second_level = subparsers.add_parser('second-level', help='Performs second level analysis using the t-maps outputted by the first level analysis. The t-maps need to be normalised to MNI space')
    second_level.add_argument('-s', '--s_maps', type=str, nargs='+', help='list containing the paths to the spatially normalised statistical maps of the volunteers', required=True)
    second_level.add_argument('-thr', '--alpha_threshold', type=float, help='The statistical threshold alpha, default = 0.05', default=0.05)
    second_level.add_argument('-mn', '--mni_img', type=str, help='The path to the MNI image', default=None)
    second_level.add_argument('-vv', '--voxel_volume', type=float, help='The volume of the voxels used', required=True)
    second_level.add_argument('-slsm', '--second_level_smoothing', type=float, help='The FWHM of the second level smoothing operator', default=None)

    args = parser.parse_args()

    if args.command == 'first-level':
      # Arrays to store the first level model objects for multiple subjects: used in the second level analysis
      flms = []
      s_maps = []

      # Running a First level analysis for each subject
      for sub in range(len(args.subject)):

        # Acquiring the events_tsv file
        events_tsv = events_tsv_func(args.path_events[sub], args.voxel_volume, args.subject[sub], args.task, args.verbose)

        flm, s_map = first_level_glm(args.path_func[sub], args.path_struct[sub],
                                     events_tsv,
                                     args.repetition_time, args.voxel_volume,
                                     args.subject[sub], args.smooth, args.analysis_type,
                                     args.alpha_threshold, args.contrast, args.task, args.verbose,
                                     args.fwhm)

        # Appending the first level outputs to the lists
        flms.append(flm)
        s_maps.append(s_map)

    # If a second level analysis is specified, run the second level analysis on the flms list
    if args.command == 'normalise':
      normalisation_mni(args.s_maps, args.transform, args.reference_img)

    if args.command == 'second-level':
      second_level_glm(args.s_maps, args.alpha_threshold, args.mni_img, args.voxel_volume, args.second_level_smoothing)

if __name__ == '__main__':
    main()