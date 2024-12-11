# Paper-code

## Installing the python packages needed
The python packages required to run both the glm_final.py and the vb_statistical_final.py code may be installed using the requirements.txt. To install via pip, run the following:
```
pip install -r requirements.txt
```

## General Linear Model
The code used to run the GLM analysis is the "glm_final.py" file. The file may be run via the command line and may perform two separate GLM analysis. Firstly, it can perform a subject level "first-level" analysis. In addition to this it may also perform a group-level "second-level analysis". The script may also be used to spatially normalise the first-level results to MNI space which is required to perform the second level analysis. 

### Example uses: first-level analysis
In this example, a first level analysis is performed on subject "20". The required arguments are the path to the preprocessed .nii.gz fMRI file, the path to the T1-weighted structural file, and the path to the events.tsv file outputted by PyschoPy during the experiment. In addition, the argument -vv specifies the voxel size of the pulse sequence used.
```
python3 glm_final.py first-level -pf $path_to_fmri_file$ -ps $path_to_T1w_image$ -pe $path_to_events.tsv$ -vv 1.8 -sub 20
```
### Example: first-level analysis with smoothing
In this example, we perform the same first level analysis but we also apply spatial smoothing to the data using a Gaussian kernel with a FWHM of 4mm.
```
python3 glm_final.py first-level -pf $path_to_fmri_file$ -ps $path_to_T1w_image$ -pe $path_to_events.tsv$ -vv 1.8 -sub 20 -sm -fw 4
```
Note that the first level analysis outputs the statistical t-maps required for the second-level analysis.

### Example: normalisation to MNI space
In order to perform a second level analysis, the t-maps need to be spatially normalised to a common space e.g. MNI. To spatially normalise with this script, one needs to specify; the path of the statistical maps, the transformation file outputted by for example fMRIprep (.h5 file), and the reference T1-weighted image normalised to MNI space of the volunteer. The algorithm will normalise using trilinear interpolation.
```
python3 glm_final.py normalise --s_maps $path_to_t-maps$ --transform $path_to_h5_file$ --reference_img $path_t1w_to_mni_img$
```
### Example: second-level analysis
The script may be used to perform a second-level analysis to provide group-level statistical inferences. To perform a second-level analysis, the paths to the spatially normalised t-maps, and the voxel volume of the image need to be passed as required arguments. In the below example, the t-maps of three volunteers are passed. In addition, a second-level smoothing operation with a FWHM of 4mm is also performed using the -slsm command in the following example:
```
python3 glm_final.py second-level --s_maps $path_to_norm_t-map1$ $path_to_norm_t-map2$ $path_to_t-norm_map3$ -vv 1.8 -slsm 4
```

## Statistical Analysis with the VB index
The code used to perform statistical testing with the VB index is the "vb_statistical_final.py" file which may be run via the command line. The file takes as input the path of the folder containing the spatially normalised VB index data. The files contained inside the folder need to have the label "vb-vol_normalised" as part of their filename. The code will search through the path specified and find files with the label vb-vol_normalised in their name, compute summary statistics (e.g. mean, median) across all files and then perform the statistical test. The code performs a one-sample t-test to determine which voxels exhibit VB indices that are statistically larger than the median VB index across all subjects. Bonferroni correction is applied. The code has some flexibility and enables you to choose a p-value (default is 0.05), to smoothen the VB data prior the statistical analysis, and to assign a custom label to the output files.

### Example: Smoothing the VB data with a FWHM of 3mm and then performing the statistical test
```
python3 vb_statistical_final.py -pvb $path_to_folder_containing_vb_data$ -vv 1.8 -sm 3 -lb smoothing_3
```

## Spatial similarity of activation maps
The spatial similarity of the activation maps obtained with the for example second level GLM or the VB analysis may be assessed using the Jaccard_Dice_coefficient.py file to compute the Jaccard index, Dice coefficient or overlap coefficient between two input activation maps. The code first binarises the input activation maps using a user-specified threshold (default = 0). The threshold may be for example the Bonferroni corrected t-value for p<0.05 as outputted by the GLM and VB algorithm previously.

### Example: Computing the Jaccard index of two activation maps
```
python3 Jaccard_Dice_coefficient.py -sm1 $path_to_activation_map1$ -sm2 $path_to_activation_map2$ -thr1 5.2 -thr2 5.4 
```
### Example: Computing the Jaccard index, Dice coefficient and overlap coefficient of two activation maps
```
python3 Jaccard_Dice_coefficient.py -sm1 $path_to_activation_map1$ -sm2 $path_to_activation_map2$ -cf 4 -thr1 5.2 -thr2 5.4 
```
