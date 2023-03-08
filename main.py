import os
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from skimage.transform import resize            #out = resize(dtest,(5,6), order=0, preserve_range=True)
from scipy.ndimage.morphology import distance_transform_edt, binary_fill_holes
from skimage.morphology import skeletonize, skeletonize_3d, remove_small_objects

# For lung segmentation
from lungmask import mask
import SimpleITK as sitk
from radiomics import featureextractor

class ExtractVesselFeatures:
    def __init__(self, path_ct, path_tumor_seg, path_vessel_seg, path_lung_seg=None, show_check=False, resize_111=None):
        self.min_v, self.max_v = -1000, 100
        # paths
        self.path_ct = path_ct
        self.path_tumor_seg = path_tumor_seg
        self.path_vessel_seg = path_vessel_seg
        # Loading ct, tumor and vessel segmentation
        self.ct = nib.load(self.path_ct)            ; self.ct_array = self.ct.get_fdata()             ; self.ct_affine=self.ct.affine
        tumor = nib.load(self.path_tumor_seg)       ; self.tumor_array = tumor.get_fdata()            ; self.tumor_affine=tumor.affine
        vessel_seg = nib.load(self.path_vessel_seg) ; self.vessel_seg_array = vessel_seg.get_fdata()  ; self.vessel_seg_affine=vessel_seg.affine        
        assert self.ct_array.shape==self.tumor_array.shape, "ct and tumor have diff shapes"
        assert self.ct_array.shape==self.vessel_seg_array.shape, "ct and vessel have diff shapes"
        #Obtain lung & vessels_only & dimension
        self.lung_array = self.obtain_lung_seg() if path_lung_seg == None else nib.load(path_lung_seg).get_fdata()
        self.dimension_orig = np.diag(np.abs(self.ct_affine))[0:3]
        self.vessel_array = self.obtain_vessel()# Should be here not after resize not before lung !

        if resize_111:
            print('Resizing to 111 spacing', )
            # the main diogonal starts with +-[1, 1, 1]
            for i in range(3): # Should be defined here before it is used for saving 
                self.ct_affine[i][i] = self.ct_affine[i][i]/abs(self.ct_affine[i][i])
            self.ct_array     = self.resize_1mm(self.ct_array, save_nii = os.path.join(resize_111, os.path.basename(path_ct[:-7])+'_ct_111.nii.gz'))
            self.lung_array   = self.resize_1mm(self.lung_array, save_nii = os.path.join(resize_111, os.path.basename(path_ct[:-7])+'_lung_111.nii.gz'))
            self.tumor_array  = self.resize_1mm(self.tumor_array, save_nii = os.path.join(resize_111, os.path.basename(path_ct[:-7])+'_tumor_111.nii.gz'))
            self.vessel_path = os.path.join(resize_111, os.path.basename(path_ct[:-7])+'_vessel.nii.gz') # Should be corrected later
            self.vessel_array = self.resize_1mm(self.vessel_array,save_nii = self.vessel_path)
            #self.vessel_seg_array = self.resize_1mm(self.vessel_seg_array,save_nii = os.path.join(resize_111, os.path.basename(path_ct[:-7])+'_vessel_seg_111.nii.gz'))
            self.vessel_seg_array = self.obtain_vessel_seg_array()
            self.dimension = [1,1,1]
            # the main diogonal starts with +-[1, 1, 1]

        self.tumor_core, self.tumor_inner, self.tumor_outer, self.chl_tumor = self.calculate_core_inner_outer()
        if show_check:
            os.makedirs('savefig', exist_ok=True)
            self.show_check()
            self.show_check_tumor(self.chl_tumor)

    def obtain_vessel_seg_array(self):
        vessel_seg_array = np.clip(self.vessel_array , self.min_v, self.max_v)
        vessel_seg_array = vessel_seg_array - self.min_v
        vessel_seg_array[vessel_seg_array != 0] = 1
        vessel_seg_array = vessel_seg_array.astype(np.uint8)
        return vessel_seg_array

    def resize_1mm(self, array_3d, save_nii = False): 
        array_3d = array_3d.astype(np.float32)
        phy = array_3d.shape*self.dimension_orig    # physical size
        iso = 1 # isotropic voxel
        new_size = np.round(phy/iso)   # new resampling size after interpolation 
        array_3d_interp = resize(array_3d, (new_size[0],new_size[1],new_size[2]), order=1, preserve_range=True)
        pixel_dim = np.round(phy/new_size)
        if save_nii:
            array_interp_NIFTI = nib.Nifti1Image(array_3d_interp, self.ct_affine)
            array_interp_NIFTI.to_filename(save_nii)
        return array_3d_interp
    
        
    def calculate_core_inner_outer(self, outer_dist=10, img_resolution=1):
        # dilate and erode the mask
        tumor_core = np.empty(self.tumor_array.shape)
        tumor_inner = np.empty(self.tumor_array.shape)
        tumor_outer = np.empty(self.tumor_array.shape)
        tumor_outer0 = np.empty(self.tumor_array.shape)
        tumor_outer1 = np.empty(self.tumor_array.shape)
        
        outer_num = np.round(outer_dist/img_resolution)
        # loop through the nonzero slices
        sum_vec = self.tumor_array.sum(0).sum(0)   # 1D, gives the number of non-zero pixels in each slide, [ 0,   0,   0, 141, 190, 228, 0, 0]
        sele_idx = np.nonzero(sum_vec)             # array([3, 4, 5])
        for k in sele_idx[0]:
            tumor_2D = self.tumor_array[:,:,k] * 1
            tumor_2D = tumor_2D.astype(np.uint8)
            mask_dist = distance_transform_edt(tumor_2D)  #(512, 512)  #computes the distance from non-zero (i.e. non-background) points to the nearest zero (i.e. background) point.
            max_dist = np.amax(mask_dist)
            mask_erode = mask_dist > (max_dist/2)   # they are far from background in the tumor #https://stackoverflow.com/questions/44770396/how-does-the-scipy-distance-transform-edt-function-work
            mask_erode = mask_erode * 1
            tumor_core[:, :, k] = mask_erode
            tumor_inner[:, :, k] = tumor_2D - mask_erode
            mask_dist2 = distance_transform_edt(1-tumor_2D)
            mask_dilate = mask_dist2 <= outer_num
            mask_dilate = mask_dilate * 1
            tumor_outer[:, :, k] = mask_dilate
        tumor_outer = tumor_outer - self.tumor_array

        sum_vec0 = self.tumor_array.sum(2).sum(1)
        sele_idx0 = np.nonzero(sum_vec0) 
        for k0 in sele_idx0[0]:
            tumor_2D0 = self.tumor_array[k0,:,:] * 1
            tumor_2D0 = tumor_2D0.astype(np.uint8)
            mask_dist2_0 = distance_transform_edt(1-tumor_2D0)
            mask_dilate0 = mask_dist2_0 <= outer_num
            tumor_outer0[k0, :, :] = mask_dilate0
        tumor_outer0 = tumor_outer0 - self.tumor_array

        sum_vec1 = self.tumor_array.sum(2).sum(0)
        sele_idx1 = np.nonzero(sum_vec1) 
        for k1 in sele_idx1[0]:
            tumor_2D1 = self.tumor_array[:,k1,:] * 1
            tumor_2D1 = tumor_2D1.astype(np.uint8)
            mask_dist2_1 = distance_transform_edt(1-tumor_2D1)
            mask_dilate1 = mask_dist2_1 <= outer_num
            tumor_outer1[:,k1, :] = mask_dilate1
        tumor_outer1 = tumor_outer1 - self.tumor_array
        
        tumor_outer_20 = np.logical_or(tumor_outer, tumor_outer0)
        tumor_outer_combined = np.logical_or(tumor_outer_20, tumor_outer1)
        lung_tumor = np.logical_or(self.tumor_array, self.lung_array)
        tumor_outer_combined = np.logical_and(lung_tumor, tumor_outer_combined).astype(np.uint8)
        chl = sele_idx[0][sele_idx[0].shape[0]//2]
        return tumor_core, tumor_inner, tumor_outer_combined, chl
    
    def skeletonize_vessel(self, img, method='lee'):
        if method=='lee':
            skeleton = skeletonize(img, method='lee')
        else:
            skeleton = skeletonize(img)
        return skeleton
    
    def obtain_lung_seg(self):
        print('Obtaining lung segmentation...')
        #from lungmask import mask
        #import SimpleITK as sitk
        input_image = sitk.ReadImage(self.path_ct)
        segmentation = mask.apply(input_image)
        segmentation[segmentation!=0] = 1
        segmentation = np.transpose(segmentation,[1,2,0])
        for i in range(segmentation.shape[2]):
            segmentation[:,:,i] = np.rot90( np.fliplr(segmentation[:,:,i]),1)
        return segmentation
    
    def show_check(self):
        fig = plt.figure(figsize= (16,4))
        fig.add_subplot(1,4,1)
        chl = self.ct_array.shape[2]//2 - 10
        plt.imshow(self.ct_array[:,:,chl],cmap='gray', vmin=-1000, vmax=200)
        fig.add_subplot(1,4,2)
        plt.imshow(self.lung_array[:,:,chl],cmap='gray', vmin=0, vmax=1)
        fig.add_subplot(1,4,3)
        plt.imshow(self.tumor_array[:,:,chl],cmap='gray', vmin=0, vmax=1)
        fig.add_subplot(1,4,4)
        plt.imshow(self.vessel_array[:,:,chl],cmap='gray', vmin=-1000, vmax=200)
        plt.savefig(os.path.join('savefig' , os.path.basename(self.path_ct[:-7])+'.png'))
        plt.show()
        
    def show_check_tumor(self, chl):
        fig = plt.figure(figsize= (12,4))
        fig.add_subplot(1,3,1)
        plt.imshow(self.tumor_core[:,:,chl],cmap='gray', vmin=0, vmax=1)
        fig.add_subplot(1,3,2)
        plt.imshow(self.tumor_inner[:,:,chl],cmap='gray', vmin=0, vmax=1)
        fig.add_subplot(1,3,3)
        plt.imshow(self.tumor_outer[:,:,chl],cmap='gray', vmin=0, vmax=1)
        plt.savefig(os.path.join('savefig' , os.path.basename(self.path_ct[:-7])+'_.png'))
        plt.show()
        
    def obtain_vessel(self):
        min_v, max_v = self.min_v, self.max_v
        vessel_array = np.clip(self.ct_array, min_v, max_v)
        vessel_array = (vessel_array-min_v)/(max_v-min_v)
        vessel_array =  np.multiply(vessel_array, self.vessel_seg_array)
        vessel_array =  np.multiply(vessel_array, self.lung_array)
        vessel_array = vessel_array*(max_v-min_v) + min_v
        return vessel_array
        
    def normalize_01(self):
        pass
    
    def histogram_11(self, bins=11):
        vessel_pixels = self.vessel_array[self.tumor_outer == 1].reshape(-1)
        frq1, edges1 = np.histogram(vessel_pixels, bins=bins, range=(self.min_v, self.max_v))
        # normalize to pdf
        frq1 = frq1 / (np.sum(frq1) + 1e-10)
        return frq1

    def saving_outer_tumor_and_vessel(self, save_folder='./outer_tumor_and_vessel'):
        print('saving outer tumor & vessel in', save_folder)
        #Let's have both with the same ct.affine here
        os.makedirs(save_folder, exist_ok=True)
        #vessel_NIFTI = nib.Nifti1Image(self.vessel_array, self.ct_affine)
        #vessel_path = os.path.join(save_folder, os.path.basename(path_ct[:-7])+'_vessel.nii.gz')
        #vessel_NIFTI.to_filename(vessel_path)
        outer_NIFTI = nib.Nifti1Image(self.tumor_outer, self.ct_affine)
        outer_path = os.path.join(save_folder, os.path.basename(path_ct[:-7])+'_tumor_outer_seg.nii.gz')
        outer_NIFTI.to_filename(outer_path)
        return outer_path , self.vessel_path
        
        
    def radiomic_features(self, path_vesel, path_tumor_outer_seg , only_first_order = True):
        #self.saving_outer_tumor_and_vessel()
        imagePath = os.path.join(path_vesel)
        labelPath = os.path.join(path_tumor_outer_seg)
        extractor = featureextractor.RadiomicsFeatureExtractor(binWidth=100,  sigma=[1, 2, 3], verbose=False)
        if only_first_order:
            extractor.disableAllFeatures()
            extractor.enableFeatureClassByName('firstorder')
        result = extractor.execute(imagePath, labelPath)
        return result

####################################################################################################################################
path_ct_folder = '/Code/ct_tumor_files/'                            #Exp: 0001_414826_170323.nii.gz          
path_t_folder  = '/Code/ct_tumor_files/'                            #Exp: 0001_414826_170323_RTS_L1.nii.gz   
path_v_folder  = '/Code/vessel_files/'   # MATLAB Output folder     #Exp: 0001_414826_170323_v.nii.gz        

only_first_order_features = False
save_img_matching = True

if __name__=='__main__':
    results = []
    for ct_file_name in os.listdir(path_ct_folder):
        if 'RTS' not in ct_file_name:
            path_ct = os.path.join(path_ct_folder, ct_file_name)
            path_t = os.path.join(path_ct_folder, ct_file_name[:-7] + '_RTS_L1.nii.gz')
            path_v = os.path.join(path_v_folder, ct_file_name[:-7] + '_v.nii.gz')
        else:
            continue
        
        resize_111 = '/Code/outer_tumor_and_vessel'
        # create instance
        case_i = ExtractVesselFeatures(path_ct,path_t,path_v, show_check=save_img_matching, resize_111=resize_111)
        #save outer_tumor and vessel
        path_outer_tumor_and_vessel = '/Code/outer_tumor_and_vessel'
        path_tumor_outer_seg, path_vesel = case_i.saving_outer_tumor_and_vessel(save_folder = path_outer_tumor_and_vessel)

        result = case_i.radiomic_features(path_vesel, path_tumor_outer_seg , only_first_order = only_first_order_features)
        result["ct"] = ct_file_name
        results.append((result))
    
    df = pd.DataFrame(results)
    df.to_csv( 'featurs.csv' , index=False)