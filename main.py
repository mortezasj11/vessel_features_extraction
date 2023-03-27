import os
import pandas as pd
from src.vessel_features import ExtractVesselFeatures

path_ct_folder = '/Code/ct_tumor_files/'                            #Exp: 0001_414826_170323.nii.gz          
path_t_folder  = '/Code/ct_tumor_files/'                            #Exp: 0001_414826_170323_RTS_L1.nii.gz   
path_v_folder  = '/Code/vessel_files/'   # MATLAB Output folder     #Exp: 0001_414826_170323_v.nii.gz        

outer_dist = 5  # distance in mm 

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
        case_i = ExtractVesselFeatures(path_ct,path_t,path_v, show_check=save_img_matching, resize_111=resize_111, outer_dist=outer_dist)
        #save outer_tumor and vessel
        path_outer_tumor_and_vessel = '/Code/outer_tumor_and_vessel'
        path_tumor_outer_seg, path_vesel = case_i.saving_outer_tumor_and_vessel(save_folder = path_outer_tumor_and_vessel)

        result = case_i.radiomic_features(path_vesel, path_tumor_outer_seg , only_first_order = only_first_order_features)
        result["ct"] = ct_file_name
        results.append((result))
    
    df = pd.DataFrame(results)
    df.to_csv( 'featurs.csv' , index=False)