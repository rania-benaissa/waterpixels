from scipy import io
import scipy
import os
###When running, you need to change the root value to the corresponding root directory where BSD500 is located
root = 'E:\DataSets\BSR\BSDS500\data'
PATH = os.path.join(root,'data\\groundTruth')


for sub_dir_name in ['train','test','val']:
    sub_pth = os.path.join(PATH,sub_dir_name)
    ##Create a new folder for the generated pictures to save
    save_pth = os.path.join(root,'data\\GT_convert',sub_dir_name)
    os.makedirs(save_pth,exist_ok=True)
    print('Start conversion'+sub_dir_name+'Content in folder')
    for filename in os.listdir(sub_pth):
        # Read all data in the mat file
        #mat file contains data stored in dictionary form
        #Include dict_keys(['__globals__','groundTruth','__header__','__version__'])
        #If you want to use the contour in'groundTruth']
        #x['groundTruth'][0][0][0][0][1] is the outline
        #x['groundTruth'][0][0][0][0][0] is the segmentation map
        data = io.loadmat(os.path.join(sub_pth,filename))
        edge_data = data['groundTruth'][0][0][0][0][1]
        #Store the normalized data: 0<x<1
        #So need to restore back to 0<x<255
        edge_data_255 = edge_data * 255
        new_img_name = filename.split('.')[0]+'.jpg'
        print(new_img_name)
        scipy.misc.imsave(os.path.join(save_pth,new_img_name), edge_data_255)  # save Picture