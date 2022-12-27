from pathlib import Path
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt

notebooks_path = Path.cwd()
repo_path = notebooks_path.parent

class ImageDataset():
    def __init__(self, set_name, csv_path : str = str(repo_path / 'data/info_images.csv')):
        df = pd.read_csv(csv_path, dtype={'ID': str}) #ID must be a string
        self.df = df[df['set_name'] == set_name]
        self.len = len(self.df)
        self.set_name = set_name
        self.IDs = list(self.df['ID'])
    def im_paths(self, preprocess = False):
        """get list of images paths

        Returns:
            list: list with images paths
        """
        folders_list = list(self.df['folder_path'])
        #add the path to the image
        paths_list = [str(Path(x) / (Path(x).stem + '.nii.gz')) for x in folders_list] if preprocess is False else [str(Path(x) / (Path(x).stem + '_norm.nii.gz')) for x in folders_list]
        return paths_list
    def labels_paths(self):
        """get list of images paths

        Returns:
            list: list with images paths
        """
        if self.set_name == 'Test':
            raise ValueError('Test set does not have labels! Pardon me!')
        folders_list = list(self.df['folder_path'])
        #add the path to the ground truth
        paths_list = [str(Path(x) / (Path(x).stem + '_seg.nii.gz')) for x in folders_list]
        return paths_list
    
class patient(ImageDataset):
    def __init__(self, ID, im_data: object):
        self.ID = ID
        self.df = im_data.df.loc[im_data.df['ID'] == self.ID] #Filter by ID
        self.len = len(self.df)
        self.set_name = im_data.set_name
        self.im_path = str(repo_path / self.im_paths()[0])
        self.labels_path = str(repo_path / self.labels_paths()[0])
        self.im_path_preprocessed = str(Path(self.im_path).parent / (Path(self.im_path).parent.stem + '_preprocessed.nii.gz'))
        self.im_path_norm = str(Path(self.im_path).parent / (Path(self.im_path).parent.stem + '_norm.nii.gz'))
    
    def im(self, format:str = 'sitk', preprocess = False): 
        #first the preprocessing
        if preprocess is False:
            sitk_im = sitk.ReadImage(self.im_path)
        elif preprocess is True:
            sitk_im = sitk.ReadImage(self.im_path_norm)
        else:
            raise ValueError('Preprocess must be a boolean')    
        #now the format
        if format=='sitk':
            return sitk_im
        elif format=='np':
            array_im = sitk.GetArrayFromImage(sitk_im)
            return array_im
        else:
            raise ValueError('Format not valid. Check allowed formats.')
    
    def labels(self, format:str='sitk'):
        sitk_label = sitk.ReadImage(str(repo_path / self.labels_paths()[0]))
        if format=='sitk':
            return sitk_label
        elif format=='np':
            array_label = sitk.GetArrayFromImage(sitk_label)
            return array_label
        else:
            raise ValueError('Format not valid. Check allowed formats.')
    
    def show(self, src:str = 'im' , slice :int = 135):
        if src=='im':
            im = self.im(format='np')
            cmap = 'gray'
        elif src=='labels':
            im = self.labels(format='np')
            cmap='jet'
        else:
            raise ValueError('What do you want to show? im or labels?')
        plt.figure(figsize=(10,10))
        plt.imshow(im[slice], cmap=cmap)
        plt.show()