from __future__ import division, print_function, absolute_import
import re
import glob
import os.path as osp
import warnings

from ..dataset import ImageDataset


class MyDataset(ImageDataset):
    """My dataset
    Dataset statistics:
        - identities: 18.
        - images: 1,870 / 900 (train) + 70 (query) + 900 (gallery).
    """
    dataset_dir = 'dataset-10-20'

    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        # allow alternative directory structure
        self.data_dir = self.dataset_dir
        data_dir = osp.join(self.data_dir, 'dataset-10-20')
        if osp.isdir(data_dir):
            self.data_dir = data_dir
        else:
            warnings.warn(
                'incorrect directory'
            )

        self.train_dir = osp.join(self.data_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.data_dir, 'query')
        self.gallery_dir = osp.join(self.data_dir, 'bounding_box_test')

        required_files = [
            self.data_dir, self.train_dir, self.query_dir, self.gallery_dir
        ]

        self.check_before_run(required_files)

        train = self.process_dir(self.train_dir, relabel=True)
        query = self.process_dir(self.query_dir, relabel=False)
        gallery = self.process_dir(self.gallery_dir, relabel=False)

        super(MyDataset, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, relabel=False):
        # Get a list of image paths
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        
        # Define the regular expression pattern for extracting information from file names
        pattern = re.compile(r'pid(\d+)_camid(\d+)_(\d+)\.jpg')

        # Create a set to store unique pids
        pid_container = set()
        
        # Iterate over image paths
        for img_path in img_paths:
            # Extract pid, camid, and _ from the file name using the pattern
            pid, _, _ = map(int, pattern.search(img_path).groups())
            
            # Add the pid to the set
            pid_container.add(pid)
        
        # Create a mapping from pid to label
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        # Create a list to store image data
        data = []
        
        # Iterate over image paths again
        for img_path in img_paths:
            # Extract pid, camid, and _ from the file name using the pattern
            pid, camid, _ = map(int, pattern.search(img_path).groups())
            
            # Assert the valid range for camid
            assert 0 <= camid <= 4
            
            # If relabel is True, use the pid2label mapping to relabel pid
            if relabel:
                pid = pid2label[pid]
            
            # Append the image path, pid, and camid to the data list
            data.append((img_path, pid, camid))
        
        return data
