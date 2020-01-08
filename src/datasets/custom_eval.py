from torch.utils.data import DataLoader
from base.base_dataset_eval import BaseADDataset_eval
from base.custom_dataset_eval import CustomDataset_eval


class CustomADDataset_eval(BaseADDataset_eval):

    def __init__(self,
                 root: str,
                 random_state=None,
                 normal_data_file='skip',
                 abnormal_data_file=''):
        super().__init__(root)

        # Define normal and outlier classes
        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = (0,)
        self.outlier_classes = (1,)

        # Get train set
        self.all_set = CustomDataset_eval(root=self.root,
                                          random_state=random_state,
                                          normal_data_file=normal_data_file,
                                          abnormal_data_file=abnormal_data_file)

    def loaders(self,
                batch_size: int,
                shuffle_all=True,
                num_workers: int = 0) -> (DataLoader, DataLoader):
        all_loader = DataLoader(dataset=self.all_set,
                                batch_size=batch_size,
                                shuffle=shuffle_all,
                                num_workers=num_workers,
                                drop_last=True)

        return all_loader
