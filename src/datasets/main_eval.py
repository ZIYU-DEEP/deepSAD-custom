from .custom_eval import CustomADDataset_eval


def load_dataset_eval(data_path,
                      normal_data_file: str='',
                      abnormal_data_file: str='',
                      random_state=1):
    """Loads the dataset."""


    dataset = CustomADDataset_eval(root=data_path,
                                   random_state=random_state,
                                   normal_data_file=normal_data_file,
                                   abnormal_data_file=abnormal_data_file)

    return dataset
