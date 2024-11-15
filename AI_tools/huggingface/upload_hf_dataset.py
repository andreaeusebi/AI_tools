## Standard modules
from pathlib import Path
import glob
import os
from typing import Optional

## Huggingface modules
from datasets import Dataset, DatasetDict, Image
from huggingface_hub import login, upload_file

def createHfDataset(images_path_ : str, labels_path_ : str):
    """ Create an Huggingface Dataset object from paths to images and labels. """
    
    ## Check that given paths exist
    assert(Path(images_path_).exists())
    assert(Path(labels_path_).exists())

    ## Retrieve all images paths inside each path
    images_paths = glob.glob(os.path.join(images_path_, "*"))
    labels_paths = glob.glob(os.path.join(labels_path_, "*"))

    dataset = Dataset.from_dict( {"pixel_values": sorted(images_paths),
                                  "label": sorted(labels_paths)} )
    dataset = dataset.cast_column("pixel_values", Image())
    dataset = dataset.cast_column("label", Image())

    return dataset

def uploadSegmentationDatasetToHf(hf_token_           : str,
                                  dataset_name_       : str,
                                  train_images_path_  : str,
                                  train_labels_path_  : str,
                                  valid_images_path_  : str,
                                  valid_labels_path_  : str,
                                  dataset_card_fpath_ : Optional[str] = None):
    """ Upload a Semantic Segmentation dataset to Huggingface Hub. """

    ## Login to Huggingface
    login(hf_token_)

    ## Create Dataset objects
    train_dataset = createHfDataset(images_path_=train_images_path_,
                                    labels_path_=train_labels_path_)
    
    valid_dataset = createHfDataset(images_path_=valid_images_path_,
                                    labels_path_=valid_labels_path_)
    
    ## Create DatasetDict
    dataset = DatasetDict( {"train": train_dataset,
                            "valid": valid_dataset} )
    
    ## Push to hub
    dataset.push_to_hub(dataset_name_, private=True)

    ## Create Dataset card
    if dataset_card_fpath_ is not None:
        assert(Path(dataset_card_fpath_).exists())

        upload_file(path_or_fileobj=dataset_card_fpath_,
                    path_in_repo="README.md",
                    repo_id=dataset_name_,
                    token=hf_token_,
                    repo_type="dataset",
                    commit_message="Test Readme code.")

    return

##### ----- Test Script ----- #####

def test():
    uploadSegmentationDatasetToHf(
        hf_token_="",
        dataset_name_="eusandre95/test_segmentation",
        train_images_path_="/home/andrea/raymond/pallet_detection_datasets/pallet_detection_dataset_v0.1/rgb/train/",
        train_labels_path_="/home/andrea/raymond/pallet_detection_datasets/pallet_detection_dataset_v0.1/segmentation/train/",
        valid_images_path_="/home/andrea/raymond/pallet_detection_datasets/pallet_detection_dataset_v0.1/rgb/val/",
        valid_labels_path_="/home/andrea/raymond/pallet_detection_datasets/pallet_detection_dataset_v0.1/segmentation/val/",
        dataset_card_fpath_="/home/andrea/raymond/pallet_detection_datasets/pallet_detection_dataset_v0.1/README.md"
    )

if __name__ == "__main__":
    test()
