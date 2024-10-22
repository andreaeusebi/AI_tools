from pathlib import Path
import glob
import os

from datasets import Dataset, DatasetDict, Image
from huggingface_hub import login

def createHfDataset(images_path_ : str, labels_path_ : str):
    """ Create an Huggingface Dataset object from paths to images and labels. """
    
    ## Check that given paths exist
    assert(Path(images_path_).exists())
    assert(Path(labels_path_).exists())

    ## Retrieve all images paths inside each path
    images_paths = glob.glob(os.path.join(images_path_, "*"))
    labels_paths = glob.glob(os.path.join(labels_path_, "*"))

    dataset = Dataset.from_dict( {"image": sorted(images_paths),
                                  "label": sorted(labels_paths)} )
    dataset = dataset.cast_column("image", Image())
    dataset = dataset.cast_column("label", Image())

    return dataset

def uploadSegmentationDatasetToHf(hf_token_          : str,
                                  dataset_name_      : str,
                                  train_images_path_ : str,
                                  train_labels_path_ : str,
                                  valid_images_path_ : str,
                                  valid_labels_path_ : str):
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
                            "validation": valid_dataset} )
    
    ## Push to hub
    dataset.push_to_hub(dataset_name_, private=True)

    return

##### ----- Test Script ----- #####

def test():
    uploadSegmentationDatasetToHf(
        hf_token_="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",
        dataset_name_="eusandre95/test_segmentation",
        train_images_path_="/home/andrea/raymond/pallet_detection_datasets/pallet_detection_dataset_v0.1/rgb/train/",
        train_labels_path_="/home/andrea/raymond/pallet_detection_datasets/pallet_detection_dataset_v0.1/segmentation/train/",
        valid_images_path_="/home/andrea/raymond/pallet_detection_datasets/pallet_detection_dataset_v0.1/rgb/val/",
        valid_labels_path_="/home/andrea/raymond/pallet_detection_datasets/pallet_detection_dataset_v0.1/segmentation/val/"
    )

if __name__ == "__main__":
    test()
