## Standard modules
import json
from   typing import List, Optional
import logging

## Segments.ai modules
from segments             import SegmentsClient
from segments.huggingface import release2dataset
from segments.utils       import get_semantic_bitmap

## Huggingface modules
from huggingface_hub import login
from datasets        import DatasetDict

## Custom modules
import py_utils.log


def _convert_segmentation_bitmap(example):
    """
    Convert an instance bitmap into a segmentation bitmap.

    Args:
        example (dict): A dictionary containing instance bitmap and label annotations.

    Returns:
        dict: A dictionary with the semantic segmentation bitmap.
    """

    return {
        "label.segmentation_bitmap":
            get_semantic_bitmap(
                instance_bitmap = example["label.segmentation_bitmap"],
                annotations     = example["label.annotations"],
            )
    }

def _convertSegBitmapsToUint8(example):
    """
    Converts segmentation bitmaps to 8-bit grayscale.

    Args:
        example (dict): A dictionary containing a segmentation bitmap image.

    Returns:
        dict: A dictionary with the converted 8-bit segmentation bitmap.
    """

    example["label.segmentation_bitmap"] = example["label.segmentation_bitmap"].convert("L")

    return {
        "label.segmentation_bitmap": example["label.segmentation_bitmap"]
    }

def segments2HfDataset(segments_api_key     : str,
                       segments_dataset_id  : str,
                       segments_release     : str,
                       hf_token             : str,
                       hf_dataset_name      : str,
                       label_status_to_keep : List[str],
                       multiple_splits      : bool,
                       splits_fpath         : Optional[str] = None):
    """
    Converts a Segments.ai release into a Hugging Face Dataset and uploads it.

    This function fetches a Segments.ai dataset release, converts it into a Hugging Face Dataset,
    filters the dataset based on label statuses, and optionally splits it into train and
    validation sets. The final dataset is pushed to the Hugging Face Hub as a private repository.

    Args:
        segments_api_key (str): The API key for accessing Segments.ai.
        segments_dataset_id (str): The ID of the Segments.ai dataset.
        segments_release (str): The release version of the dataset.
        hf_token (str): The Hugging Face Hub authentication token.
        hf_dataset_name (str): The name for the Hugging Face Hub dataset.
        label_status_to_keep (List[str]): A list of label statuses to filter by.
        multiple_splits (bool): Whether to split the dataset into train and validation sets.
        splits_fpath (Optional[str]): Path to a JSON file specifying the dataset splits.
            Splits organization must be specified as "image file name" : "split" key-value pairs,
            for example:
            "rgb_0000001.png": "train",
            "rgb_0000002.png": "valid"
            ...

    Raises:
        KeyError: If an image in the dataset is not present in the provided splits dictionary.
        FileNotFoundError: If the splits file is required but not provided.
    """
    
    # Create a logger object
    logger = py_utils.log.getCustomLogger(logger_name_=__name__,
                                          node_name_="segments2HfDataset",
                                          log_handler_=logging.StreamHandler(),
                                          logging_level_=logging.INFO)
    
    logger.debug(f"Begin of function.")
    
    ## Initialize a SegmentsDataset from the release file
    segments_client  = SegmentsClient(segments_api_key)
    segments_release = segments_client.get_release(segments_dataset_id,
                                                   segments_release)

    # Login to Huggingface
    login(hf_token)

    # Convert Segments release to an Huggingface Dataset
    hf_dataset = release2dataset(segments_release, download_images=True)

    logger.info(f"hf_dataset original:\n{hf_dataset}")

    # Keep only labeled images with proper status
    hf_dataset = hf_dataset.filter(lambda x: x["status"] in label_status_to_keep)

    logger.info(f"hf_dataset filtered:\n{hf_dataset}")

    # Convert instances bitmap to segmentation bitmap
    semantic_dataset = hf_dataset.map(_convert_segmentation_bitmap)

    #### THIS DOESN'T WORK ####
    # semantic_dataset = hf_dataset.map(_convertSegBitmapsToUint8)

    ## Dataset splits handling. First of all, check if dataset must be divided into splits
    if multiple_splits is True:

        ## If splits file is provided, divide the dataset into train and valid splits
        ds_splits = {}

        # Check if if file exists
        if splits_fpath is not None:
            # Open file and
            ds_splits = json.load(open(splits_fpath))
            ds_splits = {k: v for k, v in ds_splits.items()}
        else:
            ## If splits file not provided, divide randomly
            logger.error(f"'splits_fpath' not provided! This option is not supported yet!")
            exit(1)
        
        logger.info(f"ds_splits: {ds_splits}")

        ## Filter the dataset into train and valid splits
        try:
            train_dataset = semantic_dataset.filter(lambda x: ds_splits[x["name"]] == "train")
            valid_dataset = semantic_dataset.filter(lambda x: ds_splits[x["name"]] == "valid")
        except KeyError as err:
            logger.error(f"Error! Img '{err.args[0]}' isn't in the dataset splits JSON file!")
            exit(1)

        logger.info(f"Train split:\n{train_dataset}")
        logger.info(f"Valid split:\n{valid_dataset}")

        # Compose the two splits into a single dataset
        semantic_dataset = DatasetDict({"train": train_dataset, "valid": valid_dataset})

    ## Rearrange dataset columns s expected by Segformer model
    semantic_dataset = semantic_dataset.rename_column('image', 'pixel_values')
    semantic_dataset = semantic_dataset.rename_column('label.segmentation_bitmap', 'label')
    semantic_dataset = semantic_dataset.remove_columns(
        ['name', 'uuid', 'status', 'label.annotations']
    )

    logger.info(f"Final semantic dataset:\n{semantic_dataset}")
    if multiple_splits is True:
        logger.info(f"First row of train split:\n{semantic_dataset['train'][0]}")
    else:
        logger.info(f"First row of the dataset:\n{semantic_dataset[0]}")

    ## Push to HF hub as private repo
    semantic_dataset.push_to_hub(hf_dataset_name, private=True)

    logger.debug(f"Function completed.")

###### ----- Test Script ----- ######

def test():
    ## segments2HfDataset() arguments
    segments_api_key     = ""
    segments_dataset_id  = "andrea_eusebi/pallet_semantic_segmentation_loco"
    segments_release     = "v0.2"
    hf_token             = ""
    hf_dataset_name      = "eusandre95/pallet_semantic_segmentation_loco"
    label_status_to_keep = ["LABELED", "REVIEWED"]
    multiple_splits      = True
    splits_fpath         = "/home/andrea/raymond/AI_tools/segments_ai/pallet_segmentation_loco_ds_splits.json"

    segments2HfDataset(
        segments_api_key     = segments_api_key,
        segments_dataset_id  = segments_dataset_id,
        segments_release     = segments_release,
        hf_token             = hf_token,
        hf_dataset_name      = hf_dataset_name,
        label_status_to_keep = label_status_to_keep,
        multiple_splits      = multiple_splits,
        splits_fpath         = splits_fpath
    )

if __name__ == "__main__":
    test()
