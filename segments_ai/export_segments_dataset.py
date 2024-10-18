"""
    This script downloads a dataset from Segments.ai and saves it in the 'EXPORT_FOLDER' directory.
    Different exporting formats are supported by Segments.ai APIs.
"""

## Handle Ctrl-C to exit program
import signal as sg
sg.signal(sg.SIGINT, sg.SIG_DFL)

import os

from segments import SegmentsClient, SegmentsDataset
from segments.utils import export_dataset
from segments.utils import get_semantic_bitmap

import matplotlib.pyplot as plt

# Segments.ai API Key
API_KEY           = "e6eb70a8f4cd51d900b5ca6a0fcbb504070b5307"
# Segments.ai Dataset ID (username/dataset name)
DATASET_ID        = "andrea_eusebi/pallet_semantic_segmentation_loco"
# Segments.ai release name of the given dataset
RELEASE_NAME      = "v0.2"
# The directory where the data will be downloaded to for caching. Set to 'None` to disable caching.
SEGMENT_CACHE_DIR = "./"
# Export folder
EXPORT_FOLDER     = "/home/andrea/raymond/pallet_detection_datasets/pallet_semantic_segmentation_loco_v0.2"
# Export format (consult Segments.ai documentation for supported values)
EXPORT_FORMAT     = "semantic"
# Set to True if you want to visualize each rgb image alongside with corresponding ground truth map
VISUALIZE_DS      = False

def main():
    ## Initialize a SegmentsDataset from the release file
    segments_client  = SegmentsClient(API_KEY)
    segments_release = segments_client.get_release(DATASET_ID, RELEASE_NAME)
    segments_dataset = SegmentsDataset(release_file = segments_release,
                                       labelset     = 'ground-truth',
                                       filter_by    = ['labeled', 'reviewed'],
                                       segments_dir = SEGMENT_CACHE_DIR)

    ## Export to given format
    export_dataset(dataset       = segments_dataset,
                   export_folder = EXPORT_FOLDER,
                   export_format = EXPORT_FORMAT)

    ## Visualize the dataset (each sample!)
    if VISUALIZE_DS:
        for sample in segments_dataset:
            # Print the sample name and list of labeled objects
            print(sample['name'])
            print(sample['annotations'])
            
            fig = plt.figure()
            
            # Show the image
            fig.add_subplot(1, 2, 1)
            plt.imshow(sample['image'])
            plt.title("Original Image")
            
            if EXPORT_FORMAT == "instance":
                # Show the instance segmentation label
                fig.add_subplot(1, 2, 2)
                plt.imshow(sample['segmentation_bitmap'])
                plt.title("Instance Segmentation bitmap")
            elif EXPORT_FORMAT == "semantic":
                # Show the semantic segmentation label
                fig.add_subplot(1, 2, 2)
                semantic_bitmap = get_semantic_bitmap(sample['segmentation_bitmap'], sample['annotations'])
                plt.imshow(semantic_bitmap)
                plt.title("Semantic Segmentation bitmap")
            else:
                print("Error! Can't display annotation for given EXPORT_FORMAT!")
                break

            plt.show()

if __name__ == "__main__":
    main()
