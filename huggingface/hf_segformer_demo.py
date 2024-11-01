## Standard modules
import torch
import logging
import matplotlib.pyplot as plt
import numpy             as np
from   pathlib           import Path
from   PIL               import Image
from   typing            import Tuple  

## Custom modules
import py_utils.log

## Hugging Face modules
# Importing "SegformerImageProcessor" first is crucial to avoid segmentation faults.
# WARNING: Import "SegformerImageProcessor" BEFORE "SegformerForSemanticSegmentation" 
#          to prevent segmentation faults (root cause unknown).
from transformers import SegformerImageProcessor
from transformers import SegformerForSemanticSegmentation

def runSegformerInference(img_fpath      : str,
                          model_weights  : str,
                          segformer_proc : SegformerImageProcessor) -> np.ndarray:
    """
    Runs inference on an image using a Segformer model for semantic segmentation.

    This function loads an image, preprocesses it, runs it through a Segformer model for 
    semantic segmentation, and returns a segmentation map.

    Args:
        img_fpath (str): File path to the input image to be segmented. The image should 
            be in RGB format.
        model_weights (str): Path or model identifier to the pretrained weights for the 
            Segformer model.
        segformer_proc (SegformerImageProcessor): Processor object for preprocessing and 
            postprocessing the image data.

    Returns:
        np.ndarray: A 2D numpy array representing the segmentation map of the input image, 
        where each pixel value corresponds to a semantic class label.

    Raises:
        FileNotFoundError: If the specified image file does not exist at `img_fpath`.

    Notes:
        The function automatically detects if a CUDA-capable GPU is available and runs the 
        model on it. If no GPU is available, it defaults to using the CPU.

        The function expects the input image file path to be valid and accessible.

        The returned segmentation map is processed to match the original image size (H x W).
    """
    
    if not Path(img_fpath).exists():
        raise FileNotFoundError("Given image path doesn't exist.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ## Instantiate the model
    model = SegformerForSemanticSegmentation.from_pretrained(
        pretrained_model_name_or_path = model_weights
    ).to(device)

    ## Load and Preprocess Input Image
    image        = Image.open(img_fpath).convert("RGB")
    pixel_values = segformer_proc(image, return_tensors="pt").pixel_values.to(device)

    ## Run Inference
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)

    ## Post process the output of Segformer model
    semantic_segmentation_map = segformer_proc.post_process_semantic_segmentation(
        outputs,
        target_sizes=[image.size[::-1]] # PIL Image size is W x H, but we want H x W
    )[0]

    # Send to CPU and convert to Numpy array
    semantic_segmentation_map = semantic_segmentation_map.cpu().numpy()

    return semantic_segmentation_map

def displayInferenceResults(img_fpath        : str,
                            segmentation_map : np.ndarray,
                            palette          : Tuple[Tuple[int, int, int]]) -> None:
    """
    Displays a semantic segmentation map overlaid on the given image.

    This function reads an image from the specified file path, applies the segmentation 
    map using a given color palette, and displays the result with the segmentation overlay.

    Args:
        img_fpath (str): File path to the input image to be displayed with the segmentation overlay.
        segmentation_map (np.ndarray): 2D array representing the segmentation map, where each pixel 
            value corresponds to a semantic class label.
        palette (Tuple[Tuple[int, int, int]]): A tuple of RGB color tuples defining the color 
            for each class label in the segmentation map.

    Raises:
        FileNotFoundError: If the specified image file does not exist at `img_fpath`.

    Notes:
        The function combines the original image and the segmentation map with a 50% opacity 
        blend, where each class label in the segmentation map is colored according to the 
        specified `palette`.
    """

    if not Path(img_fpath).exists():
        raise FileNotFoundError("Given image path doesn't exist.")
    
    # Convert palette to array
    palette_arr = np.array(palette)

    ## Color segmentation mask with palette
    segm_coloured = np.zeros((segmentation_map.shape[0], segmentation_map.shape[1], 3),
                             dtype=np.uint8)
    
    for label, color in enumerate(palette_arr):
        segm_coloured[segmentation_map == label, :] = color

    # Convert to BGR
    segm_coloured = segm_coloured[..., ::-1]

    ## Show image + mask
    image = Image.open(img_fpath).convert("RGB")

    mixed_image = np.array(image)*0.5 + segm_coloured*0.5
    mixed_image = mixed_image.astype(np.uint8)

    ## Plotting
    plt.figure(figsize=(15, 10))
    plt.imshow(mixed_image)
    plt.show()

##### ----- Test Script ----- #####

def test():
    PALETTE       = (
                      (  0,   0,   0),
                      (  0, 255,   0)
                    )
    SEG_PROCESSOR = SegformerImageProcessor(do_resize=False,
                                            do_rescale=True,
                                            do_normalize=True,
                                            do_reduce_labels=True)
    
    IMG_FPATH     = ("/home/andrea/raymond/pallet_detection_datasets/pallet_detection_dataset_v0.1/"
                     "rgb/val/rgb_0080.png")
    MODEL_WEIGHTS = "/home/andrea/raymond/AI_tools/huggingface/241031_TEST_2/"

    logger = py_utils.log.getCustomLogger(logger_name_=__name__ + "_test",
                                          node_name_="hf_segformer_demo",
                                          log_handler_=logging.StreamHandler(),
                                          logging_level_=logging.INFO)
    
    logger.info("Starting test script...")
    
    segmentation_map = runSegformerInference(img_fpath=IMG_FPATH,
                                             model_weights=MODEL_WEIGHTS,
                                             segformer_proc=SEG_PROCESSOR)
    
    logger.info("Segformer inference run!")
    
    displayInferenceResults(img_fpath=IMG_FPATH,
                            segmentation_map=segmentation_map,
                            palette=PALETTE)
    
    logger.info("Inference results displayed!")
    
    logger.info("Test script completed!")

if __name__ == "__main__":
    test()
