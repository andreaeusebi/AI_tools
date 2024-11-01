## Standard modules
import logging
import numpy  as np
import json
import torch
from   torch  import nn
import cv2
from   typing import Dict, Any

## Custom modules
import py_utils.log

## Huggingface modules
from huggingface_hub import login, hf_hub_download
from datasets        import load_dataset, load_dataset_builder
from transformers    import (SegformerImageProcessor,
                             SegformerForSemanticSegmentation,
                             TrainingArguments,
                             Trainer,
                             BatchFeature)
import evaluate
from   transformers.trainer import EvalPrediction

## Albumentatios modules
import albumentations as Albu

# WARNING: To be deleted after being sure that class method versions
#          are safe.
def trainTransformExt(batch_ : dict):

    assert(len(batch_["pixel_values"]) == len(batch_["label"]))

    augm_pipeline_                = Albu.Compose([ Albu.Resize(128,
                                                  256,
                                                  interpolation=cv2.INTER_AREA,
                                                  p=1.0) ])
    
    segformer_processor    = SegformerImageProcessor(do_resize=False,
                                               do_rescale=True,
                                               do_normalize=True,
                                               do_reduce_labels=True)

    images = []
    labels = []

    ## Parse and augments both images and labels
    for (img_pil, label_pil) in zip(batch_["pixel_values"], batch_["label"]):
        augmented = augm_pipeline_(image=np.array(img_pil.convert("RGB")),
                                mask=np.array(label_pil))
        images.append(augmented["image"])
        labels.append(augmented["mask"])

    assert(len(images) == len(labels))

    # Complete preprocessing to provide data as expected by Segformer
    segformer_inputs = segformer_processor(images, labels)

    return segformer_inputs

# WARNING: To be deleted after being sure that class method versions
#          are safe.
def validTransformExt(batch_ : dict):
    assert(len(batch_["pixel_values"]) == len(batch_["label"]))

    augm_pipeline_                = Albu.Compose([ Albu.Resize(128,
                                                  256,
                                                  interpolation=cv2.INTER_AREA,
                                                  p=1.0) ])
    
    segformer_processor    = SegformerImageProcessor(do_resize=False,
                                               do_rescale=True,
                                               do_normalize=True,
                                               do_reduce_labels=True)

    images = []
    labels = []

    ## Parse and augments both images and labels
    for (img_pil, label_pil) in zip(batch_["pixel_values"], batch_["label"]):
        augmented = augm_pipeline_(image=np.array(img_pil.convert("RGB")),
                                mask=np.array(label_pil))
        images.append(augmented["image"])
        labels.append(augmented["mask"])

    assert(len(images) == len(labels))

    # Complete preprocessing to provide data as expected by Segformer
    segformer_inputs = segformer_processor(images, labels)

    return segformer_inputs

class HfSegformerTrainer:
    """
    Utility class providing a training interface for a Hugging Face-based Segformer model.

    This class sets up and manages the training pipeline for a semantic segmentation model
    using the Segformer architecture, including model instantiation, data augmentation, 
    metric computation, and result uploads to the Hugging Face Hub.
    """

    def __init__(self,
                 hf_token_         : str,
                 dataset_name_     : str,
                 pretrained_model_ : str,
                 train_augm_       : Albu.Compose,
                 valid_augm_       : Albu.Compose,
                 segformer_proc_   : SegformerImageProcessor,
                 training_args_    : TrainingArguments,
                 log_level_        : int = logging.INFO) -> None:
        """
        Initializes a Segformer model and related configurations for training.

        Args:
            hf_token_ (str): Hugging Face authentication token.
            dataset_name_ (str): Name of the dataset to use for training.
            pretrained_model_ (str): Name or path of the pretrained Segformer model.
            train_augm_ (Albu.Compose): Augmentation pipeline for training dataset.
            valid_augm_ (Albu.Compose): Augmentation pipeline for validation dataset.
            segformer_proc_ (SegformerImageProcessor): Processor for input data transformations.
            training_args_ (TrainingArguments): Training arguments and configuration.
            log_level_ (int, optional): Logging level, default is `logging.INFO`.

        Raises:
            FileNotFoundError: If the Hugging Face authentication token is invalid.
        """

        ## Logger setup
        self.logger = py_utils.log.getCustomLogger(logger_name_=__name__,
                                                   node_name_="HfTrainer",
                                                   log_handler_=logging.StreamHandler(),
                                                   logging_level_=log_level_)

        self.logger.debug("__init__() begin!")

        # Login to Huggingface
        login(hf_token_)

        ## Save data
        self.dataset_name     = dataset_name_
        self.pretrained_model = pretrained_model_
        self.out_model_name   = training_args_.hub_model_id

        ## Retrieve dataset informations (optional)
        ds_builder = load_dataset_builder(self.dataset_name)

        self.logger.debug(f"ds_builder description: {ds_builder.info.description}")
        self.logger.debug(f"ds_builder features: {ds_builder.info.features}")

        ## Load dataset splits
        self.train_ds = load_dataset(self.dataset_name, split="train")
        self.valid_ds = load_dataset(self.dataset_name, split="valid")

        self.logger.info(f"Train dataset: {self.train_ds}")
        self.logger.info(f"Valid dataset: {self.valid_ds}")

        self.logger.debug(f"Image: {self.train_ds[0]}")

        # Segformer Image processor
        self.segformer_processor = segformer_proc_
        
        ## Set transformation pipelines for each dataset split
        self.train_ds.set_transform(self.trainTransform)
        self.valid_ds.set_transform(self.validTransform)

        self.logger.debug(f"Train dataset format: {self.train_ds.format}")

        ## Augmentation pipelines (based on Albumentations)
        self.train_augm = train_augm_
        self.valid_augm = valid_augm_

        ## Retrieve id and labels of the dataset (assuming there is a id2label.json file)
        self.id2label = json.load(open(hf_hub_download(repo_id=self.dataset_name,
                                                       filename="id2label.json",
                                                       repo_type="dataset"), "r"))
        self.id2label = {int(k): v for k, v in self.id2label.items()}
        label2id = {v: k for k, v in self.id2label.items()}

        self.logger.info(f"self.id2label: {self.id2label}")

        # Index to be ignored by evaluation metrics (not suggested to change)
        self.IGNORE_IDX = 255

        ## Segformer model instantiation
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            pretrained_model_name_or_path=self.pretrained_model,
            id2label=self.id2label,
            label2id=label2id
        )

        ## Initialize metrics (mIoU and Confusion Matrix)
        self.mean_iou_metric = evaluate.load("mean_iou")

        ## Initialize the Trainer
        self.trainer = Trainer(model=self.model,
                               args=training_args_,
                               train_dataset=self.train_ds,
                               eval_dataset=self.valid_ds,
                               compute_metrics=self.computeMetrics)

        self.logger.debug("__init__() completed!")

    def train(self) -> None:
        """
        Runs the training pipeline for the Segformer model.
        """

        self.trainer.train()

    def uploadResultstoHfHub(self) -> None:
        """
        Uploads the training results and model to the Hugging Face Hub.

        Notes:
            The Segformer processor and trainer will both push their outputs to the Hub.
        """

        self.logger.debug("Publishing results to Huggingface Hub...")

        kwargs = {
            "tags": ["vision", "image-segmentation"],
            "finetuned_from": self.pretrained_model,
            "dataset": self.dataset_name,
        }

        self.segformer_processor.push_to_hub(self.out_model_name, private=True)
        self.trainer.push_to_hub(**kwargs)

        self.logger.info("Training results saved to Huggingface Hub!")

    def trainTransform(self,
                       batch_ : dict) -> BatchFeature:
        """
        Prepares a batch of training data for Segformer.

        Args:
            batch_ (dict): A dictionary containing a batch of training data.

        Returns:
            BatchFeature: The processed batch of features.
        """

        return self.batchTransform(batch_=batch_,
                                   augm_pipeline_=self.train_augm)
        
    def validTransform(self,
                       batch_ : dict) -> BatchFeature:
        """
        Prepares a batch of validation data for Segformer.

        Args:
            batch_ (dict): A dictionary containing a batch of validation data.

        Returns:
            BatchFeature: The processed batch of features.
        """

        return self.batchTransform(batch_=batch_,
                                   augm_pipeline_=self.valid_augm)
    
    def batchTransform(self,
                       batch_ : dict,
                       augm_pipeline_ : Albu.Compose) -> BatchFeature:
        """
        Applies augmentations and transformations to a given batch.

        Args:
            batch_ (dict): A batch containing pixel values and labels.
            augm_pipeline_ (Albu.Compose): Augmentation pipeline to apply.

        Returns:
            BatchFeature: A dictionary containing the transformed images and labels.
        """

        assert(len(batch_["pixel_values"]) == len(batch_["label"]))

        images = []
        labels = []

        ## Parse and augments both images and labels
        for (img_pil, label_pil) in zip(batch_["pixel_values"], batch_["label"]):
            augmented = augm_pipeline_(image=np.array(img_pil.convert("RGB")),
                                       mask=np.array(label_pil))
            images.append(augmented["image"])
            labels.append(augmented["mask"])

        assert(len(images) == len(labels))

        # Complete preprocessing to provide data as expected by Segformer
        segformer_inputs = self.segformer_processor(images, labels)

        return segformer_inputs
    
    def computeMetrics(self, eval_pred_ : EvalPrediction) -> Dict[str, Any]:
        """
        Computes evaluation metrics for the predictions.

        Args:
            eval_pred_ (EvalPrediction): Object containing model predictions and labels.

        Returns:
            Dict[str, Any]: A dictionary with calculated metrics, including per-category accuracy
            and IoU values.
        """
        
        with torch.no_grad():
            logits, labels = eval_pred_
            logits_tensor = torch.from_numpy(logits)
            # Upscale the logits to the size of the label
            logits_tensor = nn.functional.interpolate(input=logits_tensor,
                                                      size=labels.shape[-2:],
                                                      mode="bilinear",
                                                      align_corners=False).argmax(dim=1)
            pred_labels = logits_tensor.detach().cpu().numpy()

            mean_iou_results = self.mean_iou_metric._compute(
                predictions=pred_labels,
                references=labels,
                num_labels=len(self.id2label),
                ignore_index=self.IGNORE_IDX,
                reduce_labels=False # we've already reduced the labels ourselves
            )
            
        # add per category metrics as individual key-value pairs
        per_category_accuracy = mean_iou_results.pop("per_category_accuracy").tolist()
        per_category_iou = mean_iou_results.pop("per_category_iou").tolist()

        self.logger.debug(f"per_category_accuracy: {per_category_accuracy}")

        mean_iou_results.update(
            {f"accuracy_{self.id2label[i]}": v for i, v in enumerate(per_category_accuracy)}
        )
        mean_iou_results.update(
            {f"iou_{self.id2label[i]}": v for i, v in enumerate(per_category_iou)}
        )

        return mean_iou_results
    
    def __deepcopy__(self, memodict={}) -> "HfSegformerTrainer":
        """
        Overwite of deepcopy() for HfSegformerTrainer object.

        This method doesn't perform a real deep copy but a simple shallow copy for now.
        WARNING: If the original object is modified, those changes would also affect the
                 "deep copied" version, which defeats the purpose of __deepcopy__.

        Args:
            memodict (dict, optional): Dictionary for tracking copied objects to handle 
            recursive copies, default is an empty dictionary.

        Returns:
            HfSegformerTrainer: A new instance of `HfSegformerTrainer`.
        """

        cpyobj = type(self) 
        cpyobj = self
        return cpyobj

##### ----- Test Script ----- #####

def test():
    HF_TOKEN         = ""
    HF_DATASET       = "eusandre95/test_segmentation"
    PRETRAINED_MODEL = "nvidia/mit-b0"
    OUT_MODEL_NAME   = "241101_TEST"
    AUGM             = Albu.Compose([ Albu.Resize(128,
                                                  256,
                                                  interpolation=cv2.INTER_AREA,
                                                  p=1.0) ])
    SEG_PROCESSOR    = SegformerImageProcessor(do_resize=False,
                                               do_rescale=True,
                                               do_normalize=True,
                                               do_reduce_labels=True)
    TRAINING_ARGS    = TrainingArguments(
        output_dir=OUT_MODEL_NAME,
        learning_rate=0.00006,
        num_train_epochs=10,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        save_total_limit=3,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        # save_steps=20,
        # eval_steps=20,
        logging_steps=1,
        eval_accumulation_steps=5,
        load_best_model_at_end=True,
        push_to_hub=True,
        hub_model_id=OUT_MODEL_NAME,
        hub_strategy="every_save",
        hub_private_repo=True
    )

    logger = py_utils.log.getCustomLogger(logger_name_=__name__ + "_test",
                                          node_name_="hf_trainer_test",
                                          log_handler_=logging.StreamHandler(),
                                          logging_level_=logging.INFO)
    
    logger.info("Starting test script...")

    hf_trainer = HfSegformerTrainer(hf_token_=HF_TOKEN,
                                    dataset_name_=HF_DATASET,
                                    pretrained_model_=PRETRAINED_MODEL,
                                    train_augm_=AUGM,
                                    valid_augm_=AUGM,
                                    segformer_proc_=SEG_PROCESSOR,
                                    training_args_=TRAINING_ARGS,
                                    log_level_=logging.INFO)
    
    logger.info("HfSegformerTrainer correctly initialized! Running training now...")

    hf_trainer.train()

    hf_trainer.uploadResultstoHfHub()

    logger.info("Test script completed!")

if __name__ == "__main__":
    test()
