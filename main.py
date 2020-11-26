import argparse
import glob
import json
import os
import random
import shutil

import cv2
import detectron2
import numpy as np
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.logger import setup_logger
from PIL import Image
from tqdm import tqdm

setup_logger()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",
                        type=str,
                        help="Image input directory.")
    parser.add_argument("--output_dir",
                        type=str,
                        default="predictions/",
                        help="Image output directory.")
    parser.add_argument("--mask_area_threshold",
                        type=float,
                        default=0.1)
    args = parser.parse_args()

    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)

    img_exts = [".jpg", ".jpeg", ".jfif", ".png"]
    input_images = glob.glob(args.input_dir + "/**/*", recursive=True)
    input_images = list(filter(os.path.isfile, input_images))
    input_images = list(filter(lambda p: os.path.splitext(p)
                               [-1] in img_exts, input_images))
    print(f"num input images: {len(input_images)}")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "image"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "mask"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "trimap"), exist_ok=True)

    cat_id_person = 0
    interested_categories = [
        cat_id_person
    ]

    for p in tqdm(input_images, "Inferencing"):
        im = cv2.imread(p)

        outputs = predictor(im)
        # print(outputs.keys())

        masks = outputs["instances"].pred_masks.cpu().numpy()
        classes = outputs["instances"].pred_classes.cpu().numpy()
        # print(f"masks shape: {masks.shape}")
        # print(f"classes shape: {classes.shape}")

        if len(masks) > 0:
            merged_mask = np.zeros_like(masks[0], dtype=bool)
            for i, (obj_class, mask) in enumerate(zip(classes, masks)):
                # print(f"detection #{i}: {obj_class}")

                if obj_class in interested_categories:
                    mask_area = mask.sum() / mask.size
                    if mask_area >= args.mask_area_threshold:
                        # points = np.argwhere(mask == True)
                        # # print("points.shape:", points.shape)

                        # poly = (mask.copy().astype(np.float32) * 255).astype(np.uint8)
                        # cv2.fillConvexPoly(poly, points[:, ::-1], 255)

                        merged_mask = merged_mask | mask
                        # merged_mask = merged_mask | poly.astype(bool)

            merged_mask = (merged_mask.astype(np.float32) * 255) \
                .astype(np.uint8)

            # kernel = np.ones((5, 5), np.uint8)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

            # Erode mask
            eroded_mask = cv2.erode(merged_mask, kernel, iterations=8)

            # Dilate mask
            dilated_mask = cv2.dilate(merged_mask, kernel, iterations=16)

            trimap = np.clip(dilated_mask.copy(), 0, 127)
            trimap = np.where(eroded_mask == 255, 255, trimap)
            if merged_mask.sum() == 0:
                # merged_mask[:, :] = 255
                trimap = np.full_like(merged_mask, 0, dtype=np.uint8)

            cv2.imwrite(os.path.join(args.output_dir, "mask", os.path.splitext(os.path.basename(p))[0] + ".png"), merged_mask)
            cv2.imwrite(os.path.join(args.output_dir, "trimap", os.path.splitext(os.path.basename(p))[0] + ".png"), trimap)
            shutil.copyfile(p, os.path.join(args.output_dir, "image", os.path.basename(p)))

            # cv2.imwrite("mask.png",
            #             merged_mask)
            # cv2.imwrite("dilated_mask.png",
            #             dilated_mask)
            # cv2.imwrite("masked_image.png",
            #             np.concatenate([im, merged_mask[..., np.newaxis]], -1))
            # cv2.imwrite("masked_image_dilated.png",
            #             np.concatenate([im, dilated_mask[..., np.newaxis]], -1))
            # cv2.imwrite("masked_image_eroded.png",
            #             np.concatenate([im, eroded_mask[..., np.newaxis]], -1))
            # exit(0)

            # # Green screen background
            # bg = np.zeros_like(im)
            # bg[..., 1] = 177
            # bg[..., 2] = 64

            # # Compose foreground on green screen background
            # composed = Image.composite(Image.fromarray(im[..., ::-1]),
            #                            Image.fromarray(bg),
            #                            Image.fromarray(dilated_mask))
            # # composed.save("img_bgremoved.jpg")
            # composed.save(os.path.join(args.output_dir, os.path.basename(p)))

    print("Done!")
