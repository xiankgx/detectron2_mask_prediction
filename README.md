# detectron2 Mask Prediction

This repo contains code for predicting masks using a pretrained Mask-RCNN model using the detectron2 library.

## Usage

```bash
python main.py \
    --input_dir <IMAGE_INPUT_DIR> \
    --output_dir <IMAGE_OUTPUT_DIR>
```

For each image, a mask and trimap is predicted. The trimap is predicted from erotion-dilation of the mask.