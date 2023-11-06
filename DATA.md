# Data Preparation
Our model was trained using three datasets: Objects365v1, V3Det, and OpenImages. We conducted tests on the LVIS dataset in a zero-shot manner. Please organize the datasets as follows.
## Pretrained Weights
SAM Pretrain Weights (ViT-base)
```bash
mkdir -p sam_checkpoints
cd sam_checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
cd ..
```
## Data
### Training
1. Datasets preparation
    Download the datasets from their respective official websites. Ensure that you have [COCO](https://cocodataset.org/#home), [objects365](https://www.objects365.org/overview.html), [V3Det](https://v3det.openxlab.org.cn/) and [OpenImages V6](https://storage.googleapis.com/openimages/web/download_v6.html). Organize the downloaded datasets as follows:
```
${ROOT}
    -- datasets
        --coco
        --objects365
        --v3det
        --openimages
```

2. Mask Token Preparation
As the SAM (Segment Anything Model) has been set to a frozen state, we've optimized our resource usage by pre-extracting the image mask tokens. This step significantly reduces memory consumption during model training and inference. We have made these pre-extracted mask tokens available for easy access:
[Download Masks Tokens from One Drive](https://1drv.ms/f/s!AgWqwlwga-5Ka9-HT1L83INBHsU?e=wTbJz5)
We anticipate the data to be organized as follows:

``` bash
${ROOT}
    -- datasets
        -- datasets_mask_tokens_vit_b
            --objects365
            --v3det
            --openimages
            
```
### Evaluation
For model evaluation, download the LVIS dataset from [LVIS Dataset](https://www.lvisdataset.org/) and place it in the `datasets` folder at the project root:
```
${ROOT}
    -- datasets
        --lvis
```
After downloading the LVIS dataset, also obtain the bounding box results from GLIP by downloading the provided JSON file:

- Download the file from [GLIP Box Results]( https://1drv.ms/u/s!AgWqwlwga-5KdWacuP6dTKajYRg?e=PIBdYd).

Once downloaded, place the JSON file in the `glip_results` directory within `datasets`:
```
${ROOT}
    -- datasets
        --glip_results
            nms_results_glip_tiny_model_o365_goldg_cc_sbu_lvis_val.json
```
