# LumiNet
[CVPR 2025] LumiNet: Latent Intrinsics Meets Diffusion Models for Indoor Scene Relighting

🚨 **Erratum (Sept. 2025):** The results in Table 1 have been corrected after fixing a data normalization bug, which aligns with the Latent-Intrinsics evaluation code. Please refer to the updated [paper](https://arxiv.org/pdf/2412.00177) if you want to use the numbers for LumiNet and RGB2X that align with the Latent-Intrinsics evaluation protocol. The paper's core contributions, qualitative results, and user study are unaffected. We thank the Sony PlayStation team for reporting this.

🆕 **Update (Sept. 2025):** We incorporate a lightweight bypass decoder, which replaces the original Stable Diffusion decoder to improve identity preservation.  

- This module is orthogonal to the original LumiNet weights.  
- The originally released LumiNet weights (April 2025) remain valid and are not affected by this update.

You can easily experience the improvement in two ways:

1. **Run the updated Hugging Face Demo**  
   👉 [LumiNet Demo](https://huggingface.co/spaces/xyxingx/LumiNet)
   
2. **Update your original code repo**  
   - Copy the folder **`modi_vae`** into your repo.  
   - Replace the following files with the updated ones:
     - `relit_inference.py`
     - `ldm/models/diffusion/ddpm.py`
     - `cldm/LumiNet.py`



## News
[Apr. 2025] Inference code and model are released!

[Mar. 2025] We released the Huggingface [Demo](https://huggingface.co/spaces/xyxingx/LumiNet)!

[Feb. 2025] LumiNet is accepted by CVPR 2025!


## Installation
Follow these steps to install LumiNet and get started:
### Requirements
GPU with VRAM > 11G

open-clip-torch==2.0.1 (very important)

## Install Dependencies
You can install the required dependencies by running the following command:

```bash
pip install -r requirements.txt
```

## Pretrained Models Download

The weights of LumiNet and Latent-Intrinsics are hosted on [Hugging Face](https://huggingface.co/xyxingx/LumiNet). Confirmation of accepting the conditions is required. Once you accept the conditions, you can download the models, or the models will be automatically downloaded and loaded when running the demo or inference.

You may need to log in to your HF account before running the code.


## Demo
Run gardio_demo.py will allow you to quickly try the relighting performance on your own images. Each run will provide relit results with 3 different seeds.

## Local Relit
To relit the image locally, use the following command:
```bash
python relit_inference.py
```
You need to set the PATH to the input image and reference image. You can also name the number of relit examples you want to generate and the DDIM step. 

## Seed Selection (Optional)
We provide the nearest-neighbor search based on the latent extrinsic. Once you get the result images in the folder, you can run this command to help you find the image that has the closest latent extrinsic to the reference image

```bash
python seed_selection.py
```

Note: The seed selection relies on the representation of the latent extrinsic. If you cannot get a result that you expected, you may also need to check the relit images on your own.

## Flow-based Clean Up (Optional)
We provide a rectified-flow-based cleanup process to obtain higher-resolution and less noisy images. Our implementation is based on the diffusers library (version ≥ 0.31.0). Please note that access to the [FLUX](https://huggingface.co/black-forest-labs/FLUX.1-schnell) model may require permission. A minimum of 14 GB of VRAM is recommended for successful execution.


```bash
python flux_cleanup.py
```
Note: You may need to adjust the hyperparameters (mainly timesteps) to achieve better visual results, as the optimal values often vary depending on the input images.

## Adapter Training
We provide a lightweight script to fine-tune only the control adapter for improved lighting detection. Refer to [`docs/adapter_training.md`](docs/adapter_training.md) for dataset manifest format, RGB-folder-only training, required checkpoints, and end-to-end commands.



## Contact and Citation
For any questions related to the code or model, please contact [Xiaoyan Xing](mailto:x.xing@uva.nl). 
If you find our work useful for your research, please consider citing:

**LumiNet**
```tex
@inproceedings{xing2025luminet,
  title={Luminet: Latent intrinsics meets diffusion models for indoor scene relighting},
  author={Xing, Xiaoyan and Groh, Konrad and Karaoglu, Sezer and Gevers, Theo and Bhattad, Anand},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={442--452},
  year={2025}
}
```
**Latent-intrinsics**
```tex
@inproceedings{Zhang2024Latent,
    title={Latent Intrinsics Emerge from Training to Relight},
    author={Zhang, Xiao and Gao, William and Jain, Seemandhar and Maire, Michael and Forsyth, David and Bhattad, Anand},
    booktitle={NeurIPS},
    year={2024}}
```
## License
The model and pretrained weights provided in this repository are licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International ([CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)) license.
