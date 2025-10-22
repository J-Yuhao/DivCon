# DivCon: Divide and Conquer for Complex Numerical and Spatial Reasoning in Text-to-Image Generation [ECAI2025]

This is the implementation of Text-to-Image framework [DivCon (arXiv)](https://arxiv.org/abs/2403.06400).


<!-- [[Website]( )][[Demo]( )] -->

<!-- [[Paper]( )] -->

![Teaser figure](./figs/fig1.jpg)

## Updates

- **[2025-07-11]** Initial release of the official codebase.
- **[2025-07-11]** DivCon is accepted by ECAI-2025!

## Dependencies
In your environment where Python version is 3.8 or higher, or alternatively, create a new environment:

```bash
conda create --name divcon python==3.8.0
conda activate divcon
```
and install related libraries

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
pip install git+https://github.com/CompVis/taming-transformers.git
pip install git+https://github.com/openai/CLIP.git
```
## Inference: Predict layouts with DivCon 
We provide scripts to generate layouts for HRS and NSR-1K benchmark. First navigate to ```./LLM_gen_layout```, you might need to set up your hugging face login token at line 4 in ```num_spa_iner.py``` depends on the LLM you want to use:
```bash
cd LLM_gen_layout
```
Then run:
```bash
# for numerical prompts
python layout_gen.py --dataset HRS --task numerical

```
The generated layouts will be saved to ```./LLM_gen_layout``` by default.

We also provide generated layout for both benchmarks under the ```./LLM_gen_layout```.

## Inference: Generate images with DivCon

Download the layout conditioned model [GLIGEN](https://huggingface.co/gligen/gligen-generation-text-box/blob/main/diffusion_pytorch_model.bin) and put them in `gligen_checkpoints`

To generate images from numerical prompts in HRS using layouts predicted by divcon, run:
```bash
python divcon_gen.py --ckpt gligen_checkpoints/diffusion_pytorch_model.bin --file_save HRS 
                     --type counting --pred_layout ./LLM_gen_layout/HRS_counting.p
```
Where
- `--ckpt`: Path to the GLIGEN checkpoint
- `--file_save`: Path to save the generated images
- `--type`: The category to test, counting or spatial
- `--pred_layout`: Path to the predicted layout from LLM
You can modify these input parameters to generate images for different benchmarks or categories.

## Layout & Image Evaluation
To evaluate the raw layouts, navigate to ```LLM_gen_layout``` and run:
```bash
cd LLM_gen_layout
# for numerical prompts in HRS benchmark
python eval_counting_layout.py --pred_layout HRS_counting.p
# or for spatial prompts in HRS benchmark
python eval_spatial_layout.py --pred_layout HRS_spatial.p
```
To evaluate the generated images using YOLOv8, navigate to ```evaluation``` and first run:
```bash
cd evaluation
python YOLOv8.py --in_folder ../visual/HRS_img --out_file HRS_detect.p
```
then run evaluation scripts:
```bash
# for numerical prompts in HRS benchmark
python eval_counting.py --in_result detection_result/HRS_detect.p
# or for spatial prompts in HRS benchmark
python eval_spatial.py --in_result detection_result/HRS_detect.p
```

## 📖 Please cite us

If you find this work useful, please cite:

```bibtex
@misc{jia2025divcondivideconquercomplex,
      title={DivCon: Divide and Conquer for Complex Numerical and Spatial Reasoning in Text-to-Image Generation}, 
      author={Yuhao Jia and Wenhan Tan},
      year={2025},
      eprint={2403.06400},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2403.06400}, 
}
```

## Acknowledgements

This project is built upon the foundational work from [**GLIGEN**](https://github.com/gligen/GLIGEN) and [**Attention-Refocusing**](https://github.com/Attention-Refocusing/attention-refocusing).

 



