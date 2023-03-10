# stable-diffusion-experiments

This repo contains output from the experiments conducted on stable diffusion model v1.4 and v1.5. These are all early experiments conducted in a limited cimpute environment. 

## Outputs

* [Single face Image outputs](outputs/single_face_finetuning)

In this experiment stable-diffusion checkpoint v1.4 was finetuned using a single image of a friend. The goal was to create images with a vectorish style look and a blueish tint. To finetune please clone the following repo and follow the instructions in the readme. The webui extension will not work but that can be fixed by editing the following lines:

https://github.com/7eu7d7/DreamArtist-stable-diffusion
```
583 num_samples = ds.__len__()
584 epoch_num = embedding.step // num_samples
585 epoch_step = embedding.step % num_samples

and 

780 num_samples = ds.__len__()
```


* [multi_image_LoRA](outputs/multi_image_LoRA)

In this experiment stable diffusion checkpoint v1.5 was used to fine tune to create spritesheets. This is close to pixel art and the diffusion models suffer a lot as the data to fine tune is usually very low in resolution. For example a typical pixel art sprite can be 8-bit or 16-bit. Another test here was to generate evolutions of pokemons. No more than 10 spritesheets were used to finetune the model. To finetune the model using Low Rank Approximation (LoRA) there are two options:

1. We can either use diffusers with a higher learning rate:
https://github.com/huggingface/diffusers
```
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="path-to-instance-images"
export OUTPUT_DIR="path-to-save-model"

accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of sks dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --checkpointing_steps=100 \
  --learning_rate=1e-4 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --validation_prompt="A photo of sks dog in a bucket" \
  --validation_epochs=50 \
  --seed="0" \
  --push_to_hub

```

2. For better control and customization we can use the following:

```
https://github.com/cccntu/minLoRA
```
This is a minimal PyTorch re-implementation of LoRA.

* [Single Image Textual Inversion](outputs/single_image_textual_inversions)

This was the oldest experiment conducted on stable diffusion v1.4. The goal was to invert a single image to create a text prompt that would generate the same image of dragon variations. This was also done on sprite sheets so the same challenges of low resolution data apply. Again we can use diffusers here. 

* [Dreambooth experiments](outputs/dreambooth)

In this stable diffusion checkpoint v1.5 was used to finetune on pixel art spritesheets and sketch style arts. The goal here was to just study the effects of various fine tuning parameters. The results here are from early experiments only.

* [Random Experiment](outputs/random)

These are all the experiments conducted randomly on stable diffusion v1.4 and v1.5. The goal was to just study the effects of various fine tuning parameters. And, analyze the limitations of the models in terms of art styles and consistency.

## Workflow

The normal worklow would be to finetune the model depending on the data and the goal. The model can be finetuned using the following:

1. [LoRA Textural](https://huggingface.co/docs/diffusers/training/lora) Inversion (maximum compatibiltiy with open source checkpoints)
2. [LoRA Dreambooth](https://huggingface.co/docs/diffusers/training/lora) (Low compatibiltiy with open source checkpoints)
3. [Dreamartist](https://github.com/7eu7d7/DreamArtist-sd-webui-extension) (Low compatibiltiy but better with faces compared to dreambooth)
4. [ControlNet](https://github.com/lllyasviel/ControlNet) (This should be compatible with almost all checkpoints) -- (This attaches zero convolutions to the middle and decoder blocks of the UNEt and should allow finer control over the output.)
5. [Tencent T2I-Adapter](https://github.com/TencentARC/T2I-Adapter/tree/main) (This should be compatible with almost all checkpoints and can be used with other models after finetuning only once)

Then the model would be copied to a web-ui for quick interactive testing if required. The following repo is recommended for this:

https://github.com/AbdBarho/stable-diffusion-webui-docker

This packages everything in a docker container and allows to quickly setup multiple UIs for testing.

For production the model should be converted to TensorRT to reduce the inference time. For example on a T4 using the DPM++ 2M sampler for 30 iterations the inference time can be reduced to nearly half as compared to xformers and AI Template.


## Note

None of the images in the outputs were post processed with for example upscalers and face resotration models. The images are as they are generated by the model.
