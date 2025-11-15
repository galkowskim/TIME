### Train context tokens

```bash
python training.py \
  --dataset ImageNet --dataset_source hf --partition train \
  --sd_model CompVis/stable-diffusion-v1-4 --image_size 512 \
  --output_path context.pth \
  --custom_tokens '|<C*1>|' '|<C*2>|' '|<C*3>|' \
  --custom_tokens_init centered realistic photo \
  --phase context --mini_batch_size 1 \
  --enable_xformers_memory_efficient_attention
```

### Train class tokens


```bash
# zebra
python training.py \
  --dataset ImageNet --dataset_source hf --partition train \
  --label_query 340 \
  --sd_model CompVis/stable-diffusion-v1-4 --image_size 512 \
  --embedding-files context.pth \
  --output_path class-340.pth \
  --custom_tokens '|<AC*340>|' \
  --custom_tokens_init zebra \
  --phase class \
  --base_prompt 'A |<C*1>| |<C*2>| |<C*3>| photo' \
  --mini_batch_size 1 --enable_xformers_memory_efficient_attention

# sorrel (or 'horse' if sorrel isn’t available)
python training.py \
  --dataset ImageNet --dataset_source hf --partition train \
  --label_query 339 \
  --sd_model CompVis/stable-diffusion-v1-4 --image_size 512 \
  --embedding-files context.pth \
  --output_path class-339.pth \
  --custom_tokens '|<AC*339>|' \
  --custom_tokens_init sorrel \
  --phase class \
  --base_prompt 'A |<C*1>| |<C*2>| |<C*3>| photo' \
  --mini_batch_size 1 --enable_xformers_memory_efficient_attention
```

### Generate CEs (flip to sorrel):

```bash
python generate-ce.py \
  --dataset ImageNet --dataset_source hf --partition val \
  --sd_model CompVis/stable-diffusion-v1-4 \
  --sd_image_size 512 --classifier_image_size 224 \
  --embedding_files context.pth class-340.pth class-339.pth \
  --generic_custom_token '|<AC*&>|' \
  --base_prompt 'A |<C*1>| |<C*2>| |<C*3>| photo' \
  --output_path /path/to/results --exp_name zebra_to_sorrel \
  --label_query 0 --label_target 339 \
  --guidance-scale-invertion 5 --guidance-scale-denoising 5 \
  --num_inference_steps 35 --enable_xformers_memory_efficient_attention
```