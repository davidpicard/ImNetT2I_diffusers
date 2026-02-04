import torch
import torch.nn.functional as F
import numpy as np
from diffusers import SD3Transformer2DModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
from torch.optim import AdamW
from omegaconf import DictConfig, OmegaConf
import hydra
from tqdm import tqdm
from accelerate import Accelerator
from im_dataset import CaptionImageNetDataset
from sampler import FMEulerSampler
from pathlib import Path
import wandb



@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg):

    print("→ preparing accelerator...", end='', flush=True)
    accelerator = Accelerator(**(cfg.accelerator), project_dir=cfg.checkpoint.save_dir)
    print(" done.✅")
    # small utility print to avoid clutter
    def print_r0(s: str, **kwargs):
        if accelerator.is_main_process:
            print(s, **kwargs)

    print_r0("→ config:")
    print_r0(cfg)

    print_r0("→ loading dataset...", end='', flush=True)
    ds = CaptionImageNetDataset(cfg.data.imagenet_path, im_size=cfg.data.im_size, max_size=cfg.data.max_size)
    train_ds = DataLoader(ds, batch_size=cfg.training.batch_size, shuffle=True, num_workers=cfg.data.num_workers, drop_last=True)
    print_r0(" done.✅")

    print_r0("→ loading model...", end='', flush=True)
    model = SD3Transformer2DModel(sample_size=cfg.data.im_size,out_channels=cfg.model.in_channels, **(cfg.model))
    optimizer = AdamW(model.parameters(), lr=cfg.training.lr)
    print_r0(" done.✅")

    print_r0("→ loading text encoder...", end='', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg.text.model)
    text_encoder = AutoModelForCausalLM.from_pretrained(cfg.text.model)
    text_encoder.eval()
    print_r0(" done.✅")

    if (cfg.checkpoint.load_from is not None) and Path(cfg.checkpoint.load_from).exists():
        accelerator.load_state(cfg.checkpoint.load_from)
        
    print_r0("→ preparing distributed setting...", end='', flush=True)
    device = accelerator.device
    model, optimizer, text_encoder, train_ds = accelerator.prepare(model, optimizer, text_encoder, train_ds)
    print_r0(" done.✅")


    if accelerator.is_main_process:
        wandb.init(
            project="SD3-ImageNet",
            config=OmegaConf.to_container(cfg, resolve=True)
        )
    print_r0("→ start training:")
    train_scheduler = FMEulerSampler(cfg, model, tokenizer, text_encoder, train_steps=cfg.sampler.num_train_timesteps)
    for e in range(cfg.training.epochs):
        with tqdm(train_ds, miniters=cfg.accelerator.gradient_accumulation_steps, mininterval=0.5, disable=not accelerator.is_local_main_process) as bar:
            for idx, batch in enumerate(bar):
                global_idx = idx + e*len(train_ds)
                img = batch['image']
                txt = batch['caption']
                size = batch['size']
                crop = batch['crop']

                with accelerator.accumulate(model):
                    # encode caption
                    prompts = ["{size: "+str(size[i].tolist())+", crop: ""[" + ", ".join(f"{v:.2f}" for v in crop[i]) + "]"+", prompt: "+txt[i]+"}" for i in range(len(txt))]
                    enc = tokenizer(prompts, truncation=True, padding=True, max_length=cfg.text.max_length, return_tensors="pt")
                    ids = enc['input_ids'].to(device)
                    att = enc.get('attention_mask', None).to(device)
                    outputs = text_encoder.forward(input_ids=ids, attention_mask=att, output_hidden_states=True)
                    hidden = outputs.hidden_states
                    txt_latents = hidden[-1].detach()
                    # condition drop out
                    zeros_latents = torch.zeros_like(txt_latents)
                    p = (torch.rand((cfg.training.batch_size,1, 1)).to(device)>0.1).float()
                    txt_latents = p*txt_latents + (1-p)*zeros_latents

                    # noise image
                    time = torch.randint(train_scheduler.train_steps, (cfg.training.batch_size,), device=img.device)
                    noise = torch.randn_like(img)
                    noisy_sample = train_scheduler.add_noise(img, time, noise)

                    # predict and compute loss
                    target = img - noise
                    pred = model(hidden_states=noisy_sample,
                                encoder_hidden_states=txt_latents,
                                pooled_projections=torch.zeros(cfg.training.batch_size, cfg.model.pooled_projection_dim).to(device),
                                timestep=time,
                                return_dict=False)[0]
                    
                    loss = F.mse_loss(pred, target)

                    accelerator.backward(loss)
                    optimizer.step()
                    optimizer.zero_grad()
                if idx % cfg.accelerator.gradient_accumulation_steps == 0:
                    bar.set_postfix_str(f"epoch [{e}/{cfg.training.epochs}] mse: {loss.item():.3f}")
                    if accelerator.is_main_process:
                        wandb.log({"epoch": e, "global_step": global_idx, "train_loss": loss.item()})

                if global_idx % cfg.logging.log_images_every_n_steps == 0 and accelerator.is_main_process:
                    # generate image
                    with torch.no_grad():
                        prompt = "This is a close-up photograph of a pair of purple thistle flowers on a plant. "\
                            "The flowers have vibrant purple spines around the edges and are supported by thin stems. "\
                            "They are surrounded by greenery with other leaves visible in the background, "\
                            "suggesting the image might have been taken in a garden or a natural setting. "\
                            "The focus of the image is on the flowers, highlighting their details and colors. "\
                            "The thistle has a somewhat spiky appearance, which is typical for this type of flower. "\
                            "The background is softly focused, which puts emphasis on the flowers themselves."
                        xt = torch.randn_like(img)[0:1]
                        image = train_scheduler.generate(xt, prompt)
                        image = wandb.Image(image, caption=prompt)
                        wandb.log({"generated_image": image})
                
                if global_idx % cfg.checkpoint.every_n_steps == 0 and accelerator.is_main_process:
                    path = f"{cfg.checkpoint.save_dir}/epoch_{e}_step_{global_idx}.ckpt"
                    print_r0(f"→ saving checkpoint to {path}")
                    accelerator.save_state(path)

            
if __name__ == "__main__":
    main()
