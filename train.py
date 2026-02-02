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
import wandb



@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg):

    print("→ preparing accelerator...", end='', flush=True)
    accelerator = Accelerator(**(cfg.accelerator))
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
    train_scheduler = FMEulerSampler(train_steps=cfg.sampler.num_train_timesteps)
    print_r0(" done.✅")

    print_r0("→ loading text encoder...", end='', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg.text.model)
    text_encoder = AutoModelForCausalLM.from_pretrained(cfg.text.model)
    text_encoder.eval()
    print_r0(" done.✅")

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
    for e in range(cfg.training.epochs):
        with tqdm(train_ds, miniters=cfg.accelerator.gradient_accumulation_steps, mininterval=0.5, disable=not accelerator.is_local_main_process) as bar:
            for idx, batch in enumerate(bar):
                img = batch['image']
                txt = batch['caption']
                size = batch['size']
                crop = batch['crop']
                #mask = batch['mask']

                with accelerator.accumulate(model):
                    # encode caption
                    prompts = ["{size: "+str(size[i].tolist())+", crop: ""[" + ", ".join(f"{v:.2f}" for v in crop[i]) + "]"+", prompt: "+txt[i]+"}" for i in range(len(txt))]
                    enc = tokenizer(prompts, truncation=True, padding=True, max_length=cfg.text.max_length, return_tensors="pt")
                    ids = enc['input_ids'].to(device)
                    att = enc.get('attention_mask', None).to(device)
                    outputs = text_encoder.forward(input_ids=ids, attention_mask=att, output_hidden_states=True)
                    hidden = outputs.hidden_states
                    txt_latents = hidden[-1].detach()

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
                        wandb.log({"epoch": e, "global_tep": idx+(e*len(train_ds)), "train_loss": loss.item()})

                if idx % cfg.logging.log_images_every_n_steps == 0 and accelerator.is_main_process:
                    # generate image
                    with torch.no_grad():
                        print(f"prompts: {prompts[0]}")
                        xt = torch.randn_like(img)
                        for t in train_scheduler.get_timesteps(50):
                            t = torch.tensor([t,]).to(device)
                            pred = model(hidden_states=xt,
                                encoder_hidden_states=txt_latents,
                                pooled_projections=torch.zeros(cfg.training.batch_size, cfg.model.pooled_projection_dim).to(device),
                                timestep=t,
                                return_dict=False)[0].detach()
                            xt = train_scheduler.step(xt, pred, 50)
                        image = (xt[0].cpu().permute(1,2,0)/2.+0.5) # between 0 and 1
                        image = np.uint8(255*image.numpy())
                        image = wandb.Image(image, caption=prompts[0][0:256])
                        wandb.log({"generated_image": image})

            
if __name__ == "__main__":
    main()


      
    

