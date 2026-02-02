import torch
import torch.nn.functional as F
import numpy as np
from diffusers import SD3Transformer2DModel, FlowMatchEulerDiscreteScheduler
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
from torch.optim import AdamW
from omegaconf import DictConfig, OmegaConf
import hydra
from tqdm import tqdm
from accelerate import Accelerator
from im_dataset import CaptionImageNetDataset
import wandb

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg):

    print("→ config:")
    print(cfg)

    print("→ loading dataset...", end='', flush=True)
    ds = CaptionImageNetDataset(cfg.data.imagenet_path, im_size=cfg.data.im_size)
    train_ds = DataLoader(ds, batch_size=cfg.training.batch_size, shuffle=True, num_workers=cfg.data.num_workers, drop_last=True)
    print(" done.✅")

    print("→ loading model...", end='', flush=True)
    model = SD3Transformer2DModel(sample_size=cfg.data.im_size,out_channels=cfg.model.in_channels, **(cfg.model))
    optimizer = AdamW(model.parameters(), lr=cfg.training.lr)
    train_scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=cfg.sampler.num_train_timesteps)
    val_scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=cfg.sampler.num_train_timesteps)
    print(" done.✅")

    print("→ loading text encoder...", end='', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg.text.model)
    text_encoder = AutoModelForCausalLM.from_pretrained(cfg.text.model)
    text_encoder.eval()
    print(" done.✅")

    print("→ preparing accelerator...", end='', flush=True)
    accelerator = Accelerator(**(cfg.accelerator))
    device = accelerator.device
    model, optimizer, text_encoder, train_ds = accelerator.prepare(model, optimizer, text_encoder, train_ds)
    print(" done.✅")

    print("→ start training:")
    wandb.init(
        project="SD3-ImageNet",
        config=OmegaConf.to_container(cfg, resolve=True)
    )
    for e in range(cfg.training.epochs):
        with tqdm(train_ds, miniters=cfg.accelerator.gradient_accumulation_steps, mininterval=0.5) as bar:
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
                    time = torch.randint(cfg.sampler.num_train_timesteps, (cfg.training.batch_size,))
                    sigma = train_scheduler.sigmas[time].to(img.device)
                    time = time.long().to(img.device)
                    time = time.long().to(img.device)
                    noise = torch.randn_like(img)
                    noisy_sample = sigma.view(-1, 1, 1, 1)*noise + (1.-sigma).view(-1, 1, 1, 1)*img 

                    # predict and compute loss
                    target = img - noise
                    pred = model(hidden_states=noisy_sample,
                                encoder_hidden_states=txt_latents,
                                pooled_projections=torch.zeros(cfg.training.batch_size, 1).to(device),
                                timestep=time,
                                return_dict=False)[0]
                    
                    loss = F.mse_loss(pred, target)

                    accelerator.backward(loss)
                    optimizer.step()
                    optimizer.zero_grad()
                if idx % cfg.accelerator.gradient_accumulation_steps == 0:
                    bar.set_postfix_str(f"epoch [{e}/{cfg.training.epochs}] mse: {loss.item():.3f}")
                    wandb.log({"epoch": e, "global_tep": idx+(e*len(train_ds)), "train_loss": loss.item()})

                if idx % cfg.logging.log_images_every_n_steps == 0:
                    # generate image
                    with torch.no_grad():
                        print(f"prompts: {prompts}")
                        xt = torch.randn_like(img)
                        val_scheduler.set_timesteps(50)
                        for t in val_scheduler.timesteps:
                            t = torch.tensor([t,]).to(device)
                            pred = model(hidden_states=xt,
                                encoder_hidden_states=txt_latents,
                                pooled_projections=torch.zeros(cfg.training.batch_size, 1).to(device),
                                timestep=t,
                                return_dict=False)[0].detach()
                            xt = val_scheduler.step(pred, t, xt, return_dict=False)[0]
                        image = (xt[0].cpu().permute(1,2,0)/2.+0.5) # between 0 and 1
                        image = np.uint8(255*image.numpy())
                        image = wandb.Image(image, caption=prompts[0][0:256])
                        wandb.log({"generated_image": image})

            
if __name__ == "__main__":
    main()


      
    

