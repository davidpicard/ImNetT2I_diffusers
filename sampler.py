import torch
import numpy as np

class FMEulerSampler():
    def __init__(self, cfg, model, tokenizer, text_encoder, train_steps: int = 1000):
        self.cfg = cfg
        self.train_steps = train_steps
        self.model = model
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder

    def add_noise(self, sample, t, noise):
        t = t.view(-1, 1, 1, 1)
        return sample * t/self.train_steps + noise*(1-t/self.train_steps)
    
    def get_v(self, pred, noisy_sample, t):
        return (pred - noisy_sample)/(1.-t.view(-1, 1, 1, 1).float()/self.train_steps).clamp_min(0.05)

    def get_timesteps(self, num_timesteps: int = 50):
        t = torch.linspace(0, self.train_steps-1, num_timesteps).long()
        return t

    def step(self, sample, pred, num_timesteps):
        return sample + 1./num_timesteps * pred

    def generate(self, xt, prompt: str, cfg: int=2.5, num_steps: int = 50, device: str = "cuda"):
        with torch.no_grad():
            prompts = ["{size: [512, 512], crop: [0.00, 1.00, 0.00, 1.00], prompt: "+prompt+"}"]
            enc = self.tokenizer(prompts, truncation=True, padding=True, max_length=64, return_tensors="pt")
            ids = enc['input_ids'].to(device)
            att = enc.get('attention_mask', None).to(device)
            outputs = self.text_encoder.forward(input_ids=ids, attention_mask=att, output_hidden_states=True)
            hidden = outputs.hidden_states
            txt_latents = hidden[-1].detach()

            print(f"prompts: {prompts[0]}")
            for t in self.get_timesteps(num_steps):
                t = torch.tensor([t,]).to(device)
                pred = self.model(hidden_states=xt,
                                encoder_hidden_states=txt_latents,
                                pooled_projections=torch.zeros(1, self.cfg.model.pooled_projection_dim).to(device),
                                timestep=t,
                                return_dict=False)[0].detach()
                pred = self.get_v(pred, xt, t)
                if cfg > 0:
                    u_pred = self.model(hidden_states=xt,
                                encoder_hidden_states=torch.zeros_like(txt_latents),
                                pooled_projections=torch.zeros(1, self.cfg.model.pooled_projection_dim).to(device),
                                timestep=t,
                                return_dict=False)[0].detach()
                    u_pred = self.get_v(u_pred, xt, t)
                    pred = pred + cfg*(pred - u_pred)
                xt = self.step(xt, pred, num_steps)
            image = (xt[0].cpu().permute(1,2,0)/2.+0.5).clamp(0,1) # between 0 and 1
            image = np.uint8(255*image.numpy())
            return image
