import torch


class FMEulerSampler():
    def __init__(self, train_steps: int = 1000):
        self.train_steps = train_steps

    def add_noise(self, sample, t, noise):
        t = t.view(-1, 1, 1, 1)
        return sample * t/self.train_steps + noise*(1-t/self.train_steps)

    def get_timesteps(self, num_timesteps: int = 50):
        t = torch.linspace(0, self.train_steps, num_timesteps).long()
        return t

    def step(self, sample, pred, num_timesteps):
        return sample + 1./num_timesteps * pred