import os

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

class DatasetAgent:
    def __init__(self, state_size, action_size, is_best=False):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.model = self._build_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.best_score = 0

        if is_best:
            self.load_best_model()

    def _build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Dropout(0.3),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Dropout(0.3),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Dropout(0.3),
            nn.Linear(512, self.action_size)
        )
        return model

    def remember(self, state, action):
        self.memory.append((state, action))

    def act(self, state):
        self.model.eval()
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            act_values = self.model(state)
        return np.argmax(act_values.cpu().data.numpy())

    def load_best_model(self, file_path='best_dataset_model.pth'):
        if os.path.isfile(file_path):
            try:
                self.model.load_state_dict(torch.load(file_path), strict=False)
                self.model.eval()
                print(f"Model loaded successfully from {file_path}")
            except RuntimeError as e:
                print(f"Error loading model: {e}")
        else:
            print(f"No model found at {file_path}")

    def save_best_model(self, file_path='best_dataset_model.pth'):
        torch.save(self.model.state_dict(), file_path)
        torch.save(self.best_score, 'best_dataset_score.pth')
        print(f"Best Dataset model and score saved: {self.best_score}")

    def train(self, batch_size=64):
        if len(self.memory) < batch_size:
            return
        self.model.train()
        minibatch = random.sample(self.memory, batch_size)
        for state, action in minibatch:
            target_f = self.model(torch.FloatTensor(state).unsqueeze(0))
            target_f[0][action] = target_f[0][action] * 0.99
            self.optimizer.zero_grad()
            loss = self.criterion(target_f[0][action], torch.tensor(1.0))
            loss.backward()
            self.optimizer.step()
