import os
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# from functools import lru_cache
# from sys import exit as sys_exit



IS_TEST = False
# IS_TEST = True

# Hyper-cutesy parameters
BATCH_SIZE = 8
EPOCHS = 100 if not IS_TEST else 2
LEARNING_RATE = 1e-3
IMG_SIZE = (240, 320)
DATASET_FOR_TEST_LIMIT = 30



device_str = "cuda" if torch.cuda.is_available() else "cpu"
print(f"DEVICE: {device_str}")
device = torch.device(device_str)

def main():
	# Setup
	dataset = FrameDataset("../traininggrounds/screenshots/r001/")
	dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

	model = FramePredictor().to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
	criterion = nn.MSELoss()

	train(model, dataloader, optimizer, criterion, EPOCHS)
	print()

	# "game" loop:
	this_frame = load_img("../traininggrounds/screenshots/r001/s_000000_l.jpg")
	frame_n = 1
	while (input_:=input("fblr0? ")) != "q":
		# TODO: match case, continue training
		if input_ not in {"f", "b", "l", "r", "0"}: continue
		if input_ == "0":
			this_frame = load_img("../traininggrounds/screenshots/r001/s_000000_l.jpg")
			continue
		this_frame = predict_next_frame(model, this_frame, action_to_tensor(input_))
		save_img(this_frame, frame_n)
		frame_n += 1



# Neural Net: image + action -> next image
class FramePredictor(nn.Module):
	def __init__(self):
		super(FramePredictor, self).__init__()
		# Encoder for this image
		self.encoder = nn.Sequential(
			nn.Conv2d(3, 32, 4, stride=2, padding=1),  # 120x160
			nn.ReLU(),
			nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 60x80
			nn.ReLU(),
			nn.Conv2d(64, 128, 4, stride=2, padding=1),  # 30x40
			nn.ReLU(),
		)
		self.action_fc = nn.Linear(4, 128 * 30 * 40)

		# Decoder
		self.decoder = nn.Sequential(
			nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 60x80
			nn.ReLU(),
			nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 120x160
			nn.ReLU(),
			nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),  # 240x320
			nn.Sigmoid()
		)


	def forward(self, image, action):
		x = self.encoder(image)
		a = self.action_fc(action).view(-1, 128, 30, 40)
		combined = torch.cat([x, a], dim=1)
		return self.decoder(combined)



# Training time OwO <3
def train(model, dataloader, optimizer, criterion, epochs: int):
	model.train()
	for epoch in range(epochs):
		total_loss = 0
		i_max = len(dataloader)
		for i, (this_img, action, next_img) in enumerate(dataloader):
			print(f"{i+1}/{i_max} ", end="", flush=True)
			this_img = this_img.to(device)
			action = action.to(device)
			next_img = next_img.to(device)

			optimizer.zero_grad()
			output = model(this_img, action)
			loss = criterion(output, next_img)
			loss.backward()
			optimizer.step()

			total_loss += loss.item()

		print(f"\nEpoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")



# UwU Dataset Loader
class FrameDataset(Dataset):
	def __init__(self, folder):
		self.folder = folder
		self.samples = []
		self.img_cache: dict[int, tuple] = {} # Cache imgs for that extra zoomies >:3

		all_files = set(os.listdir(folder))
		# print(all_files)
		for filename in all_files:
			_, idx, action = filename.split(".")[0].split("_")
			idx = int(idx)
			if IS_TEST and len(self.samples) >= DATASET_FOR_TEST_LIMIT:
				break

			next_filename = f"s_{idx+1:06}_f.jpg"
			if next_filename in all_files:
				# next_action = "f"
				self.samples.append((filename, next_filename, action))

			next_filename = f"s_{idx+1:06}_b.jpg"
			if next_filename in all_files:
				# next_action = "b"
				self.samples.append((filename, next_filename, action))

			next_filename = f"s_{idx+1:06}_l.jpg"
			if next_filename in all_files:
				# next_action = "l"
				self.samples.append((filename, next_filename, action))

			next_filename = f"s_{idx+1:06}_r.jpg"
			if next_filename in all_files:
				# next_action = "r"
				self.samples.append((filename, next_filename, action))

		# print(self.samples)
		print(f"TRAINING SET SIZE: {len(self.samples)}")


	def __len__(self):
		return len(self.samples)


	# @lru_cache(maxsize=None) # bad?
	def __getitem__(self, idx):
		this_img_name, next_img_name, action = self.samples[idx]
		if self.img_cache.get(idx) is None:
			this_img = Image.open(os.path.join(self.folder, this_img_name)).convert('RGB')
			next_img = Image.open(os.path.join(self.folder, next_img_name)).convert('RGB')
			self.img_cache[idx] = (this_img, next_img)
		else:
			this_img, next_img = self.img_cache[idx]

		this_tensor = img_pil_to_tensor(this_img)
		next_tensor = img_pil_to_tensor(next_img)
		action_tensor = action_to_tensor(action)

		return this_tensor, action_tensor, next_tensor



# Helper to turn action into one-hot vector :3
def action_to_tensor(action):
	action_dict = {'f': 0, 'b': 1, 'l': 2, 'r': 3}
	vec = torch.zeros(4)
	vec[action_dict[action]] = 1.0
	return vec



# Inference function desu~
def predict_next_frame(model, this_img, action):
	model.eval()
	with torch.no_grad():
		this_img = this_img.to(device).unsqueeze(0)
		action = action.to(device).unsqueeze(0)
		predicted = model(this_img, action)
		return predicted.squeeze(0).cpu()



def load_img(filepath: str):
	img_pil = Image.open(filepath).convert('RGB')
	return img_pil_to_tensor(img_pil)

def save_img(img_tensor, frame_n: int):
	filepath = f"frames/f_{frame_n:06}.jpg"
	img_pil = img_tensor_to_pil(img_tensor)
	img_pil.save(filepath)



def img_pil_to_tensor(img_pil):
	return torch.from_numpy(np.array(img_pil, dtype=np.float32) / 255).permute(2, 0, 1)

def img_tensor_to_pil(img_tensor):
	img = img_tensor.detach().cpu().clamp(0, 1)  # safety uwu
	arr_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
	img_pil = Image.fromarray(arr_np)
	return img_pil




if __name__ == "__main__":
	main()

