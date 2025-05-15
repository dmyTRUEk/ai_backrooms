# next frame predictor

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

# Hyper-cutesy parameters UwU
BATCH_SIZE = 8
EPOCHS = 10 if not IS_TEST else 2
LEARNING_RATE = 1e-3
IMG_SIZE = (240, 320)
DATASET_SIZE_LIMIT = 10**4 if not IS_TEST else 30
ACTIONS_NS = [1, 3, 10] if not IS_TEST else [1, 3]

INITIAL_FRAME = '../traininggrounds/screenshots/r001/s_000000_l.jpg'



device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'DEVICE: {device_str}')
device = torch.device(device_str)

def main():
	model = FramePredictor().to(device)
	print('Model created.')
	this_frame = load_img(INITIAL_FRAME)
	frame_i = 1
	while (inp:=input('> ')) != 'q':
		try:
			if inp in ['?', 'h', 'help']:
				print(HELP_MSG)

			elif inp == 'reset':
				model = FramePredictor().to(device)
				print('Model reset.')

			elif inp == '0':
				this_frame = load_img(INITIAL_FRAME)

			elif inp[0] in ['f', 'b', 'l', 'r']:
				n = max(1, int('0'+inp[1:]))
				for _ in range(n):
					this_frame = predict_next_frame_prod(model, this_frame, action_to_tensor(inp[0]))
					save_img(this_frame, frame_i)
					frame_i += 1

			elif inp == 't':
				print('Input training params:')

				epochs = input(f'Epochs (default is {EPOCHS}): ')
				epochs = int(epochs) if epochs != '' else EPOCHS

				learning_rate = input(f'Learning Rate (default is {LEARNING_RATE}): ')
				learning_rate = float(learning_rate) if learning_rate != '' else LEARNING_RATE

				dataset_size_limit = input(f'Dataset Size Limit (default is {DATASET_SIZE_LIMIT}): ')
				dataset_size_limit = int(dataset_size_limit) if dataset_size_limit != '' else DATASET_SIZE_LIMIT

				actions_ns = input(f'Actions NS (default is {ACTIONS_NS}): ')
				actions_ns = list(map(int, string_multisplit(actions_ns, [',', ' ']))) if actions_ns != '' else ACTIONS_NS

				print()
				print('Starting training...')
				print()

				for actions_n in actions_ns:
					train(model, epochs=epochs, learning_rate=learning_rate, dataset_size_limit=dataset_size_limit, actions_n=actions_n)
					print()

			# TODO: save/load NN to/from a file

			else:
				print('Unknown input. Use `?` or `h` or `help` to get help.')

		except Exception as e:
			print(f'Error occured: {e}')



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
def train(
	model,
	*,
	epochs: int,
	learning_rate: float,
	dataset_size_limit: None | int,
	actions_n: int,
):
	dataset = FrameDataset('../traininggrounds/screenshots/r001/', dataset_size_limit=dataset_size_limit, actions_n=actions_n)
	dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # TODO: try alternatives: AdamW, Lion
	criterion = nn.MSELoss()
	model.train()
	# torch.autograd.set_detect_anomaly(True)

	for epoch in range(epochs):
		total_loss = 0
		i_max = len(dataloader)

		print(f'/{i_max}:', end='', flush=True)
		for traindata_i, (imgs, actions) in enumerate(dataloader):
			print(f' {traindata_i+1}', end='', flush=True)

			this_img = imgs[0].to(device)
			for i, action in enumerate(actions):
				# input:
				# this_img = imgs[i].to(device)
				action = actions[i].to(device)
				next_img = imgs[i+1].to(device)

				optimizer.zero_grad()
				output = model(this_img, action)
				loss = criterion(output, next_img)
				loss.backward(retain_graph=True)
				optimizer.step()

				total_loss += loss.item()

				this_img = output.detach()

		print()
		print(f'--- Actions: {actions_n} --- Epoch: {epoch+1}/{epochs} --- Loss: {total_loss:.4f} ---')
		# TODO: relative loss
		# TODO: time spent
	# TODO: time spent



# UwU Dataset Loader
class FrameDataset(Dataset):
	def __init__(self, folder: str, *, dataset_size_limit: None | int = None, actions_n: int):
		assert actions_n > 0
		self.folder = folder
		self.samples: list[tuple[list[str], list[str]]] = []
		self.img_cache: dict[str, Image.Image] = {} # Cache imgs for that extra zoomies >:3

		all_files = sorted(os.listdir(folder))
		for file_i, filename in enumerate(all_files): # TODO: random order
			if file_i+actions_n >= len(all_files):
				break
			if dataset_size_limit is not None and len(self.samples) >= dataset_size_limit:
				break

			filenames = [filename]
			actions = []
			for k in range(1, actions_n+1):
				filenames.append(all_files[file_i+k])
				actions.append(action_from_filename(all_files[file_i+k-1]))
			self.samples.append((filenames.copy(), actions.copy()))

		# print(self.samples)
		print(f'TRAINING SET SIZE: {len(self.samples)}')


	def __len__(self):
		return len(self.samples)


	# @lru_cache(maxsize=None) # bad?
	def __getitem__(self, idx):
		img_names, actions = self.samples[idx]

		imgs = []
		for img_name in img_names:
			if self.img_cache.get(img_name) is None:
				img = Image.open(os.path.join(self.folder, img_name)).convert('RGB')
				self.img_cache[img_name] = img
			else:
				img = self.img_cache[img_name]
			imgs.append(img)

		img_tensors = [img_pil_to_tensor(img) for img in imgs]
		action_tensors = [action_to_tensor(action) for action in actions]

		return img_tensors, action_tensors



def action_from_filename(filename: str) -> str:
	_f, _idx, action = filename.split('.')[0].split('_')
	return action


# Helper to turn action into one-hot vector :3
def action_to_tensor(action):
	action_dict = {'f': 0, 'b': 1, 'l': 2, 'r': 3}
	vec = torch.zeros(4)
	vec[action_dict[action]] = 1.0
	return vec



# Inference function desu~
def predict_next_frame_prod(model, this_img, action):
	model.eval()
	with torch.no_grad():
		this_img = this_img.to(device).unsqueeze(0)
		action = action.to(device).unsqueeze(0)
		predicted = model(this_img, action)
		return predicted.squeeze(0).cpu()



def load_img(filepath: str):
	img_pil = Image.open(filepath).convert('RGB')
	return img_pil_to_tensor(img_pil)

def save_img(img_tensor, frame_i: int):
	filepath = f'frames/f_{frame_i:06}.jpg'
	img_pil = img_tensor_to_pil(img_tensor)
	img_pil.save(filepath)



def img_pil_to_tensor(img_pil):
	return torch.from_numpy(np.array(img_pil, dtype=np.float32) / 255).permute(2, 0, 1)

def img_tensor_to_pil(img_tensor):
	img = img_tensor.detach().cpu().clamp(0, 1)  # safety uwu
	arr_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
	img_pil = Image.fromarray(arr_np)
	return img_pil



def string_multisplit(string: str, seps: list[str]) -> list[str]:
	prev = [string]
	for sep in seps:
		next = []
		for s in prev:
			next.extend(s.split(sep))
		prev = next
	return [s for s in prev if s != '']





HELP_MSG = '''\
todo help msg
'''





if __name__ == '__main__':
	main()

