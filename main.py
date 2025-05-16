# next frame predictor

from datetime import datetime
from math import log
import os
# from sys import exit as sys_exit

from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from pipe import enumerate as enumerate_, map as map_, filter as filter_
from pipe_ext import int_, list_, shuffled_, sorted_, string_multisplit_, time_to_my_format_, to_base36_


__version__ = "0.4.0"


IS_TEST = False
# IS_TEST = True

# Hyper-cutesy parameters UwU
BATCH_SIZE = 8
EPOCHS = 10 if not IS_TEST else 2
LEARNING_RATE = 1e-3
IMG_SIZE = (240, 320)
DATASET_SIZE_LIMIT = 10**4 if not IS_TEST else 30
PREDICTION_DEPTHS = [1, 2, 3]

DATASET_PATH  = 'datasets/r002-111k/'
INITIAL_FRAME = 'initial_frame.jpg'



device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'DEVICE: {device_str}')
device = torch.device(device_str)

def main():
	model = FramePredictor().to(device)
	print('Model created.')
	this_frame = load_img(INITIAL_FRAME)
	frame_i = 1
	while (inp:=input('> ')) not in ['q', 'quit', 'exit']:
		try:
			if inp == '':
				pass

			elif inp in ['?', 'h', 'help']:
				print(HELP_MSG)

			elif inp == 'reset':
				model = FramePredictor().to(device)
				print('Model reset.')

			elif inp == '0':
				this_frame = load_img(INITIAL_FRAME)

			elif inp == 't':
				print('Input training params:')

				epochs = input(f'Epochs (default is {EPOCHS}): ')
				epochs = int(epochs) if epochs != '' else EPOCHS

				learning_rate = input(f'Learning Rate (default is {LEARNING_RATE}): ')
				learning_rate = float(learning_rate) if learning_rate != '' else LEARNING_RATE

				dataset_size_limit = input(f'Dataset Size Limit (default is {DATASET_SIZE_LIMIT}): ')
				dataset_size_limit = int(dataset_size_limit) if dataset_size_limit != '' else DATASET_SIZE_LIMIT

				pds = input(f'Prediction Depths (default is {PREDICTION_DEPTHS}): ')
				pds = pds | string_multisplit_([',', ' ']) | map_(int) | list_ if pds != '' else PREDICTION_DEPTHS

				print()
				print('Starting training...')
				print()

				time_trains_begin = datetime.now()
				for pd in pds:
					train(model, epochs=epochs, learning_rate=learning_rate, dataset_size_limit=dataset_size_limit, prediction_depth=pd)
					save_nn(model, autosave=True)
					print()
				time_trains_end = datetime.now()
				time_trains = time_trains_end - time_trains_begin
				print(f'TOTAL TIME: {time_trains | time_to_my_format_}')

			elif inp == 'save':
				save_nn(model, autosave=False)

			elif inp == 'load':
				model = load_nn()

			elif inp[0] in ['f', 'b', 'l', 'r']:
				n = max(1, int('0'+inp[1:]))
				for _i in range(n):
					this_frame = predict_next_frame_prod(model, this_frame, action_to_tensor(inp[0]))
					save_img(this_frame, frame_i)
					frame_i += 1

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
	prediction_depth: int,
):
	dataset = FrameDataset(DATASET_PATH, dataset_size_limit=dataset_size_limit, prediction_depth=prediction_depth)
	dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

	optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate) # good options: Adam, AdamW, Lion?
	criterion = nn.MSELoss()
	model.train()
	# torch.autograd.set_detect_anomaly(True)

	time_train_begin = datetime.now()
	for epoch in range(epochs):
		time_epoch_begin = datetime.now()
		total_loss: float = 0
		total_loss_n: int = 0
		i_max = len(dataloader)

		print(f'/{i_max | to_base36_}:', end='', flush=True)
		for traindata_i, (imgs, actions) in dataloader | enumerate_:
			print(f' {traindata_i+1 | to_base36_}', end='', flush=True)

			this_img = imgs[0].to(device)
			for i, action in actions | enumerate_:
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
				total_loss_n += 1

				this_img = output.detach()

		rel_loss = total_loss / total_loss_n
		time_epoch_end = datetime.now()
		time_epoch = time_epoch_end - time_epoch_begin
		print()
		print(f'--- Prediction Depth: {prediction_depth} --- Epoch: {epoch+1}/{epochs} --- Loss Log: {-log(total_loss):.3f} --- Relative Loss Log: {-log(rel_loss):.3f} --- Time: {time_epoch | time_to_my_format_} ---')

	time_train_end = datetime.now()
	time_train = time_train_end - time_train_begin
	print(f'TIME: {time_train | time_to_my_format_}')



# UwU Dataset Loader
class FrameDataset(Dataset):
	def __init__(self, folder: str, *, dataset_size_limit: None | int = None, prediction_depth: int):
		assert prediction_depth > 0
		self.folder = folder
		self.samples: list[tuple[list[str], list[str]]] = []
		self.img_cache: dict[str, Image.Image] = {} # Cache imgs for that extra zoomies >:3

		all_files = os.listdir(folder) | sorted_
		for file_i, filename in all_files | enumerate_ | list_ | shuffled_:
			if dataset_size_limit is not None and len(self.samples) >= dataset_size_limit:
				break
			if file_i+prediction_depth >= len(all_files):
				continue

			filenames = [filename]
			actions = []
			for k in range(1, prediction_depth+1):
				filenames.append(all_files[file_i+k])
				actions.append(action_from_filename(all_files[file_i+k-1]))
			self.samples.append((filenames.copy(), actions.copy()))

		# print(self.samples)
		print(f'TRAINING SET SIZE: {len(self.samples)}')


	def __len__(self):
		return len(self.samples)


	# @functools.lru_cache(maxsize=None) # for some reason its bad
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



def save_nn(model: FramePredictor, *, autosave: bool):
	now = datetime.now()
	default_model_filename = f'model_{now.year}-{now.month:02}-{now.day:02}_{now.hour:02}-{now.minute:02}-{now.second:02}.pth'
	if not autosave:
		model_filename = input(f'Filename (default is `{default_model_filename}`): ')
		model_filename = model_filename if model_filename != '' else default_model_filename
	else:
		model_filename = default_model_filename
	print(f'Saving model to `{model_filename}`... ', end='', flush=True)
	torch.save(model.state_dict(), model_filename)
	print('Saved successfully.')

def load_nn() -> FramePredictor:
	files_in_dir = os.listdir() | sorted_
	models_in_dir = files_in_dir | filter_(lambda filename: filename.endswith('.pth')) | list_
	for i, filename in models_in_dir | enumerate_:
		print(f'{i}. {filename}')
	model_filename = input(f'Path to file or number (default is last): ')
	try:
		model_filename_n = model_filename | int_
		assert model_filename_n >= 0
		model_filename = models_in_dir[model_filename_n]
	except ValueError:
		model_filename = model_filename if model_filename != '' else models_in_dir[-1]
	model = FramePredictor()
	print(f'Loading model from `{model_filename}`... ', end='', flush=True)
	model.load_state_dict(torch.load(model_filename, weights_only=True))
	model.to(device)
	print('Loaded successfully.')
	return model





HELP_MSG = f'''\
Backrooms AI v{__version__}
Internal CLI commands:
? h help   	get help message
reset      	reset NN
t          	train NN
0          	reset frame sequence
f<number>  	move forward  <number> times
b<number>  	move backward <number> times
l<number>  	move left     <number> times
r<number>  	move right    <number> times
save       	save NN to file
load       	load NN from file
q quit exit	quit
'''[:-1]





if __name__ == '__main__':
	main()

