import os
import itertools
import pickle
from tqdm import tqdm

import numpy as np

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.distributions import Categorical
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from network.modules import Generator, FactorEncoder, ResidualEncoder, VGGFeatures, VGGDistance
from network.utils import NamedTensorDataset

from evaluation import dci, classifier


class FactorModel(nn.Module):

	def __init__(self, config):
		super().__init__()

		self.config = config

		self.factor_embeddings = nn.ModuleList([
			nn.Embedding(
				num_embeddings=config['factor_sizes'][f],
				embedding_dim=config['factor_dim'],
				_weight=torch.rand((config['factor_sizes'][f], config['factor_dim'])) * 0.05
			)

			for f in range(config['n_factors'])
		])

		self.factor_encoders = nn.ModuleList([
			FactorEncoder(img_shape=config['img_shape'], latent_dim=config['factor_sizes'][f])
			for f in range(config['n_factors'])
		])

	def forward(self, img, factors, label_masks):
		factor_codes = []

		assignments = []
		for f in range(self.config['n_factors']):
			assignment = Categorical(logits=self.factor_encoders[f](img))

			factor_code = (
				self.factor_embeddings[f](factors[:, f]) * label_masks[:, [f]]
				+ torch.matmul(assignment.probs, self.factor_embeddings[f].weight) * (~label_masks[:, [f]])
			)

			assignments.append(assignment)
			factor_codes.append(factor_code)

		return torch.cat(factor_codes, dim=1), assignments


class LatentModel(nn.Module):

	def __init__(self, config):
		super().__init__()

		self.config = config

		self.factor_model = FactorModel(config)

		self.residual_embeddings = nn.Embedding(
			num_embeddings=config['n_imgs'],
			embedding_dim=config['residual_dim'],
			_weight=torch.rand((config['n_imgs'], config['residual_dim'])) * 0.05
		)

		self.generator = Generator(
			latent_dim=config['n_factors'] * config['factor_dim'] + config['residual_dim'],
			img_size=config['img_shape'][0]
		)


class Model:

	def __init__(self, config):
		super().__init__()

		self.config = config
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		self.latent_model = None
		self.amortized_model = None

		self.vgg_features = VGGFeatures()
		self.perceptual_loss = VGGDistance(self.vgg_features, config['perceptual_loss']['layers'])

		self.rs = np.random.RandomState(seed=1337)

	@staticmethod
	def load(checkpoint_dir):
		with open(os.path.join(checkpoint_dir, 'config.pkl'), 'rb') as config_fd:
			config = pickle.load(config_fd)

		model = Model(config)

		if os.path.exists(os.path.join(checkpoint_dir, 'latent.pth')):
			model.latent_model = LatentModel(config)
			model.latent_model.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'latent.pth')))

		return model

	def save(self, checkpoint_dir):
		if not os.path.exists(checkpoint_dir):
			os.mkdir(checkpoint_dir)

		with open(os.path.join(checkpoint_dir, 'config.pkl'), 'wb') as config_fd:
			pickle.dump(self.config, config_fd)

		if self.latent_model:
			torch.save(self.latent_model.state_dict(), os.path.join(checkpoint_dir, 'latent.pth'))

	def train(self, imgs, factors, label_masks, model_dir, tensorboard_dir):
		self.latent_model = LatentModel(self.config)

		data = dict(
			img=torch.from_numpy(imgs).permute(0, 3, 1, 2),
			img_id=torch.from_numpy(np.arange(imgs.shape[0])),
			factors=torch.from_numpy(factors.astype(np.int64)),
			label_masks=torch.from_numpy(label_masks.astype(np.bool))
		)

		dataset = NamedTensorDataset(data)
		data_loader = DataLoader(
			dataset, batch_size=self.config['train']['batch_size'],
			shuffle=True, pin_memory=True, drop_last=False
		)

		optimizer = Adam([
			{
				'params': itertools.chain(
					self.latent_model.factor_model.factor_embeddings.parameters(),
					self.latent_model.residual_embeddings.parameters()
				),

				'lr': self.config['train']['learning_rate']['latent']
			},
			{
				'params': self.latent_model.factor_model.factor_encoders.parameters(),
				'lr': self.config['train']['learning_rate']['encoder']
			},
			{
				'params': self.latent_model.generator.parameters(),
				'lr': self.config['train']['learning_rate']['generator']
			}
		], betas=(0.5, 0.999))

		scheduler = CosineAnnealingLR(
			optimizer,
			T_max=self.config['train']['n_epochs'] * len(data_loader),
			eta_min=self.config['train']['learning_rate']['min']
		)

		self.latent_model.to(self.device)
		self.vgg_features.to(self.device)

		summary = SummaryWriter(log_dir=tensorboard_dir)
		for epoch in range(self.config['train']['n_epochs']):
			self.latent_model.train()

			pbar = tqdm(iterable=data_loader)
			for batch in pbar:
				batch = {name: tensor.to(self.device) for name, tensor in batch.items()}

				losses = self.train_latent_generator(batch)
				loss_total = 0
				for term, loss in losses.items():
					loss_total += self.config['train']['loss_weights'][term] * loss

				optimizer.zero_grad()
				loss_total.backward()
				optimizer.step()
				scheduler.step()

				pbar.set_description_str('epoch #{}'.format(epoch))
				pbar.set_postfix(loss=loss_total.item())

			pbar.close()

			summary.add_scalar(tag='loss/generator', scalar_value=loss_total.item(), global_step=epoch)

			for term, loss in losses.items():
				summary.add_scalar(tag='loss/generator/{}'.format(term), scalar_value=loss.item(), global_step=epoch)

			if epoch % 10 == 0:
				latent_factors = self.encode_factors(dataset)
				scores = dci.compute_dci(latent_factors, factors)

				summary.add_scalar(tag='dci/informativeness', scalar_value=scores['informativeness_test'], global_step=epoch)
				summary.add_scalar(tag='dci/disentanglement', scalar_value=scores['disentanglement'], global_step=epoch)
				summary.add_scalar(tag='dci/completeness', scalar_value=scores['completeness'], global_step=epoch)

				latent_residuals = self.encode_residuals(dataset)
				for factor_idx, factor_name in enumerate(self.config['factor_names']):
					acc_train, acc_test = classifier.logistic_regression(latent_residuals, factors[:, factor_idx])
					summary.add_scalar(tag='residual/{}'.format(factor_name), scalar_value=acc_test, global_step=epoch)

			for factor_idx, factor_name in enumerate(self.config['factor_names']):
				figure_fixed = self.visualize_translation(dataset, factor_idx, randomized=False)
				figure_random = self.visualize_translation(dataset, factor_idx, randomized=True)

				summary.add_image(tag='{}-fixed'.format(factor_name), img_tensor=figure_fixed, global_step=epoch)
				summary.add_image(tag='{}-random'.format(factor_name), img_tensor=figure_random, global_step=epoch)

			self.save(model_dir)

		summary.close()

	def train_latent_generator(self, batch):
		factor_codes, assignments = self.latent_model.factor_model(batch['img'], batch['factors'], batch['label_masks'])
		residual_code = self.latent_model.residual_embeddings(batch['img_id'])

		if self.config['residual_std'] != 0:
			noise = torch.zeros_like(residual_code)
			noise.normal_(mean=0, std=self.config['residual_std'])

			residual_code_regularized = residual_code + noise
		else:
			residual_code_regularized = residual_code

		latent_code_regularized = torch.cat((factor_codes, residual_code_regularized), dim=1)
		img_reconstructed = self.latent_model.generator(latent_code_regularized)
		loss_reconstruction = self.perceptual_loss(img_reconstructed, batch['img'])

		loss_entropy = torch.stack([a.entropy() for a in assignments], dim=1).mean()
		loss_residual_decay = torch.mean(residual_code ** 2, dim=1).mean()

		return {
			'reconstruction': loss_reconstruction,
			'entropy': loss_entropy,
			'residual_decay': loss_residual_decay
		}

	@torch.no_grad()
	def encode_factors(self, dataset):
		self.latent_model.eval()

		codes = []
		data_loader = DataLoader(dataset, batch_size=64, shuffle=False, pin_memory=True, drop_last=False)
		for batch in data_loader:
			batch = {name: tensor.to(self.device) for name, tensor in batch.items()}

			batch_codes, _ = self.latent_model.factor_model(batch['img'], batch['factors'], batch['label_masks'])
			codes.append(batch_codes.cpu())

		codes = torch.cat(codes, dim=0)
		return torch.stack(torch.split(codes, split_size_or_sections=self.config['factor_dim'], dim=1), dim=1).numpy()

	@torch.no_grad()
	def encode_residuals(self, dataset):
		self.latent_model.eval()

		codes = []
		data_loader = DataLoader(dataset, batch_size=64, shuffle=False, pin_memory=True, drop_last=False)
		for batch in data_loader:
			batch = {name: tensor.to(self.device) for name, tensor in batch.items()}

			batch_codes = self.latent_model.residual_embeddings(batch['img_id'])
			codes.append(batch_codes.cpu())

		codes = torch.cat(codes, dim=0)
		return codes

	@torch.no_grad()
	def visualize_translation(self, dataset, factor_idx, n_samples=10, randomized=False, amortized=False):
		random = self.rs if randomized else np.random.RandomState(seed=0)
		img_idx = torch.from_numpy(random.choice(len(dataset), size=n_samples, replace=False))
		batch = dataset[img_idx]
		batch = {name: tensor.to(self.device) for name, tensor in batch.items()}

		if amortized:
			self.amortized_model.eval()
			pass

		else:
			self.latent_model.eval()

			batch['factor_codes'], _ = self.latent_model.factor_model(batch['img'], batch['factors'], batch['label_masks'])
			batch['residual_code'] = self.latent_model.residual_embeddings(batch['img_id'])

		generator = self.amortized_model.generator if amortized else self.latent_model.generator

		figure = []
		for i in range(n_samples):
			converted_imgs = [batch['img'][i]]

			factor_codes = list(torch.split(batch['factor_codes'][i], split_size_or_sections=self.config['factor_dim'], dim=0))
			factor_values = torch.arange(self.config['factor_sizes'][factor_idx], dtype=torch.int64).to(self.device)
			factor_embeddings = self.latent_model.factor_model.factor_embeddings[factor_idx](factor_values)

			for j in range(factor_embeddings.shape[0]):
				factor_codes[factor_idx] = factor_embeddings[j]
				latent_code = torch.cat(factor_codes + [batch['residual_code'][i]], dim=0)
				converted_img = generator(latent_code.unsqueeze(dim=0))
				converted_imgs.append(converted_img[0])

			figure.append(torch.cat(converted_imgs, dim=2))

		figure = torch.cat(figure, dim=1)
		return figure.clamp(min=0, max=1)
