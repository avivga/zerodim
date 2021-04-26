import argparse
import os

import numpy as np

import data
from assets import AssetManager
from network.training import Model
from config import base_config


def preprocess(args, extras=[]):
	assets = AssetManager(args.base_dir)

	img_dataset_def = data.supported_datasets[args.dataset_id]
	img_dataset = img_dataset_def(args.dataset_path, extras)

	np.savez(file=assets.get_preprocess_file_path(args.data_name), **img_dataset.read())


def train(args):
	assets = AssetManager(args.base_dir)
	model_dir = assets.recreate_model_dir(args.model_name)
	tensorboard_dir = assets.recreate_tensorboard_dir(args.model_name)

	data = np.load(assets.get_preprocess_file_path(args.data_name))
	imgs = data['imgs'].astype(np.float32) / 255.0

	factors = np.load(os.path.join(args.base_dir, 'cache', 'preprocess', 'celeba-clip-l5000.npy')).astype(np.int64).T
	label_masks = np.ones_like(factors).astype(np.bool)
	label_masks[factors == -1] = False
	factors[factors == -1] = 0  # dummy valid value

	factor_sizes = [np.unique(factors[:, i]).size for i in range(factors.shape[1])]

	config = dict(
		img_shape=imgs.shape[1:],
		n_imgs=imgs.shape[0],
		factor_sizes=factor_sizes,
		n_factors=len(factor_sizes),
		factor_names=["age", "gender", "ethnicity", "hair", "face", "attractive", "thin", "hairstyle", "beard", "glasses"]
	)

	# factors = data['factors'][:, :3]
	#
	# # TODO: separate partial N and incomplete D
	# label_masks = (np.random.rand(*factors.shape) < 0.1)
	#
	# # TODO: not well defined
	# # label_masks[factors == -1] = False
	# # factors[factors == -1] = 0  # dummy
	#
	# config = dict(
	# 	img_shape=imgs.shape[1:],
	# 	n_imgs=imgs.shape[0],
	# 	factor_sizes=data['factor_sizes'][:3],
	# 	n_factors=len(data['factor_sizes'][:3]),
	# 	factor_names=data['factor_names'][:3]
	# )

	config.update(base_config)

	model = Model(config)
	model.train(imgs, factors, label_masks, model_dir, tensorboard_dir)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-bd', '--base-dir', type=str, required=True)

	action_parsers = parser.add_subparsers(dest='action')
	action_parsers.required = True

	preprocess_parser = action_parsers.add_parser('preprocess')
	preprocess_parser.add_argument('-di', '--dataset-id', type=str, choices=data.supported_datasets, required=True)
	preprocess_parser.add_argument('-dp', '--dataset-path', type=str, required=True)
	preprocess_parser.add_argument('-dn', '--data-name', type=str, required=True)
	preprocess_parser.set_defaults(func=preprocess)

	train_parser = action_parsers.add_parser('train')
	train_parser.add_argument('-dn', '--data-name', type=str, required=True)
	train_parser.add_argument('-mn', '--model-name', type=str, required=True)
	train_parser.set_defaults(func=train)

	args, extras = parser.parse_known_args()
	if len(extras) == 0:
		args.func(args)
	else:
		args.func(args, extras)


if __name__ == '__main__':
	main()
