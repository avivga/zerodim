from torch.utils.data.dataset import Dataset


class ImageTensorDataset(Dataset):

	def __init__(self, named_tensors):
		assert all(list(named_tensors.values())[0].size(0) == tensor.size(0) for tensor in named_tensors.values())
		self.named_tensors = named_tensors

	def __getitem__(self, index):
		item = {name: tensor[index] for name, tensor in self.named_tensors.items()}

		if 'img' in item:
			item['img'] = item['img'].float() / 255.0

		return item

	def __len__(self):
		return list(self.named_tensors.values())[0].size(0)
