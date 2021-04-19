from torch.utils.data.dataset import Dataset


class NamedTensorDataset(Dataset):

	def __init__(self, named_tensors):
		assert all(list(named_tensors.values())[0].size(0) == tensor.size(0) for tensor in named_tensors.values())
		self.named_tensors = named_tensors

	def __getitem__(self, index):
		return {name: tensor[index] for name, tensor in self.named_tensors.items()}

	def __len__(self):
		return list(self.named_tensors.values())[0].size(0)
