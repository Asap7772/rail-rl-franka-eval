import torch
import torch.nn as nn
import torch.nn.functional as F
from rlkit.policies.base import DatasetPolicy, ExplorationPolicy, Policy
from railrl.torch.core import eval_np
from railrl.torch.distributions import TanhNormal
import railrl.torch.pytorch_util as ptu
from rlkit.torch.networks import Mlp
from functools import reduce
import dill
import operator
from rlkit.torch import resnet

"""
From Dopamine (architechture)
3 convolution layers and 2 feedforward fully connected layers

conv1 = nn.Conv2d(obs_dim, 32, (8, 8), stride=4)
conv2 = nn.Conv2d(32, 64, (4, 4), stride=2)
conv3 = nn.Conv2d(64, 64, (38, 3), stride=1)
"""
LOG_SIG_MAX = 2
LOG_SIG_MIN = -2
LOG_MEAN_MAX = 5.0
LOG_MEAN_MIN = -5.0

def identity(x): return x

def batch_flatten(tensor):
	return tensor.view(tensor.shape[0], reduce(operator.mul, tensor.shape[1:]))

class ConvNet(nn.Module):
	"""
		Initialize a CNN with convolution layers and fully connected layers
	"""

	def __init__(self, input_dim, output_dim, conv_sizes, conv_kernel_sizes, conv_strides, fc_sizes, concat_size = 0,
				 conv_activation=F.relu, fcs_activation=F.relu, output_activation=identity, padding=-1, init_w=1e-4,hidden_init=nn.init.xavier_uniform_,):
		super(ConvNet, self).__init__()
		self.convs = []
		self.fcs = []

		self.conv_activation = conv_activation
		self.fcs_activation = fcs_activation
		self.output_activation = output_activation

		curr_input = input_dim
		# convolution layers
		for i in range(len(conv_sizes)):
			next_size = conv_sizes[i]
			curr_kernel = conv_kernel_sizes[i]
			curr_stride = conv_strides[i]
			# padding is an optional parameter
			if padding == -1:
				conv_layer = nn.Conv2d(curr_input, next_size, curr_kernel, stride=curr_stride)
			else:
				conv_layer = nn.Conv2d(curr_input, next_size, curr_kernel, stride=curr_stride, padding=padding)
			hidden_init(conv_layer.weight)
			self.convs.append(conv_layer)
			curr_input = next_size

		curr_input = concat_size

		# hidden feed-forward fully connected layers
		for next_size in fc_sizes:
			linear_layer = nn.Linear(curr_input, next_size)
			hidden_init(linear_layer.weight)
			# hidden_init(linear_layer.bias)
			# linear_layer.weight.data.uniform_(-init_w, init_w)
			linear_layer.bias.data.uniform_(-init_w, init_w)

			self.fcs.append(linear_layer)
			curr_input = next_size

		linear_layer = nn.Linear(curr_input, output_dim)
		# linear_layer.weight.data.uniform_(-init_w, init_w)
		linear_layer.bias.data.uniform_(-init_w, init_w)
		hidden_init(linear_layer.weight)
		# hidden_init(linear_layer.bias)
		self.fcs.append(linear_layer)

		layers = self.convs + self.fcs
		for i in range(len(layers)):
			setattr(self,  'layer'+str(i), layers[i])

	def forward(self, observation, state, actions=None):
		tensor = observation
		# normalize
		# import ipdb; ipdb.set_trace()
		# tensor = torch.div(tensor, 255.)

		for layer in self.convs:
			tensor = layer(tensor)
			tensor = self.conv_activation(tensor)

		tensor = batch_flatten(tensor)
		if state is not None:
			tensor = torch.cat((tensor, state),1)
		if actions is not None:
			tensor = torch.cat((tensor, actions),1)

		i = 0
		for layer in self.fcs:
			tensor = layer(tensor)
			if i == len(self.fcs) - 1:
				tensor = self.output_activation(tensor)
			else:
				tensor = self.fcs_activation(tensor)
			i += 1

		return tensor

	def __str__(self):
		return str([layer.weight for layer in self.convs]) + str([layer.weight for layer in self.fcs])

class TanhGaussianConvPolicy(ConvNet, DatasetPolicy):
	def __init__(self, input_dim, output_dim, conv_sizes, conv_kernel_sizes, conv_strides, fc_sizes, concat_size = 0,
				 conv_activation=F.relu, fcs_activation=F.relu, output_activation=identity, padding=-1, init_w=1e-4):
		""" output dim is the size of action dimensional vector"""
		super().__init__(input_dim, output_dim, conv_sizes, conv_kernel_sizes, conv_strides, fc_sizes, concat_size, conv_activation, fcs_activation, output_activation, padding)
		self.last_fc_log_std = nn.Linear(fc_sizes[-1], output_dim) # linear layer for std-deviation
		self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
		self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)

	def get_action(self, obs_np, image, deterministic=False):
		actions = self.get_actions(obs_np, image, deterministic=deterministic)
		return actions[0, :], {}

	def get_actions(self, obs_np, image, deterministic=False):
		# import ipdb; ipdb.set_trace()
		obs_np = obs_np[None] if obs_np != None else None
		return eval_np(self, image[None], obs_np, actions=None, reparameterize=True, deterministic=deterministic, return_log_prob=False)[0]

	def atanh(self, x):
		one_plus_x = (1 + x).clamp(min=1e-6)
		one_minus_x = (1 - x).clamp(min=1e-6)
		return 0.5 * torch.log(one_plus_x / one_minus_x)

	def log_prob(self, observation, state, actions=None):
		raw_actions = self.atanh(actions)
		tensor = observation
		# normalize
		# tensor = torch.div(tensor, 255.)
		for layer in self.convs:
			tensor = layer(tensor)
			tensor = self.conv_activation(tensor)

		tensor = batch_flatten(tensor)
		if state is not None:
			tensor = torch.cat((tensor, state), 1)
		i = 0
		for layer in self.fcs:
			h = tensor
			tensor = layer(tensor)
			if i == len(self.fcs) - 1:
				tensor = self.output_activation(tensor)
			else:
				tensor = self.fcs_activation(tensor)
			i += 1

		mean = tensor
		mean = torch.clamp(mean, LOG_MEAN_MIN, LOG_MEAN_MAX)
		log_std = self.last_fc_log_std(h)
		log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
		std = torch.exp(log_std)

		tanh_normal = TanhNormal(mean, std)
		log_prob = tanh_normal.log_prob(value=actions, pre_tanh_value=raw_actions)
		return log_prob.sum(-1)

	def forward(self, observation, state, actions=None, reparameterize=True, deterministic=False, return_log_prob=False,):
		# import ipdb; ipdb.set_trace()
		tensor = observation
		# normalize
		# tensor = torch.div(tensor, 255.)
		for layer in self.convs:
			tensor = layer(tensor)
			tensor = self.conv_activation(tensor)

		tensor = batch_flatten(tensor)
		if state is not None:
			tensor = torch.cat((tensor, state), 1)
		i = 0
		for layer in self.fcs:
			h = tensor
			tensor = layer(tensor)
			if i == len(self.fcs) - 1:
				tensor = self.output_activation(tensor)
			else:
				tensor = self.fcs_activation(tensor)
			i += 1

		mean = tensor
		log_std = self.last_fc_log_std(h)
		log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
		std = torch.exp(log_std)

		mean = torch.clamp(mean, LOG_MEAN_MIN, LOG_MEAN_MAX)
		tanh_normal = TanhNormal(mean, std)

		log_prob = None
		entropy = None
		mean_action_log_prob = None
		pre_tanh_value = None
		
		if deterministic:
			action = torch.tanh(mean)
		else:
			tanh_normal = TanhNormal(mean, std)
			if return_log_prob:
				if reparameterize is True:
					action, pre_tanh_value = tanh_normal.rsample(
						return_pretanh_value=True
					)
				else:
					action, pre_tanh_value = tanh_normal.sample(
						return_pretanh_value=True
					)
				log_prob = tanh_normal.log_prob(
					action,
					pre_tanh_value=pre_tanh_value
				)
				log_prob = log_prob.sum(dim=1, keepdim=True)
			else:
				if reparameterize is True:
					action = tanh_normal.rsample()
				else:
					action = tanh_normal.sample()

		return (
			action, mean, log_std, log_prob, entropy, std,
			mean_action_log_prob, pre_tanh_value,
		)


class ResidualConvNet(nn.Module):
	"""
	residual network for training
	"""

	def __init__(self, input_dim, output_dim, layers, zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None, \
				 representation_dim=256, state_dim=10, concat_dim=None, block=None):
		super(ResidualConvNet, self).__init__()

		if norm_layer is None:
			norm_layer = torch.nn.BatchNorm2d

		if block is None:
			block = resnet.BasicBlock

		self._norm_layer = norm_layer

		self.inplanes = 64
		self.dilation = 1

		if replace_stride_with_dilation is None:
			# each element in the tuple indicates if we should replace
			# the 2x2 stride with a dilated convolution instead
			replace_stride_with_dilation = [False, False, False]
		if len(replace_stride_with_dilation) != 3:
			raise ValueError("replace_stride_with_dilation should be None "
							 "or a 3-element tuple, got {}".format(replace_stride_with_dilation))

		self.groups = groups
		self.base_width = width_per_group
		self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
							   bias=False)

		self.bn1 = norm_layer(self.inplanes)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		self.layer1 = self._make_layer(block, 64, layers[0])
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
									   dilate=replace_stride_with_dilation[0])
		self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
									   dilate=replace_stride_with_dilation[1])
		self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
									   dilate=replace_stride_with_dilation[2])
		self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

		representation_dim = representation_dim + state_dim
		self.fc = nn.Linear(512 * block.expansion, representation_dim)

		if concat_dim is not None:
			self.action_concat_fc = nn.Linear(representation_dim + concat_dim, 512)
		else:
			self.action_concat_fc = nn.Linear(representation_dim, 512)

		self.intermed_fc = nn.Linear(512, 512)
		self.final_fc = nn.Linear(512, output_dim)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

		# Zero-initialize the last BN in each residual branch,
		# so that the residual branch starts with zeros, and each residual block behaves like an identity.
		# This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
		if zero_init_residual:
			for m in self.modules():
				if isinstance(m, Bottleneck):
					nn.init.constant_(m.bn3.weight, 0)
				elif isinstance(m, BasicBlock):
					nn.init.constant_(m.bn2.weight, 0)

	def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
		norm_layer = self._norm_layer
		downsample = None
		previous_dilation = self.dilation

		if dilate:
			self.dilation *= stride
			stride = 1
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				resnet.conv1x1(self.inplanes, planes * block.expansion, stride),
				norm_layer(planes * block.expansion),
			)

		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
							self.base_width, previous_dilation, norm_layer))
		self.inplanes = planes * block.expansion
		for _ in range(1, blocks):
			layers.append(block(self.inplanes, planes, groups=self.groups,
								base_width=self.base_width, dilation=self.dilation,
								norm_layer=norm_layer))

		return nn.Sequential(*layers)

	def _forward_impl(self, x, state, actions=None):
		# import ipdb; ipdb.set_trace()
		x = self.conv1(x)
		x = self.bn1(x)

		x = self.relu(x)
		x = self.maxpool(x)

		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)

		x = self.avgpool(x)
		x = torch.flatten(x, 1)

		if state is not None:
			x = torch.cat([x, state], 1)
		if actions is not None:
			x = torch.cat([x, actions], 1)

		x = self.action_concat_fc(x)
		x = self.relu(x)
		x = self.intermed_fc(x)
		x = self.relu(x)
		x = self.final_fc(x)
		return x

	def forward(self, x, state, actions=None):
		return self._forward_impl(x, state, actions)
