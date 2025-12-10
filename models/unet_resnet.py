import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class DoubleConv(nn.Module):
	"""(conv => BN => ReLU) * 2"""
	def __init__(self, in_ch, out_ch):
		super().__init__()
		self.double_conv = nn.Sequential(
			nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_ch),
			nn.ReLU(inplace=True),
		)

	def forward(self, x):
		return self.double_conv(x)


class UpBlock(nn.Module):
	def __init__(self, in_ch, skip_ch, out_ch):
		super().__init__()
		self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
		self.conv = DoubleConv(in_ch + skip_ch, out_ch)
	

	def forward(self, x, skip):
		x = self.up(x)
		# In case of mismatch due to rounding, pad
		if x.size()[-2:] != skip.size()[-2:]:
			x = F.interpolate(x, size=skip.size()[-2:], mode='bilinear', align_corners=True)
		x = torch.cat([skip, x], dim=1)
		return self.conv(x)


class UNetResNet50(nn.Module):
	"""UNet with a ResNet-50 encoder (backbone from torchvision).

	Notes:
	- The encoder layers are taken from a standard ResNet-50: conv1, layer1..layer4
	- Decoder upsamples and concatenates skip connections from earlier stages.
	- Returns raw logits (no sigmoid/softmax). Use appropriate loss (BCEWithLogitsLoss, CrossEntropyLoss).
	"""

	def __init__(self, num_classes=1, pretrained=True):
		super().__init__()
		self.num_classes = num_classes


		resnet = models.resnet50(pretrained=pretrained)

		# Encoder (ResNet-50) layers
		self.encoder_conv1 = nn.Sequential(
			resnet.conv1,
			resnet.bn1,
			resnet.relu,
		)
		self.encoder_maxpool = resnet.maxpool
		self.encoder_layer1 = resnet.layer1
		self.encoder_layer2 = resnet.layer2
		self.encoder_layer3 = resnet.layer3
		self.encoder_layer4 = resnet.layer4

		# Decoder / upsampling blocks
		# Channel sizes from ResNet-50: conv1->64, layer1->256, layer2->512, layer3->1024, layer4->2048
		self.center = DoubleConv(2048, 2048)

		self.up4 = UpBlock(in_ch=2048, skip_ch=1024, out_ch=1024)
		self.up3 = UpBlock(in_ch=1024, skip_ch=512, out_ch=512)
		self.up2 = UpBlock(in_ch=512, skip_ch=256, out_ch=256)
		# After conv1 the channel is 64, but conv1 is before maxpool. We'll use conv1 output as skip.
		self.up1 = UpBlock(in_ch=256, skip_ch=64, out_ch=128)

		# Final up to original resolution (optional extra upsample)
		self.up0 = nn.Sequential(
			nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
			DoubleConv(128, 64),
		)

		self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

		self._init_weights()

	def _init_weights(self):
		# initialize decoder weights
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					nn.init.zeros_(m.bias)
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.ones_(m.weight)
				nn.init.zeros_(m.bias)

	def forward(self, x):
		# Encoder
		x0 = self.encoder_conv1(x)        # -> [B, 64, H/2, W/2]
		x1 = self.encoder_maxpool(x0)     # -> [B, 64, H/4, W/4]
		x1 = self.encoder_layer1(x1)      # -> [B, 256, H/4, W/4]
		x2 = self.encoder_layer2(x1)      # -> [B, 512, H/8, W/8]
		x3 = self.encoder_layer3(x2)      # -> [B, 1024, H/16, W/16]
		x4 = self.encoder_layer4(x3)      # -> [B, 2048, H/32, W/32]

		# Center
		c = self.center(x4)

		# Decoder with skip connections
		d4 = self.up4(c, x3)   # -> [B, 1024, H/16, W/16]
		d3 = self.up3(d4, x2)  # -> [B, 512, H/8, W/8]
		d2 = self.up2(d3, x1)  # -> [B, 256, H/4, W/4]
		d1 = self.up1(d2, x0)  # -> [B, 128, H/2, W/2]
		d0 = self.up0(d1)      # -> [B, 64, H, W]

		logits = self.final_conv(d0)
		return logits


def get_unet_resnet50(num_classes=2, pretrained=True):
	return UNetResNet50(num_classes=num_classes, pretrained=pretrained)

