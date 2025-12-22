import torch
import torch.nn.functional as F


def l1(x, y, mask=None):
	if mask is None:
		mask = torch.ones_like(x)
	return mask * torch.abs(x - y)

def l2(x, y, mask=None):
	if mask is None:
		mask = torch.ones_like(x)
	return mask * (x - y) ** 2

def huber(x, y, c=1.0):
	diff = x - y
	abs_diff = torch.abs(diff)
	l2 = 0.5 * (diff ** 2)
	l1 = c * (abs_diff - 0.5 * c)
	return torch.where(abs_diff < c, l2, l1)

def mean_huber(x, y, mask=None):
	if mask is None:
		mask = torch.ones_like(x)
	return torch.mean(huber(x, y) * mask)

def sum_huber(x, y, mask=None):
	if mask is None:
		mask = torch.ones_like(x)
	return torch.sum(huber(x, y) * mask)

def ZNCC(x, y):
	mean_x = torch.mean(x)
	mean_y = torch.mean(y)
	norm_x = x - mean_x
	norm_y = y - mean_y
	variance_x = torch.sqrt(torch.sum(norm_x ** 2))
	variance_y = torch.sqrt(torch.sum(norm_y ** 2))
	zncc = torch.sum(norm_x * norm_y) / (variance_x * variance_y + 1e-8)
	return 1 - zncc

def SSIM(x, y):
	# x, y: (B, C, H, W)
	C1 = 0.01 ** 2
	C2 = 0.03 ** 2
	mu_x = F.avg_pool2d(x, 3, 1, 1)
	mu_y = F.avg_pool2d(y, 3, 1, 1)
	sigma_x = F.avg_pool2d(x ** 2, 3, 1, 1) - mu_x ** 2
	sigma_y = F.avg_pool2d(y ** 2, 3, 1, 1) - mu_y ** 2
	sigma_xy = F.avg_pool2d(x * y, 3, 1, 1) - mu_x * mu_y
	SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
	SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
	ssim_map = SSIM_n / (SSIM_d + 1e-8)
	return torch.clamp((1 - ssim_map) / 2, 0, 1)

def SSIM_l1(x, y, alpha=0.85):
	ss = SSIM(x, y)
	ll = l1(x, y)
	return alpha * ss + (1 - alpha) * ll

def mean_SSIM(x, y):
	return torch.mean(SSIM(x, y))

def mean_SSIM_l1(x, y):
	return 0.4 * mean_SSIM(x, y) + 0.6 * mean_l1(x, y)

def smoothness(x, y):
	# x, y: (B, C, H, W)
	def gradient_x(img):
		sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=img.dtype, device=img.device).view(1, 1, 3, 3)
		if img.shape[1] == 3:
			sobel_x = sobel_x.repeat(3, 1, 1, 1)
		return F.conv2d(img, sobel_x, padding=1, groups=img.shape[1])
	def gradient_y(img):
		sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=img.dtype, device=img.device).view(1, 1, 3, 3)
		if img.shape[1] == 3:
			sobel_y = sobel_y.repeat(3, 1, 1, 1)
		return F.conv2d(img, sobel_y, padding=1, groups=img.shape[1])
	x = x / 255.0
	y = y / 255.0
	disp_gradients_x = gradient_x(x)
	disp_gradients_y = gradient_y(x)
	image_gradients_x = torch.mean(gradient_x(y), dim=1, keepdim=True)
	image_gradients_y = torch.mean(gradient_y(y), dim=1, keepdim=True)
	weights_x = torch.exp(-torch.mean(torch.abs(image_gradients_x), dim=1, keepdim=True))
	weights_y = torch.exp(-torch.mean(torch.abs(image_gradients_y), dim=1, keepdim=True))
	smoothness_x = torch.abs(disp_gradients_x) * weights_x
	smoothness_y = torch.abs(disp_gradients_y) * weights_y
	return torch.mean(smoothness_x + smoothness_y)

def mean_l1(x, y, mask=None):
	if mask is None:
		mask = torch.ones_like(x)
	
	return torch.sum(mask * torch.abs(x - y)) / (torch.sum(mask) + 1e-8)

def mean_l2(x, y, mask=None):
	if mask is None:
		mask = torch.ones_like(x)
	return torch.sum(mask * (x - y) ** 2) / (torch.sum(mask) + 1e-8)

def sum_l1(x, y, mask=None):
	if mask is None:
		mask = torch.ones_like(x)
	return torch.sum(mask * torch.abs(x - y))

def sum_l2(x, y, mask=None):
	if mask is None:
		mask = torch.ones_like(x)
	return torch.sum(mask * (x - y) ** 2)

def sign_and_elementwise(x, y):
	element_wise_sign = torch.sigmoid(10 * (torch.sign(x) * torch.sign(y)))
	return torch.mean(torch.sigmoid(element_wise_sign))

def cos_similarity(x, y, normalize=False):
	if normalize:
		x = F.normalize(x, p=2, dim=-1)
		y = F.normalize(y, p=2, dim=-1)
	return torch.sum(x * y)

def sobel_edges(img):
	channel = img.shape[1]
	kerx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=img.dtype, device=img.device).view(1, 1, 3, 3)
	kery = torch.tensor([[-1, -2, 1], [0, 0, 0], [1, 2, 1]], dtype=img.dtype, device=img.device).view(1, 1, 3, 3)
	kerx = kerx.repeat(channel, 1, 1, 1)
	kery = kery.repeat(channel, 1, 1, 1)
	gx = F.conv2d(img, kerx, padding=1, groups=channel)
	gy = F.conv2d(img, kery, padding=1, groups=channel)
	return torch.cat([gx, gy], dim=1)

def sobel_gradient_loss(x, y, mask=None):
	if mask is None:
		mask = torch.ones_like(x)
	g1 = sobel_edges(x)
	g2 = sobel_edges(y)
	return torch.sum(mask * torch.sum(torch.abs(g1 - g2), dim=1, keepdim=True)) / (torch.sum(mask) + 1e-8)

SUPERVISED_LOSS = {
	'mean_l1': mean_l1,
	'sum_l1': sum_l1,
	'mean_l2': mean_l2,
	'sum_l2': sum_l2,
	'smoothness': smoothness,
	'SSIM': SSIM,
	'SSIM_l1': SSIM_l1,
	'mean_SSIM': mean_SSIM,
	'mean_SSIM_l1': mean_SSIM_l1,
	'ZNCC': ZNCC,
	'cos_similarity': cos_similarity,
	'huber': huber,
	'mean_huber': mean_huber,
	'sum_huber': sum_huber,
	'sobel_gradient': sobel_gradient_loss
}

ALL_LOSSES = dict(SUPERVISED_LOSS)

def get_supervised_loss(name, x, y, mask=None):
	if name not in ALL_LOSSES.keys():
		print('Unrecognized loss function, pick one among: {}'.format(ALL_LOSSES.keys()))
		raise Exception('Unknown loss function selected')
	base_loss_function = ALL_LOSSES[name]
	return base_loss_function(x, y, mask)