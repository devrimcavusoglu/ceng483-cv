"""
Implementing simple convolution operations for 1d & 2d
"""

import numpy as np
import scipy.signal as ss
from PIL import Image

from playground.common import load_lena


def conv1d(input_array: np.ndarray, kernel: np.ndarray, stride: int = 1):
	"""
	Implements 1D valid convolution
	"""
	if input_array.ndim > 1 or kernel.ndim > 1:
		raise ValueError

	kernel = np.flip(kernel)

	n = len(input_array)
	k = len(kernel)
	s = n - k + 1
	offset = k // 2
	output_array = np.zeros(s)
	for i in range(offset, n-offset, stride):
		r = input_array[i-offset:i+offset+1]
		output_array[i-offset] = np.dot(r, kernel)
	return output_array


def conv2d(input_array: np.ndarray, kernel: np.ndarray, stride: int = 1):
	"""
	Implements 2D valid convolution 
	"""
	if input_array.ndim != 2 or kernel.ndim != 2:
		raise ValueError
	
	kernel = np.flipud(np.fliplr(kernel))

	m, n = input_array.shape
	k1, k2 = kernel.shape
	assert k1 == k2 and k1 % 2 != 0, "Kernel expected to be a square matrix of an odd size."
	k = k1

	s = m - k + 1
	t = n - k + 1
	offset = k // 2
	output_array = np.zeros((s,t))
	for i in range(offset, m-offset, stride):
		for j in range(offset, n-offset, stride):
			r = input_array[i-offset:i+offset+1, j-offset:j+offset+1]
			output_array[i-offset, j-offset] = np.dot(r.ravel(), kernel.ravel())
	return output_array


def test_conv1d():
	print("Testing conv1d..")
	f = np.array([1,2,3,4,5,6,7,8,9])
	h = np.array([-1, 0, 1])
	g1 = np.convolve(f, h, mode="valid")
	print("numpy ->", g1)
	g2 = conv1d(f, h)
	print("own ->", g2)
	assert all(g1 == g2)
	print("Test PASSED!")


def test_conv2d():
	print("Testing conv2d..")
	f = np.array([1,2,3,4,5,6,7,8,9]).reshape(3, 3)
	h = np.tile([-1, 0, 1], 3).reshape(3,3)
	g1 = ss.convolve2d(f, h, mode="valid")
	print("scipy ->", g1)
	g2 = conv2d(f, h)
	print("own ->", g2)
	assert all(g1 == g2)
	print("Test PASSED!")


def test_lena():
	print("Testing Lena..")
	lena = load_lena()
	pil_img = lena.convert('L')
	img = np.array(pil_img)

	h = np.tile([-1, 0, 1], 3).reshape(3,3)

	g1 = ss.convolve2d(img, h, mode="valid")
	g2 = conv2d(img, h)
	assert (g1 == g2).all()
	print("Test PASSED!")


def experiment_lenna_rgb():
	"""
	WIP: How to fuse channels ?

	ref: http://ai.stanford.edu/~ruzon/compass/color.html
	"""
	lena = load_lena()
	pil_img = lena.convert('RGB')
	img = np.array(pil_img)
	h = np.tile([-1, 0, 1], 3).reshape(3,3)

	out = []
	for c in range(3):
		conv_out = ss.convolve2d(img[:,:,c], h, mode="valid")
		out.append(conv_out)
	out = np.dstack(out)
	print(out.shape)
	# out = np.mean(out, axis=2)
	# print(out.shape)

	filtered_img = Image.fromarray(out, mode="RGB")
	filtered_img.show()

	# g2 = conv2d(img, h)
	# assert (g1 == g2).all()

	# filtered_img = Image.fromarray(g2)
	# filtered_img.show()
	# print("Test PASSED!")


def lena_experiment():
	lena = load_lena()
	pil_img = lena.convert('L')
	img = np.array(pil_img)
	h = (1/9) * np.array([
		[1, 1, 1],
		[1, 1, 1],
		[1, 1, 1]
	])

	g = ss.convolve2d(img, h)
	pil_img.show()
	filtered_img = Image.fromarray(g)
	filtered_img.show()


if __name__ == "__main__":
	test_conv1d()
	test_conv2d()
	test_lena()
	# experiment_lenna_rgb()
	# lena_experiment()
