import numpy as np
import random

# 从Beta分布中选取参数
def get_ab(beta):
  if np.random.random() < 0.5:
    a = np.float32(np.random.beta(beta, 1))
    b = np.float32(np.random.beta(1, beta))
  else:
    a = 1 + np.float32(np.random.beta(1, beta))
    b = -np.float32(np.random.beta(1, beta))
  return a, b

def add(img1, img2, beta):
  a,b = get_ab(beta)
  img1, img2 = img1 * 2 - 1, img2 * 2 - 1
  out = a * img1 + b * img2
  out=(((out + 1) / 2)*255+0.5)//1
  return out.clip(0,255)

def multiply(img1, img2, beta):
  a,b = get_ab(beta)
  img1, img2 = img1 * 2, img2 * 2
  out = (img1 ** a) * (img2.clip(1e-37) ** b)

  return out/2

mixings = [add, multiply]
def invert(img):
  return 1 - img

def screen(img1, img2, beta):
  img1, img2 = invert(img1), invert(img2)
  out = multiply(img1, img2, beta)
  return invert(out)

def overlay(img1, img2, beta):
  case1 = multiply(img1, img2, beta)
  case2 = screen(img1, img2, beta)
  if np.random.random() < 0.5:
    cond = img1 < 0.5
  else:
    cond = img1 > 0.5
  return np.where(cond, case1, case2)

def darken_or_lighten(img1, img2):
  if np.random.random() < 0.5:
    cond = img1 < img2
  else:
    cond = img1 > img2
  return np.where(cond, img1, img2)

def swap_channel(img1, img2):
  channel = np.random.randint(3)
  img1[channel] = img2[channel]
  return img1

def randomMask(alpha):
    data = [i for i in range(640 * 640)]
    idx = random.sample(data, int(640 * 640 * alpha))
    return idx


def RandomMix(tar, obj, alpha):
  idx = randomMask(alpha)
  for k in idx:
    i, j = k // 640, k % 640
    tar[i][j] = obj[i][j]
  return tar


def getIPMix(tar, obj, beta):
  tar = tar * (1 - beta) + beta * obj
  return tar

