import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from glob import glob
import numpy as np 
import os
from PIL import Image
import cv2 as cv
import shutil
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from os import path 
import random
import pickle
import torch
import torchvision.transforms.v2 as transforms
from torchvision.transforms.v2 import functional as F
import torchvision.transforms as T
from natsort import natsorted
from collections import defaultdict

root_folder = "../Retina data/Images Only"
main_folder = "../Retina data" #/Images Only"
sub_folders = os.listdir(main_folder)
sub_folders = [path.join(main_folder, sf) for sf in sub_folders if path.isdir(path.join(main_folder, sf))]

sub_root = os.listdir(root_folder)
sub_root = [path.join(root_folder, sf) for sf in sub_root if path.isdir(path.join(root_folder, sf))]

ann_folder = "../annotation"


im_save_dir = '../processed_images'
ann_save_dir = '../processed_anns'
pickle_dir = '../pickleio'


def change_contrast(img, level):
	factor = (259 * (level + 255)) / (255 * (259 - level))
	def contrast(c):
		return 128 + factor * (c - 128)
	return img.point(contrast)


def smart_crop(src, buffer=100, a=450, b=400):
	ddepth = cv.CV_16S
	scale = 1
	delta = 0
	oh, ow, _ = src.shape
	src = src[:a, b:]

	image = change_contrast(Image.fromarray(src), level=200)
	image = np.array(image.convert('L'))
	

	image = cv.GaussianBlur(image, (3, 3), 0)
	gray = cv.morphologyEx(image, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT,(5,5)))
	
	grad_x = cv.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
	grad_y = cv.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)


	abs_grad_x = cv.convertScaleAbs(grad_x)
	abs_grad_y = cv.convertScaleAbs(grad_y)
	grad = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

	roundedx = (abs_grad_x/255).astype(np.int32)
	x = np.argmax(roundedx, axis=1)

	if (x==0).all():
		x = 0
	else:
		x = min(x[x>0])

	roundedy = (abs_grad_y/255).astype(np.int32)

	y = np.argmax(abs_grad_y, axis=0)
	y_0 = min(y[y>0])
	y_1 = max(y)

	y_0 = max(0, y_0-buffer)
	y_1 = min(image.shape[0], y_1 + buffer)

	img = src[y_0 :y_1, x + int(0.5*buffer):]
	
	y_1 = min(a, y_1)
	c = [y_0, y_1, b + x + int(0.5*buffer), ow]
	return img, c


def bboxconvert(old_image, new_image, old_bbox, cb):
	h0, w0, _ = old_image.shape 
	h, w, _ = new_image.shape
	rh = h/h0
	rw = w/w0 

	
	a,b,c,d = old_bbox
	# a,b,c,d = a*w0, b*h0, c*w0, d*h0

	xtop, ytop, xbot, ybot = cb[2], cb[0], cb[3], cb[1]
	xtop, ytop, xbot, ybot = xtop/w0, ytop/h0, xbot/w0, ybot/h0



	if a<=xbot:
		anew = a - xtop

	elif a>xbot:
		anew = (a-xtop) - (1- xbot)

	if b<=ybot:
		bnew = b - ytop 

	elif b>ybot:
		bnew = (b-ytop) - (1-ybot)

	
	new_bbox = [anew/rw, bnew/rh, c/rw, d/rh]

	# if h<350:
	new_bbox[-1] = 1
	new_bbox[1] = 0.5
	
	return new_bbox



def make_crop(im, labels):
	# image = cv.imread(im)
	image = np.asarray(im)
	img, cropbox = smart_crop(image)

	new_labels = []
	for lb in labels:
		new_labels.append(bboxconvert(image, img, lb, cropbox))
	new_labels = np.array(new_labels)
	new_labels = np.clip(new_labels, 0.1, 0.9)
	# labels[:,:-1] = new_labels
	img = Image.fromarray(img)
	return img, new_labels



def ids(filename):
	'''
	Get person id and test id
	'''
	f, p, t = [], [], []

	for files in filename:
		file, pid, testid = id_per_file(files)
		f.append(file)
		p.append(pid)
		t.append(testid)
	return f, p, t


def id_per_file(fn):
	file = (fn.split('/')[-1]).split('.')[0]
	lr = file.find('L')
	if lr==-1:
		lr = file.find('R')
	pid = file[:lr+1]
	testid = file[:lr+3]
	return file, pid, testid


def boxes_cxcywh_to_xyxy(v):
	boxes = []
	for val in v:
		boxes.append(box_cxcywh_to_xyxy(val))
	return np.stack(boxes)


def box_cxcywh_to_xyxy(v):
	if type(v)==torch.Tensor:
		x_c, y_c, w, h = v.unbind(-1)
	else:
		x_c, y_c, w, h = v#.unbind(-1)
	b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
	return np.stack(b, axis=-1)


def pltbbox(image, bbox=None, cls=None, name='a', clr='r'):

	if type(image)==str:
		image = plt.imread(image)

	if type(image)==torch.Tensor:
		_,h, w = image.shape
		image = image.permute(1,2,0)

	elif isinstance(image, Image.Image):
		w, h = image.size

	if cls is None or cls[0]==2:
		return

	if isinstance(cls, torch.Tensor):
		cls = [i.item() for i in cls]

	if bbox is not None and len(bbox)>0:
		plt.imshow(image, aspect=1)
		bbox = boxes_cxcywh_to_xyxy(bbox)
		for i, box in enumerate(bbox):
			a,b,c,d = box
			plt.gca().add_patch(Rectangle((w*a, h*b), width=w*(c-a), height=h*(d-b), edgecolor=clr, facecolor='none'))
			# if cls is not None:
			plt.annotate(str(cls[i]), (w*a + w*(c-a)/2, h*b + h*(d-b)/2), color=clr, weight='bold', 
					  fontsize=14, ha='center', va='center')
		if name is not None:
			name = path.join(im_save_dir, name)
			plt.gca().set_axis_off()
			plt.savefig(name+'.png', bbox_inches = 'tight', pad_inches = 0)
			plt.close()

			# plt.savefig(name+'.png')
			# plt.close()



def get_annotations():
	ann = natsorted(glob(ann_folder+'/*.txt'))
	return ann


def read_annotations(ann):
	labels = []
	for a in ann:
		# if path.exists(a): 
		label = read_one_annotation(a)
		# else:
		# 	label = make_fake_annotation()
		labels.append(label)
	return labels


def read_one_annotation(file):
	if path.exists(file):
		with open(file, 'r') as f:
			no = f.readlines()
			classes = [int(n[0]) for n in no if n!='\n']
			boxes = [n[1:-1] for n in no if n!='\n']
		boxes = [list(map(float, n.split(' ')[1:])) for n in boxes]
		[b.append(c) for c,b in zip(classes, boxes)]
		boxes = np.array(boxes)
	else:
		boxes = make_fake_annotation()
	return boxes


def make_fake_annotation():
	return torch.tensor([[0.0, 0.0, 0.0, 0.0, 2]])


def write_annotation(ca, cc, fn):
	cp = np.zeros((ca.shape[0], ca.shape[1]+1))
	cp[:,0] = cc
	cp[:,1:] = ca

	cp = np.round(cp, 2)
	
	with open(fn+'.txt', 'w') as f:
		for c in cp:
			f.write(f'{int(c[0])} {c[1]} {c[2]} {c[3]} {c[4]}\n')



def get_images(sf=sub_folders):
	images = [natsorted(glob(path.join(s) + '/*.jpg')) for s in sf]
	images = [a for b in images for a in b]
	images = [i for i in images if i.split('.')[0][-1:]!='M']
	return images


def copy_image_from_one_folder(images):
	for i in images:
		dst = i.replace(i.split('/')[1]+'/', '')
		print(A, i, dst)
		shutil.copy(i, dst)


def copy_images():
	images = []
	for sf in sub_folders:
		images.extend(get_images(sf))
	copy_image_from_one_folder(images)


def image_from_anno(ann_files):
	img_files = [a.replace(ann_folder, main_folder).replace('txt', 'jpg') for a in ann_files]
	return img_files


def anno_from_image(img_files):
	ann_files = [a.replace(main_folder, ann_folder).replace('jpg', 'txt') for a in img_files]
	return ann_files


def save_img(fn, image=None):

	if image is not None:
		plt.imshow(image)
	plt.gca().set_axis_off()
	plt.margins(0,0)
	plt.savefig(fn+'.png', bbox_inches = 'tight', pad_inches = 0)
	plt.close()


def read_images(img_files):
	im = []
	for imf in img_files:
		if not path.exists(imf):
			print(imf, 'does not exist')
			continue
		im.append(read_one_image(imf))
	return im 

def read_one_image(file):
	im = plt.imread(file)
	return im

def save_pickle(im, bx, cx, n, folder=None):
	if folder:
		fn = path.join(folder, n)
	else:
		fn = path.join(pickle_dir, n)
	tensor = {'img':im, 'box':bx, 'label':cx, 'name':n}
	# if not path.exists(fn+'.pkl'):
	# 	os.mkdir(fn)
	if not path.exists(pickle_dir):
		os.mkdir(pickle_dir)
	with open(fn + '.pkl', 'wb') as f:
		pickle.dump(tensor, f)


def load_pickles():
	pick_list = []
	pick = natsorted(glob(pickle_dir + '/*.pkl'))
	for p in pick:
		pick_list.append(load_one_pickle(p))
	return pick_list



def load_one_pickle(fn):
	with open(fn, 'rb') as f:
		pic = pickle.load(f)
	return pic


def operation_and_save(i, an, fnlist, n, plot=True, save=True):
	c = torch.tensor(an[:,-1]).int()
	a = an[:,:-1]
	fname = path.join(ann_save_dir, n)
	for f in fnlist:
		if f == 'crop':
			i, a = make_crop(i, a)

		if f == 'tx':
			i, a = transform(i, a)

	if c[0]==2:
		a = make_fake_annotation()[:,:-1]	

	if plot and c[0]!=2:
		pltbbox(i, a.numpy(), cls=c.numpy(), name=n)
		if save:
			write_annotation(a, c, fname)

	if save:
		save_pickle(i, a.to(torch.float16), c, n)
	


def compose_transformations():
	trans = transforms.Compose([transforms.Resize((256, 576)), 
		transforms.ColorJitter(contrast=0.5),
		transforms.GaussianBlur((7,5), (0.1,3.0)),
		transforms.ToTensor()
	])
	return trans


def permanent_transform(img, bx):
	flip = T.RandomHorizontalFlip(p=1.0)
	img = flip(img)
	# bx[:,::2]=torch.from_numpy(1-(bx.numpy()[:,-2::-2])[:,::-1])
	bx[:,0] = 1-bx[:,0]
	
	return img, bx

def transform(image, labels):
	tx = compose_transformations()
	im, bbox = tx(image, labels)
	bbox = bbox.squeeze(0)
	return im, bbox


def generate_flip_names(names):
	new_names = []
	for name in names:
		new_name = generate_one_flip_name(name)
		new_names.append(new_name)
	return new_names

def generate_one_flip_name(one_name):
	f,_, _ = id_per_file(one_name)
	new_name = (f.replace('_L_', '_Y_')).replace('_R_', '_Z_')
	return new_name 


def read_paths_textfile(fileto):
	names = []
	with open(fileto, 'r') as f:
		for files in f.readlines():
			names.append(files.replace('\n',''))
	return names


def write_paths_textfile(fileto, paths):
	with open(fileto, 'w') as f:
		for path in paths:
			f.write(path+'\n')


def append_textfile(fileto, new_names):
	existing = read_paths_textfile(fileto)
	new_names = ['../pickle/'+nn+'.pkl' for nn in new_names]
	existing.extend(new_names)
	random.shuffle(existing)
	write_paths_textfile(fileto.replace('train', 'example'), existing)




def flippers():
	textfilepath = '../scr/dent/data/train.txt'
	pickle_file_path = '../scr/'
	filenames = read_paths_textfile(textfilepath)
	new_names = generate_flip_names(filenames)
	for file, new_name in zip(filenames, new_names):
		# new_name = generate_one_flip_name(file)
		file = file.replace('../', pickle_file_path)
		print('Old name:', file, 'New Name:', new_name)
		pick = load_one_pickle(file)
		image, box, label, name = pick['img'], pick['box'], pick['label'], pick['name']
		image = T.ToPILImage()(image)
		fim, fbx = permanent_transform(image, box.clone())

		# pltbbox(image, bbox=box, cls=label, name=id_per_file(file)[0], clr='r')
		pltbbox(fim, bbox=fbx, cls=label, name=new_name, clr='r')
		fim = T.ToTensor()(fim)
		save_pickle(fim, fbx, label, new_name, folder=pickle_file_path+pickle_dir)

	append_textfile(textfilepath, new_names)
		


def make_images_and_annos(imf, anf, f):
	for i, a, n in zip(imf, anf, f):
		i = read_one_image(i)
		a = read_one_annotation(a)
		print(n)
		operation_and_save(i, a, ['crop', 'tx'], n=n)

def getsubs(folder):
	subs = os.listdir(folder)
	subs = [path.join(folder, sf) for sf in subs if path.isdir(path.join(folder, sf))]
	return subs


def getstats(folder):
	img_files = get_images(folder)
	f, p, t = ids(img_files)
	print('Total data in directory {} = {}'.format(folder, len(f)))
	uniquep = list(set(p))
	uniquet = list(set(t))
	peopleonly = set([i.split('_')[0] for i in uniquep])

	print('There are {} unique patients'.format(len(peopleonly)))
	print('There are {} unique eyes and {} OCT tests were done by them'.format(len(uniquep), len(uniquet)))
	pleft = [i for i in uniquep if 'L' in i]
	pright = [i for i in uniquep if 'R' in i]

	tleft = [i for i in uniquet if 'L' in i]
	tright = [i for i in uniquet if 'R' in i]

	

	print('Among the {} unique eyes, there were {} left and {} right OCTs.'.format(len(uniquep), len(pleft), len(pright)))
	print('Among the {} unique OCT tests (including re-visits), there were {} left and {} right OCTs.'.format(len(uniquet), len(tleft), len(tright)))

	testpatient = {} # number of tests per patient eye
	for i in uniquep:
		testpatient[i] = []
		for j in uniquet:
			if ''.join(i.split('_')) ==''.join(j.split('_')[:-1]):
				testpatient[i].append(j)

	testcountpatient = {k:len(v) for k, v in testpatient.items()} # count of number of test per patient eye

	patienttest = defaultdict(list) # what patients are there with n numnber of tests? 
	for key, value in testcountpatient.items():
		patienttest[value].append(key)

	patientcounttest = {k:len(v) for k, v in patienttest.items()} # how many patients with n tests? 

	print('Count of people (cp) that performed n number of tests [n:cp] = ', patientcounttest)
	

	


if __name__ == '__main__':

	img_files = get_images()
	ann_files = anno_from_image(img_files)

	f, p, t = ids(img_files)

	a = getsubs(main_folder)
	b = getsubs(root_folder)
	a.extend(b)
	

	getstats(a)
	

	# make_images_and_annos(img_files, ann_files, f)


	

		
	
	