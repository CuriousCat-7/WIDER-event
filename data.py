import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import os
import os.path

def default_loader(path):
	return Image.open(path).convert('RGB')

def default_flist_reader(flist):
	"""
	flist format: impath label\nimpath label\n ...(same to caffe's filelist)
	"""
	imlist = []
	with open(flist, 'r') as rf:
		for line in rf.readlines():
			spline = line.strip().split()
			if len(spline) == 3:
			    impath = spline[0]+' '+spline[1]
			    imlabel = spline[2]
			else:
			    impath, imlabel = spline
			imlist.append( (impath, int(imlabel)) )

	return imlist

class ImageFilelist(data.Dataset):
	def __init__(self, root, flist, transform=None, target_transform=None,
			flist_reader=default_flist_reader, loader=default_loader):
		self.root   = root
		self.imlist = flist_reader(flist)
		self.transform = transform
		self.target_transform = target_transform
		self.loader = loader

	def __getitem__(self, index):
		impath, target = self.imlist[index]
		img = self.loader(os.path.join(self.root,impath))
		if self.transform is not None:
			img = self.transform(img)
		if self.target_transform is not None:
			target = self.target_transform(target)

		return img, target

	def __len__(self):
		return len(self.imlist)

transform_train = transforms.Compose([
    transforms.Resize([300,300]),
    transforms.RandomCrop([224,224]),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
])

transform_test = transforms.Compose([
    transforms.Resize([300,300]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]), 
])
trainset = lambda root, flist: ImageFilelist(root, flist, transform=transform_train)
testset = lambda root, flist: ImageFilelist(root, flist, transform=transform_test)

