import torchvision.transforms as transforms


class resizeNormalize(object):
    def __init__(self, size, interpolation=Image.LANCZOS,is_test=False):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()
        self.is_test = is_test

    def __call__(self, img):
        w,h = self.size
        w0 = img.size[0]
        h0 = img.size[1]
        if w<=(w0/h0*h):
            img = img.resize(self.size, self.interpolation)
            img = self.toTensor(img)
            img.sub_(0.5).div_(0.5)
        else:
            w_real = int(w0/h0*h)
            img = img.resize((w_real,h), self.interpolation)
            img = self.toTensor(img)
            img.sub_(0.5).div_(0.5)
            start = random.randint(0,w-w_real-1)
            if self.is_test:
                start = 5
                w+=10
            tmp = torch.zeros([img.shape[0], h, w])+0.5
            tmp[:,:,start:start+w_real] = img
            img = tmp
        return img