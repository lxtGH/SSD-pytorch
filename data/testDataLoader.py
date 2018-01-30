import torch.utils.data as data
import torchvision.transforms as transforms
from voc_dataset import VOCDetection
from voc_dataset import AnnotationTransform,detection_collate
import config as cfg
from utils.augmentations import SSDAugmentation
train_sets = [('2007', 'trainval'), ('2012', 'trainval')]  # for two data sets

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

ssd_dim =  300
means = (104, 117, 123)
dataset = VOCDetection(cfg.ddir,train_sets,transform=SSDAugmentation(ssd_dim, means),target_transform=AnnotationTransform())

data_loader = data.DataLoader(dataset,batch_size=cfg.BATCHES,num_workers=cfg.WORKERS,shuffle=True,
                              collate_fn=detection_collate,pin_memory=True)



print (dataset.ids)
print (len(dataset.ids))
print (dataset.image_set)


for iteration in range(1,100):
    batch_iterator = iter(data_loader)

    images, targets = next(batch_iterator)
    # print (targets)
    # print (len(images),len(targets))
    print (type(images))
    print("image size:", images[0].size(), "targe size:", targets[0].size())

    if iteration == 5:
        exit()