# config.py for choose data set
import os.path

home = os.path.expanduser("~")
ddir = os.path.join(home,"data/VOCdevkit/")

VOCroot = ddir

# default batch size
BATCHES = 32
# data reshuffled at every epoch
SHUFFLE = True
# number of subprocesses to use for data loading
WORKERS = 4

