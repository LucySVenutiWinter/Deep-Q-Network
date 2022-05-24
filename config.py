try:
    if torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"
except:
    DEVICE = "cpu"

#Comment out if you want to use CUDA
DEVICE = "cpu"

BATCH_SIZE = 32

#Number of frames to stack together for network input
STEPS = 4

#Size of discrete action space
ACT_SPACE = 3

#Directories to store checkpoints and logs in
CHECKPOINT = "checkpoints"
LOG = "logs"
