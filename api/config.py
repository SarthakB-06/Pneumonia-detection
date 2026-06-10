import os 
from pathlib import Path



BASE_DIR = Path(__file__).resolve().parent.parent


RAW_DATA_DIR = BASE_DIR/"data"/"raw"
TRAIN_DATA = RAW_DATA_DIR/"train"
VAL_DIR = RAW_DATA_DIR/"val"
TEST_DIR = RAW_DATA_DIR/"test"

IMG_HEIGHT = 224
IMG_WIDTH = 224
CHANNELS = 3
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)


BATCH_SIZE = 32
RANDOM_SEED = 42