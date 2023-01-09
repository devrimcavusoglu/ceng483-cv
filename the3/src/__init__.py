from pathlib import Path

SRC_ROOT = Path.cwd()
PROJECT_ROOT = SRC_ROOT.parent
LOG_DIR = PROJECT_ROOT / "checkpoints"
DATA_ROOT = PROJECT_ROOT / "ceng483-f22-hw3-dataset"
TEST_IMAGES_PATH = PROJECT_ROOT / "test_images.txt"
