from rossmann_store_sales.config import OLDER_VERSIONS, TRAINED_MODEL_DIR
from rossmann_store_sales.training import move_older_version
import os

def test_move_older_version():
    move_older_version()

    # check if the folder was created
    assert os.path.isdir(OLDER_VERSIONS) == True

    # check if there are elements inside of it
    items_to_move = os.listdir(OLDER_VERSIONS)
    assert len(items_to_move) > 0

    # check if the elements were removed from the other folder
    items_to_keep = os.listdir(TRAINED_MODEL_DIR)
    assert len(items_to_keep) == 2

