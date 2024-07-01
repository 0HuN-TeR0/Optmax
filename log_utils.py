import os
from datetime import datetime


def create_log(user_id: str, ts: datetime, data: dict) -> None:

    path = f"./datas/{user_id}"
    log_file = f"{path}/recommendatation.log"
    open_mode = "a"

    if not os.path.exists(path):
        os.mkdir(path)
        if not os.path.isfile(log_file):
            open_mode = "w"

    with open(log_file, open_mode) as file:
        file.writelines(f"{str(ts)}\t{data}")
