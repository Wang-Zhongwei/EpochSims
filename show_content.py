import argparse
import os

import numpy as np
import scipy.stats as stats
import sdf_helper as sh


def show_content(file_path, var_name=None):
    file_name = os.path.basename(file_path)
    file_size = os.path.getsize(file_path)
    print(f"File size of {file_name}: {file_size} bytes")

    data = sh.getdata(file_path, verbose=False)

    # show stats of var_name
    if var_name is not None:
        var_data = data.__getattribute__(var_name).data
        try: 
            print(stats.describe(var_data))
        except Exception as e:
            print(f"Type of {var_name} is {type(var_data)}: {e}")
        return

    # show content of file_name
    for key, val in data.__dict__.items():
        if hasattr(val, "data"):
            if isinstance(val.data, np.ndarray):
                print(key, ": ", val.data.shape, " array")
            elif isinstance(val.data, list) or isinstance(val.data, tuple):
                print(key, ": ", len(val.data), " list or tuple")
            else:
                print(key, ": ", type(val.data))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Display content of an SDF file.")
    parser.add_argument("file_path", type=str, help="Path to the SDF file")
    parser.add_argument("--var_name", type=str, help="Variable name to display statistics for", default=None)
    args = parser.parse_args()
    show_content(args.file_path, args.var_name)