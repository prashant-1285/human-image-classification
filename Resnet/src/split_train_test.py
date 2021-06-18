import shutil
import os
import numpy as np
import argparse
import splitfolders

def main(args):
     splitfolders.ratio(args.input_dir, args.output_dir, seed=1337, ratio=(.7, .2, .1), group_prefix=None) # default values

if __name__ == "__main__":
 
     parser = argparse.ArgumentParser()
     parser.add_argument('--input_dir', type=str,
                         help='path to the directory where the images will be read from')
     parser.add_argument('--output_dir', type=str,
                         help='path to the directory where the train, test and validation images will be stored')
     args = parser.parse_args()
     main(args)