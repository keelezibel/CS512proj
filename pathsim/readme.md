# Setup
1. Install Python 3, Pandas, numpy

# Usage
1. Run `python list_all_x.py <data file> <save directory>`

   where `<data file>` is the path to the input data file and `<save firectory>` is the path to a directory where you intend to save the outputs.

   For example,
   `python list_all_x.py '../train_top_3000author.txt' 'train_data'`
   and
   `python list_all_x.py '../test_top_3000author.txt' 'test_data'`

2. Run `python pathsim.py <data file> <directory that contains output files of list_all_x.py for train data> <directory that contains output files of list_all_x.py for test data>`

   For example,
   `python pathsim.py '../train_top_3000author.txt' 'train_data' 'test_data'`
