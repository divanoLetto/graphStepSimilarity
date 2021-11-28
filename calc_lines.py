import os


def calc_lines_files():
    print("Number of lines x model:")
    model_dir_path = "C:/Users/Computer/PycharmProjects/graphStepSimilarity/models/"
    for filename in os.listdir(model_dir_path):
       with open(os.path.join(model_dir_path, filename), 'r') as f: # open in readonly mode
           num_lines = sum(1 for line in f)
           print(str(filename) + " : " + str(num_lines))


calc_lines_files()
