import pandas as pd
import os
from Printing_and_plotting.Printing import write_dataFrame_by_images


def score_2_excel(excel_score_path, high_max, image_dir_path, save_name, scale_x=0.03, scale_y=0.03):
    df = pd.read_excel(excel_score_path, skiprows=1)
    dir_path = os.path.dirname(excel_score_path)
    results_path = dir_path + "/"
    part_names = list(df.columns.values)[1:]
    name = list(pd.read_excel(excel_score_path, usecols="A", nrows=0).columns.values)[0]
    df = pd.read_excel(excel_score_path, skiprows=1, usecols=list(range(1, len(part_names)+1)))
    df_dict = {name: df}
    write_dataFrame_by_images(df_dict=df_dict, file_name=save_name, names=part_names,
                              base_path=results_path, image_dir_path=image_dir_path, by_min=high_max, scale_x=scale_x,
                              scale_y=scale_y)
