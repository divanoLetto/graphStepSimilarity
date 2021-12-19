import pandas as pd
from pandas import DataFrame
import math
import numpy as np
import xlsxwriter
import os


def highlight_max_(s):
    bolean_row = np.copy(s)
    max_values = s.nlargest(3).values
    for i in range(len(s)):
        if s[i] in max_values:
            bolean_row[i] = True
        else:
            bolean_row[i] = False
    return bolean_row


def my_max(s, id):
    flat = s.flatten()
    flat.sort()
    return flat[-id]


def my_min(s, id):
    flat = s.flatten()
    flat.sort()
    return flat[id]


def highlight_max(df):
    prop_matrix = np.empty(df.shape, dtype=object)
    attr_3 = 'background-color: {}'.format("yellow")
    attr_2 = 'background-color: {}'.format("orange")
    attr_1 = 'background-color: {}'.format("red")
    for i, (index_i, row) in enumerate(df.iterrows()):
        for index_j, cell in enumerate(row):
            if len(row) >= 3 and cell == my_max(row.values, 3):
                prop_matrix[i, index_j] = attr_3
            elif len(row) >= 2 and cell == my_max(row.values, 2):
                prop_matrix[i, index_j] = attr_2
            elif cell == my_max(row.values, 1):
                prop_matrix[i, index_j] = attr_1
    return pd.DataFrame(prop_matrix, index=df.index, columns=df.columns)


def highlight_min(df):
    prop_matrix = np.empty(df.shape, dtype=object)
    attr_3 = 'background-color: {}'.format("yellow")
    attr_2 = 'background-color: {}'.format("orange")
    attr_1 = 'background-color: {}'.format("red")
    for i, (index_i, row) in enumerate(df.iterrows()):
        for index_j, cell in enumerate(row):
            if len(row) >= 3 and cell == my_min(row.values, 2):
                prop_matrix[i, index_j] = attr_3
            elif len(row) >= 2 and cell == my_min(row.values, 1):
                prop_matrix[i, index_j] = attr_2
            elif cell == my_min(row.values, 0):
                prop_matrix[i, index_j] = attr_1
    return pd.DataFrame(prop_matrix, index=df.index, columns=df.columns)


def write_dataFrame(df_dict, file_name, high_max, base_path=""):
    spaces = 1
    file_name = base_path + file_name
    writer = pd.ExcelWriter(file_name, engine="openpyxl")
    row = 0
    for idx, (dataframe_key, dataframe) in enumerate(df_dict.items()):
        header = DataFrame([dataframe_key])
        header.to_excel(writer, startrow=row, startcol=0, header=None, index=None)
        row = row + 1
        tmp_list = []
        count = 0
        for col in dataframe.columns:
            if col in tmp_list:
                col = col + "_" + str(count)
                count = count + 1
            tmp_list.append(col)
        dataframe.columns = tmp_list
        dataframe.index = tmp_list

        if high_max[idx]:
            dataframe.style.apply(highlight_max, axis=None).to_excel(writer, startrow=row, startcol=0)
        else:
            dataframe.style.apply(highlight_min, axis=None).to_excel(writer, startrow=row, startcol=0)
        # dataframe.to_excel(writer, sheet_name=sheets, startrow=row, startcol=0)
        row = row + len(dataframe.index) + spaces + 1
    writer.save()


def write_dataFrame_ordered_by_name(df_dict, file_name, names, base_path=""):
    spaces = 1
    file_name = base_path + file_name
    writer = pd.ExcelWriter(file_name, engine='openpyxl')
    row = 0
    num_models = len(names)
    for dataframe_key, dataframe in df_dict.items():
        header = pd.DataFrame([dataframe_key])
        header.to_excel(writer, startrow=row, startcol=0, header=None, index=None)
        row = row + 1

        score_matrix = dataframe.to_numpy()
        retrieval_matrix = []
        for matrix_row in score_matrix:
            values_names = []
            for j in range(num_models):
                values_names.append((matrix_row[j], names[j]))
            values_names.sort(key=lambda tup: tup[0])
            tmp = []
            for j in range(num_models):
                tmp.append(values_names[j][1])
            retrieval_matrix.append(tmp)

        index = [n for n in names]
        df = pd.DataFrame(retrieval_matrix, index= index)
        df.columns = [n for n in range(num_models)]
        df.to_excel(writer, startrow=row, startcol=0)
        row = row + len(names) + spaces + 1

    writer.save()


def write_dataFrame_by_images(df_dict, file_name, names, base_path, image_dir_path, by_min, scale_x=0.07, scale_y=0.07):
    spaces = 1
    workbook = xlsxwriter.Workbook(base_path + file_name)
    worksheet = workbook.add_worksheet()
    row = 0
    num_models = len(names)
    tmp = []
    for n in names:
        tmp.append(n.replace("-","_"))
    names = tmp

    for k, (dataframe_key, dataframe) in enumerate(df_dict.items()):
        worksheet.write(row, 0, dataframe_key)
        row = row + 2

        worksheet.write_row('C'+str(row), names)
        worksheet.write_column('A'+str(row+2), names)
        score_matrix = dataframe.to_numpy()
        retrieval_matrix = []
        for j in range(num_models):
            filename = image_dir_path + names[j]
            for f in os.listdir(image_dir_path):
                f_name, f_ext = os.path.splitext(f)
                if f_name == names[j]:
                    filename = filename + f_ext
                    worksheet.insert_image(row=row, col=j + 2, filename=filename, options={'x_scale': scale_x, 'y_scale': scale_y})
                    break
        for i, matrix_row in enumerate(score_matrix):
            worksheet.set_row(row + i + 1, 25)

            filename = image_dir_path + names[i]
            for f in os.listdir(image_dir_path):
                f_name, f_ext = os.path.splitext(f)
                if f_name == names[i]:
                    filename = filename + f_ext
                    worksheet.insert_image(row=row + i + 1, col=1, filename=filename, options={'x_scale': scale_x, 'y_scale': scale_y})
                    break
            values_names = []
            for j in range(num_models):
                values_names.append((matrix_row[j], names[j]))
            if by_min[k]:
                values_names.sort(key=lambda tup: tup[0], reverse=True)
            else:
                values_names.sort(key=lambda tup: tup[0])
            tmp = []
            for j in range(num_models):
                tmp.append(values_names[j][1])

                filename = image_dir_path + values_names[j][1]
                for f in os.listdir(image_dir_path):
                    f_name, f_ext = os.path.splitext(f)
                    if f_name == values_names[j][1]:
                        filename = filename + f_ext
                        worksheet.insert_image(row=row + i + 1, col=j + 2, filename=filename, options={'x_scale': scale_x, 'y_scale': scale_y})
                        break
            retrieval_matrix.append(tmp)

        worksheet.set_row(row, 32)
        row = row + len(names) + spaces + 2
        worksheet.set_row(row-1, 25)
        worksheet.set_row(row, 25)

    # worksheet.set_row(num_models, 25)
    # worksheet.set_row(num_models + 1, 25)
    worksheet.set_column(1, num_models + 2, 10)
    worksheet.set_column(1, 1, 12)

    workbook.close()



