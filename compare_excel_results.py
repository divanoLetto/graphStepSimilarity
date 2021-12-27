import pathlib
import pandas as pd


def main():
    base_path = str(pathlib.Path(__file__).parent)
    ground_truth_path = base_path + "/Datasets/DS_4/results/wasserstein/ww_parts_score.xlsx"
    excel2_path = base_path + "/Datasets/DS_4/results/simgnn/256/simgnn_score.xlsx"
    num_corrected_retrieval = 8

    df1 = pd.read_excel(ground_truth_path)
    double_hash1 = {}
    for i in range(len(df1.index.values) - 1):
        for j in range(len(df1.columns.values) - 1):
            i_name = df1.iloc[i + 1, 0]
            j_name = df1.iloc[0, j + 1]
            dist = df1.iloc[i + 1, j + 1]
            if i_name not in double_hash1.keys():
                double_hash1[i_name] = {}
            double_hash1[i_name][j_name] = dist

    df2 = pd.read_excel(excel2_path)
    double_hash2 = {}
    for i in range(len(df2.index.values) - 1):
        for j in range(len(df2.columns.values) - 1):
            i_name = df2.iloc[i + 1, 0]
            j_name = df2.iloc[0, j + 1]
            dist = df2.iloc[i + 1, j + 1]
            if i_name not in double_hash2.keys():
                double_hash2[i_name] = {}
            double_hash2[i_name][j_name] = dist

    if len(df1.keys()) != len(df2.keys()):
        raise Exception("Dimentions of excel files not matching")

    names = list(df1.iloc[0, 1:])

    score_matrix1 = df1.iloc[1:, 1:].to_numpy()
    matrix_retrival1 = []
    for i, matrix_row in enumerate(score_matrix1):
        row_names = zip(matrix_row, names)
        row_names_sorted = sorted(row_names, key=lambda x: x[0])
        matrix_retrival1.append([t[1] for t in row_names_sorted])
    score_matrix2 = df2.iloc[1:, 1:].to_numpy()
    matrix_retrival2 = []
    for i, matrix_row in enumerate(score_matrix2):
        row_names = zip(matrix_row, names)
        row_names_sorted = sorted(row_names, key=lambda x: x[0])
        matrix_retrival2.append([t[1] for t in row_names_sorted])

    true_matrix_retrieval1 = []
    for row in matrix_retrival1:
        true_matrix_retrieval1.append(row[:num_corrected_retrieval])

    mAP = 0
    for i in range(len(score_matrix1[0])):
        row_predicted = matrix_retrival1[i]
        row_ground_truth = true_matrix_retrieval1[i]
        ap = 0
        count = 0
        num_found = 0
        for element in row_predicted:
            count += 1
            if element in row_ground_truth:
                num_found += 1
                ap += num_found / count
        ap = ap/num_corrected_retrieval
        mAP += ap
    mAP = mAP / len(score_matrix1[0])
    print("Perfect mAP " +str(mAP))

    mAP = 0
    for i in range(len(score_matrix1[0])):
        row_predicted = matrix_retrival2[i]
        row_ground_truth = true_matrix_retrieval1[i]
        ap = 0
        count = 0
        num_found = 0
        for element in row_predicted:
            count += 1
            if element in row_ground_truth:
                num_found += 1
                ap += num_found/count
        ap = ap/num_corrected_retrieval
        mAP += ap
    mAP = mAP/len(score_matrix1[0])
    print("mAP " +str(mAP))


if __name__ == "__main__":
    main()