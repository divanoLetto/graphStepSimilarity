import pathlib
import pandas as pd


def main():
    base_path = str(pathlib.Path(__file__).parent)
    ground_truth_path = base_path + "/results/wassertein/parts_score_match.xlsx"
    excel2_path = base_path + "/results/simgnn/simgnn_match.xlsx"
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

    names = list(df1.columns.values)[1:]

    score_matrix1 = df1.to_numpy()
    matrix_retrival1 = []
    for i, matrix_row in enumerate(score_matrix1):
        row_names = zip(matrix_row, names)
        row_names_sorted = sorted(row_names, key=lambda x: x[0])
        matrix_retrival1.append([t[1] for t in row_names_sorted])
    score_matrix2 = df2.to_numpy()
    matrix_retrival2 = []
    for i, matrix_row in enumerate(score_matrix2):
        row_names = zip(matrix_row, names)
        row_names_sorted = sorted(row_names, key=lambda x: x[0])
        matrix_retrival2.append([t[1] for t in row_names_sorted])

    true_matrix_retrieval1 = []
    for row in matrix_retrival1:


if __name__ == "__main__":
    main()