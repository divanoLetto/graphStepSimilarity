import pathlib
import pandas as pd


def main():
    base_path = str(pathlib.Path(__file__).parent)
    ground_truth_path = base_path + "/Datasets/DS_4/results/wasserstein/ww_components_score.xlsx"
    excel2_path = base_path + "/Datasets/DS_4/results/simgnn/slim_256/simgnn_components_score.xlsx"
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

    num_item_1 = len(double_hash1.keys())
    num_item_2 = len(double_hash2.keys())
    if num_item_1 != num_item_2:
        raise Exception("Dimentions of excel files not matching")

    matrix_retrival1 = {}
    for a_key, row in double_hash1.items():
        listt = row.items()
        listt_sorted = sorted(listt, key=lambda x: x[1])
        row_names_sorted = [l[0] for l in listt_sorted]
        matrix_retrival1[a_key] = row_names_sorted

    matrix_retrival2 = {}
    for a_key, row in double_hash2.items():
        listt = row.items()
        listt_sorted = sorted(listt, key=lambda x: x[1])
        row_names_sorted = [l[0] for l in listt_sorted]
        matrix_retrival2[a_key] = row_names_sorted

    target_matrix_retrieval = {}
    for key, row in matrix_retrival1.items():
        target_matrix_retrieval[key] = row[:num_corrected_retrieval]

    mAP = 0
    for key in target_matrix_retrieval.keys():
        row_predicted = matrix_retrival1[key]
        row_ground_truth = target_matrix_retrieval[key]
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
    mAP = mAP / num_item_1
    print("Perfect mAP " +str(mAP))

    mAP = 0
    for key in target_matrix_retrieval.keys():
        row_predicted = matrix_retrival2[key]
        row_ground_truth = target_matrix_retrieval[key]
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
    mAP = mAP / num_item_1
    print("mAP " + str(mAP))


if __name__ == "__main__":
    main()