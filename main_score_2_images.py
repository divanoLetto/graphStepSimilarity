from Printing_and_plotting.score_2_excel import score_2_excel
import pathlib


def main():
    base_path = str(pathlib.Path(__file__).parent)
    excel_score_path = base_path + "/Datasets/DS_4/results/simgnn/256/simgnn_score.xlsx"
    high_max = [False]
    image_dir_path = base_path + "/images/models_images/parts/"
    save_name = 'simgnn_retrieval.xlsx'
    score_2_excel(excel_score_path, high_max, image_dir_path, save_name=save_name)


if __name__ == "__main__":
    main()