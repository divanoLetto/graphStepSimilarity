from Printing_and_plotting.score_2_excel import score_2_excel
import pathlib


def main():
    base_path = str(pathlib.Path(__file__).parent)
    excel_score_path = base_path + "/Datasets/Dataset2/results/wasserstein/ww_parts_score.xlsx"
    high_max = [False]
    image_dir_path = base_path + "/images/models_images/parts/"
    save_name = 'ww_retrieval_score.xlsx'
    score_2_excel(excel_score_path, high_max, image_dir_path, save_name=save_name)


if __name__ == "__main__":
    main()