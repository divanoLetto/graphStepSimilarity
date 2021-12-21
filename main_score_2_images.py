from Printing_and_plotting.score_2_excel import score_2_excel
import pathlib


def main():
    base_path = str(pathlib.Path(__file__).parent)
    excel_score_path = base_path + "/results/simgnn/simgnn_match_all.xlsx"
    high_max = [False]
    image_dir_path = base_path + "/images/models_images/parts/"
    save_name = 'parts_retrieval_match_.xlsx'
    score_2_excel(excel_score_path, high_max, image_dir_path, save_name=save_name)


if __name__ == "__main__":
    main()