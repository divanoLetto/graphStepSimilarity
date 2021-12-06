from Printing_and_plotting.score_2_excel import score_2_excel


def main():
    excel_score_path = "C:/Users/Computer/PycharmProjects/graphStepSimilarity/results/wassertein/parts_score_match.xlsx"
    high_max = [False]
    image_dir_path = "C:/Users/Computer/PycharmProjects/graphStepSimilarity/images/models_images/parts/"
    score_2_excel(excel_score_path, high_max, image_dir_path)


if __name__ == "__main__":
    main()