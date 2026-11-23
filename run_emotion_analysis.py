"""Run document-level and corpus-level emotion analysis and visualizations."""

from utils import (
    batch_generate_emotion_plots,
    find_highest_emotional_charge_per_passage,
    find_highest_emotional_climax,
    generate_corpus_level_plots,
    plot_all_emotions_flow,
    plot_corpus_average_emotions,
    top_passages_per_emotion,
)


INPUT_FOLDER = "processed_data"
OUTPUT_FOLDER = "plots/literary_work_emotion_dynamics"
SAMPLE_FILE = (
    f"{INPUT_FOLDER}/SRP18963_Hajduk_Stanko-L2_analysis.json"
)
CORPUS_FOLDER = "data/level2"
CORPUS_SUMMARY_PATH = (
    "processed_data/corpus_global_emotion_report.tsv"
)


if __name__ == "__main__":
    print("=== RUNNING DOCUMENT-LEVEL ANALYSIS ===")
    plot_all_emotions_flow(
        SAMPLE_FILE,
        output_image_folder=OUTPUT_FOLDER,
        smoothing_window=50,
    )

    print("\n=== BATCH GENERATING PLOTS FOR THE CORPUS ===")
    batch_generate_emotion_plots(
        INPUT_FOLDER,
        output_image_folder=OUTPUT_FOLDER,
    )

    print("\n=== RUNNING AGGREGATED CORPUS ANALYSIS ===")
    plot_corpus_average_emotions(
        INPUT_FOLDER,
        resolution=20,
        output_image="plots/corpus_level/corpus_emotion_average.png",
    )
    generate_corpus_level_plots(
        CORPUS_SUMMARY_PATH,
        output_directory="plots/corpus_level",
    )

    print("\n=== EXTRACTING HIGH-INTENSITY SEGMENTS ===")
    find_highest_emotional_charge_per_passage(INPUT_FOLDER, top_n=3)
    top_passages_per_emotion(
        "disgust",
        top_n=10,
        json_folder=INPUT_FOLDER,
    )
    find_highest_emotional_climax(
        INPUT_FOLDER,
        xml_folder=CORPUS_FOLDER,
        top_n=5,
    )

    print("complete")
