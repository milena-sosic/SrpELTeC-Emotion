from pathlib import Path
from collections import Counter, defaultdict
import atexit
import json
import sys

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FOLDER = Path("processed_data")
REPORT_FOLDER = Path("reports")
POS_REPORT_FOLDER = REPORT_FOLDER / "lemma_pos_statistics"
OUTPUT_FOLDER = REPORT_FOLDER / "lemma_statistics"
TEXT_REPORT_PATH = POS_REPORT_FOLDER / "lemma_pos_statistics.txt"

EMOTIONS = [
    "anger",
    "anticipation",
    "disgust",
    "fear",
    "joy",
    "sadness",
    "surprise",
    "trust",
]

GLOBAL_TOP_N = 50
FILE_TOP_N = 15
EMOTION_TOP_N = 20

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
POS_REPORT_FOLDER.mkdir(parents=True, exist_ok=True)


class TeeOutput:
    """Write console output both to the terminal and to a text report."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            try:
                stream.write(text)
            except UnicodeEncodeError:
                encoding = getattr(stream, "encoding", None) or "utf-8"
                safe_text = text.encode(
                    encoding,
                    errors="replace",
                ).decode(
                    encoding,
                    errors="replace",
                )
                stream.write(safe_text)
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


_original_stdout = sys.stdout
_text_report = TEXT_REPORT_PATH.open("w", encoding="utf-8")
sys.stdout = TeeOutput(_text_report, _original_stdout)


def close_text_report():
    """Restore standard output and close the text report."""
    sys.stdout = _original_stdout
    _text_report.close()


atexit.register(close_text_report)


# ============================================================
# DATA STRUCTURES
# ============================================================

# Raw lemma frequencies
file_word_counts = defaultdict(Counter)
global_word_counts = Counter()

# Cumulative contribution of each lemma to individual emotions
file_emotion_contribution = defaultdict(
    lambda: {
        emotion: defaultdict(float)
        for emotion in EMOTIONS
    }
)

global_emotion_contribution = {
    emotion: defaultdict(float)
    for emotion in EMOTIONS
}

# Cumulative contribution of each lemma across all emotions
file_total_emotional_contribution = defaultdict(
    lambda: defaultdict(float)
)

global_total_emotional_contribution = defaultdict(float)

# Number of occurrences for which a lemma contributes to an emotion
file_emotion_nonzero_counts = defaultdict(
    lambda: {
        emotion: Counter()
        for emotion in EMOTIONS
    }
)

global_emotion_nonzero_counts = {
    emotion: Counter()
    for emotion in EMOTIONS
}

# POS-aware frequency and emotion-weighted statistics.
file_pos_word_counts = defaultdict(Counter)
global_pos_word_counts = Counter()

file_pos_emotion_weights = {
    emotion: defaultdict(lambda: defaultdict(float))
    for emotion in EMOTIONS
}

global_pos_emotion_weights = {
    emotion: defaultdict(float)
    for emotion in EMOTIONS
}


# ============================================================
# HELPERS
# ============================================================

def load_json_files(folder: Path) -> list[dict]:
    """
    Load all JSON files from the input folder.

    Invalid JSON files are skipped with a warning.
    """
    loaded_files = []

    if not folder.exists():
        raise FileNotFoundError(
            f"Input folder does not exist: {folder.resolve()}"
        )

    json_paths = sorted(folder.glob("*.json"))

    if not json_paths:
        raise FileNotFoundError(
            f"No JSON files found in: {folder.resolve()}"
        )

    for json_path in json_paths:
        try:
            data = json.loads(
                json_path.read_text(encoding="utf-8")
            )

            # Preserve the source filename when it is missing
            # from the JSON content.
            data.setdefault("file_name", json_path.name)

            loaded_files.append(data)

        except json.JSONDecodeError as error:
            print(
                f"[WARNING] Invalid JSON skipped: "
                f"{json_path.name}: {error}"
            )

    return loaded_files


def top_values(mapping, n: int = 20) -> list[tuple[str, float]]:
    """
    Return the n highest-valued items from a Counter-like mapping.
    """
    return sorted(
        mapping.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:n]


def normalize_mapping(mapping) -> dict[str, float]:
    """
    Normalize mapping values so that they sum to 1.
    """
    total = float(sum(mapping.values()))

    if total <= 0:
        return {
            key: 0.0
            for key in mapping
        }

    return {
        key: float(value) / total
        for key, value in mapping.items()
    }


def safe_divide(
    numerator: float,
    denominator: float,
) -> float:
    """
    Safely divide two numeric values.
    """
    if denominator == 0:
        return 0.0

    return numerator / denominator


def clean_lemma(value) -> str:
    """
    Normalize a lemma before counting.
    """
    if value is None:
        return ""

    return str(value).strip().lower()


def get_numeric_weight(weights: dict, emotion: str) -> float:
    """
    Read one emotion weight and safely convert it to float.
    """
    try:
        return float(weights.get(emotion, 0.0))
    except (TypeError, ValueError):
        return 0.0


def iter_word_details(document: dict):
    """
    Yield word-detail dictionaries from the hierarchical JSON.
    """
    for chapter in document.get("chapters", []):
        for passage in chapter.get("passages", []):
            for sentence in passage.get("sentences", []):
                for word in sentence.get("word_details", []):
                    yield word


# ============================================================
# LOAD AND PROCESS
# ============================================================

files = load_json_files(INPUT_FOLDER)

for document in files:
    file_name = document["file_name"]

    for word_data in iter_word_details(document):
        weights = word_data.get("weights", {})
        if not isinstance(weights, dict):
            weights = {}

        pos_lemma = word_data.get("lemma", "")
        pos_tag = word_data.get("pos", "")
        pos_key = (pos_lemma, pos_tag)

        file_pos_word_counts[file_name][pos_key] += 1
        global_pos_word_counts[pos_key] += 1

        for emotion in EMOTIONS:
            weight = get_numeric_weight(weights, emotion)
            file_pos_emotion_weights[emotion][file_name][pos_key] += weight
            global_pos_emotion_weights[emotion][pos_key] += weight

        lemma = clean_lemma(
            word_data.get("lemma")
        )

        if not lemma:
            continue

        # ----------------------------------------------------
        # RAW FREQUENCY
        # ----------------------------------------------------

        file_word_counts[file_name][lemma] += 1
        global_word_counts[lemma] += 1

        # ----------------------------------------------------
        # EMOTION-WEIGHTED CONTRIBUTION
        # ----------------------------------------------------

        occurrence_total_weight = 0.0

        for emotion in EMOTIONS:
            weight = get_numeric_weight(
                weights,
                emotion,
            )

            file_emotion_contribution[
                file_name
            ][emotion][lemma] += weight

            global_emotion_contribution[
                emotion
            ][lemma] += weight

            occurrence_total_weight += weight

            if weight != 0:
                file_emotion_nonzero_counts[
                    file_name
                ][emotion][lemma] += 1

                global_emotion_nonzero_counts[
                    emotion
                ][lemma] += 1

        file_total_emotional_contribution[
            file_name
        ][lemma] += occurrence_total_weight

        global_total_emotional_contribution[
            lemma
        ] += occurrence_total_weight


# ============================================================
# CREATE GLOBAL LEMMA STATISTICS
# ============================================================

global_rows = []

for lemma, frequency in global_word_counts.items():
    total_contribution = (
        global_total_emotional_contribution.get(
            lemma,
            0.0,
        )
    )

    row = {
        "lemma": lemma,
        "frequency": frequency,
        "total_emotional_contribution": total_contribution,
        "mean_emotional_intensity": safe_divide(
            total_contribution,
            frequency,
        ),
    }

    for emotion in EMOTIONS:
        emotion_contribution = (
            global_emotion_contribution[
                emotion
            ].get(
                lemma,
                0.0,
            )
        )

        nonzero_count = (
            global_emotion_nonzero_counts[
                emotion
            ].get(
                lemma,
                0,
            )
        )

        row[f"{emotion}_contribution"] = (
            emotion_contribution
        )

        # Average over all occurrences of the lemma.
        row[f"{emotion}_mean_all_occurrences"] = (
            safe_divide(
                emotion_contribution,
                frequency,
            )
        )

        # Average only over occurrences with a non-zero
        # contribution to this emotion.
        row[f"{emotion}_mean_nonzero_occurrences"] = (
            safe_divide(
                emotion_contribution,
                nonzero_count,
            )
        )

        row[f"{emotion}_nonzero_occurrences"] = (
            nonzero_count
        )

    global_rows.append(row)

global_lemma_stats = pd.DataFrame(global_rows)

if not global_lemma_stats.empty:
    global_lemma_stats = global_lemma_stats.sort_values(
        by=[
            "total_emotional_contribution",
            "frequency",
        ],
        ascending=[
            False,
            False,
        ],
    ).reset_index(drop=True)


# ============================================================
# CREATE FILE-LEVEL LEMMA STATISTICS
# ============================================================

file_rows = []

for file_name, counter in file_word_counts.items():
    for lemma, frequency in counter.items():
        total_contribution = (
            file_total_emotional_contribution[
                file_name
            ].get(
                lemma,
                0.0,
            )
        )

        row = {
            "file": file_name,
            "lemma": lemma,
            "frequency": frequency,
            "total_emotional_contribution": total_contribution,
            "mean_emotional_intensity": safe_divide(
                total_contribution,
                frequency,
            ),
        }

        for emotion in EMOTIONS:
            emotion_contribution = (
                file_emotion_contribution[
                    file_name
                ][emotion].get(
                    lemma,
                    0.0,
                )
            )

            nonzero_count = (
                file_emotion_nonzero_counts[
                    file_name
                ][emotion].get(
                    lemma,
                    0,
                )
            )

            row[f"{emotion}_contribution"] = (
                emotion_contribution
            )

            row[f"{emotion}_mean_all_occurrences"] = (
                safe_divide(
                    emotion_contribution,
                    frequency,
                )
            )

            row[
                f"{emotion}_mean_nonzero_occurrences"
            ] = safe_divide(
                emotion_contribution,
                nonzero_count,
            )

            row[f"{emotion}_nonzero_occurrences"] = (
                nonzero_count
            )

        file_rows.append(row)

file_lemma_stats = pd.DataFrame(file_rows)

if not file_lemma_stats.empty:
    file_lemma_stats = file_lemma_stats.sort_values(
        by=[
            "file",
            "total_emotional_contribution",
            "frequency",
        ],
        ascending=[
            True,
            False,
            False,
        ],
    ).reset_index(drop=True)


# ============================================================
# CREATE GLOBAL EMOTION–LEMMA TABLE
# ============================================================

emotion_lemma_rows = []

for emotion in EMOTIONS:
    normalized_contribution = normalize_mapping(
        global_emotion_contribution[emotion]
    )

    for lemma, contribution in (
        global_emotion_contribution[
            emotion
        ].items()
    ):
        frequency = global_word_counts.get(
            lemma,
            0,
        )

        nonzero_count = (
            global_emotion_nonzero_counts[
                emotion
            ].get(
                lemma,
                0,
            )
        )

        emotion_lemma_rows.append({
            "emotion": emotion,
            "lemma": lemma,
            "frequency": frequency,
            "nonzero_occurrences": nonzero_count,
            "cumulative_contribution": contribution,
            "normalized_contribution":
                normalized_contribution.get(
                    lemma,
                    0.0,
                ),
            "mean_weight_all_occurrences":
                safe_divide(
                    contribution,
                    frequency,
                ),
            "mean_weight_nonzero_occurrences":
                safe_divide(
                    contribution,
                    nonzero_count,
                ),
        })

global_emotion_lemma_stats = pd.DataFrame(
    emotion_lemma_rows
)

if not global_emotion_lemma_stats.empty:
    global_emotion_lemma_stats = (
        global_emotion_lemma_stats
        .sort_values(
            by=[
                "emotion",
                "cumulative_contribution",
                "frequency",
            ],
            ascending=[
                True,
                False,
                False,
            ],
        )
        .reset_index(drop=True)
    )


# ============================================================
# CREATE POS-AWARE FREQUENCY REPORTS
# ============================================================

global_pos_frequency = pd.DataFrame(
    [
        {
            "lemma": lemma,
            "pos": pos,
            "frequency": frequency,
        }
        for (lemma, pos), frequency in global_pos_word_counts.items()
    ],
    columns=["lemma", "pos", "frequency"],
).sort_values("frequency", ascending=False)

global_pos_frequency.to_csv(
    POS_REPORT_FOLDER / "global_lemma_pos_frequency.csv",
    index=False,
)

file_pos_frequency_rows = [
    {
        "file": file_name,
        "lemma": lemma,
        "pos": pos,
        "frequency": frequency,
    }
    for file_name, counter in file_pos_word_counts.items()
    for (lemma, pos), frequency in counter.items()
]

pd.DataFrame(
    file_pos_frequency_rows,
    columns=["file", "lemma", "pos", "frequency"],
).to_csv(
    POS_REPORT_FOLDER / "file_lemma_pos_frequency.csv",
    index=False,
)

all_pos_keys = sorted(global_pos_word_counts.keys())
normalized_pos_emotions = {
    emotion: normalize_mapping(global_pos_emotion_weights[emotion])
    for emotion in EMOTIONS
}

global_pos_emotion_rows = []
for lemma, pos in all_pos_keys:
    row = {
        "lemma": lemma,
        "pos": pos,
        "frequency": global_pos_word_counts[(lemma, pos)],
    }
    for emotion in EMOTIONS:
        row[emotion] = normalized_pos_emotions[emotion].get(
            (lemma, pos),
            0.0,
        )
    global_pos_emotion_rows.append(row)

pd.DataFrame(
    global_pos_emotion_rows,
    columns=["lemma", "pos", "frequency"] + EMOTIONS,
).to_csv(
    POS_REPORT_FOLDER / "global_lemma_pos_emotion_lexicon.csv",
    index=False,
)

file_pos_emotion_rows = [
    {
        "file": file_name,
        "lemma": lemma,
        "pos": pos,
        "frequency": frequency,
        **{
            emotion: file_pos_emotion_weights[emotion][file_name].get(
                (lemma, pos),
                0.0,
            )
            for emotion in EMOTIONS
        },
    }
    for file_name, counter in file_pos_word_counts.items()
    for (lemma, pos), frequency in counter.items()
]

pd.DataFrame(
    file_pos_emotion_rows,
    columns=["file", "lemma", "pos", "frequency"] + EMOTIONS,
).to_csv(
    POS_REPORT_FOLDER / "file_lemma_pos_emotion_lexicon.csv",
    index=False,
)

file_emotion_summary_rows = [
    {
        "file": file_name,
        "emotion": emotion,
        "weighted_sum": sum(
            file_pos_emotion_weights[emotion][file_name].values()
        ),
    }
    for file_name in file_pos_word_counts
    for emotion in EMOTIONS
]

pd.DataFrame(
    file_emotion_summary_rows,
    columns=["file", "emotion", "weighted_sum"],
).to_csv(
    POS_REPORT_FOLDER / "file_emotion_summary.csv",
    index=False,
)


# ============================================================
# CONSOLE OUTPUT
# ============================================================

print("\n===== GLOBAL TOP LEMMA + POS =====")
print(global_pos_word_counts.most_common(50))

for file_name, counter in file_pos_word_counts.items():
    print(f"\n===== FILE: {file_name} =====")
    print(counter.most_common(15))

for emotion in EMOTIONS:
    print(f"\n===== EMOTION: {emotion} =====")
    print(Counter(global_pos_emotion_weights[emotion]).most_common(20))


print("\n===== GLOBAL TOP WORDS BY RAW FREQUENCY =====")

for lemma, frequency in global_word_counts.most_common(
    GLOBAL_TOP_N
):
    print(
        f"{lemma:<25} "
        f"frequency={frequency}"
    )


print(
    "\n===== GLOBAL TOP WORDS BY "
    "CUMULATIVE EMOTIONAL CONTRIBUTION ====="
)

for lemma, score in top_values(
    global_total_emotional_contribution,
    GLOBAL_TOP_N,
):
    print(
        f"{lemma:<25} "
        f"contribution={score:.4f} "
        f"frequency={global_word_counts[lemma]}"
    )


print(
    "\n===== GLOBAL TOP WORDS BY "
    "MEAN EMOTIONAL INTENSITY ====="
)

if not global_lemma_stats.empty:
    top_mean_intensity = (
        global_lemma_stats
        .sort_values(
            by=[
                "mean_emotional_intensity",
                "frequency",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .head(GLOBAL_TOP_N)
    )

    for _, row in top_mean_intensity.iterrows():
        print(
            f"{row['lemma']:<25} "
            f"mean={row['mean_emotional_intensity']:.4f} "
            f"frequency={int(row['frequency'])} "
            f"contribution="
            f"{row['total_emotional_contribution']:.4f}"
        )


for file_name, counter in file_word_counts.items():
    print(
        f"\n===== FILE: {file_name} — "
        f"TOP RAW FREQUENCY ====="
    )

    for lemma, frequency in counter.most_common(
        FILE_TOP_N
    ):
        print(
            f"{lemma:<25} "
            f"frequency={frequency}"
        )


for emotion in EMOTIONS:
    print("\n==============================")
    print(
        f"EMOTION: {emotion.upper()} — "
        f"TOP CUMULATIVE CONTRIBUTION"
    )

    for lemma, contribution in top_values(
        global_emotion_contribution[
            emotion
        ],
        EMOTION_TOP_N,
    ):
        frequency = global_word_counts.get(
            lemma,
            0,
        )

        mean_weight = safe_divide(
            contribution,
            frequency,
        )

        print(
            f"{lemma:<25} "
            f"contribution={contribution:.4f} "
            f"mean={mean_weight:.4f} "
            f"frequency={frequency}"
        )


# ============================================================
# SAVE OUTPUTS
# ============================================================

global_lemma_stats.to_csv(
    OUTPUT_FOLDER / "global_lemma_statistics.csv",
    index=False,
    encoding="utf-8-sig",
)

file_lemma_stats.to_csv(
    OUTPUT_FOLDER / "file_lemma_statistics.csv",
    index=False,
    encoding="utf-8-sig",
)

global_emotion_lemma_stats.to_csv(
    OUTPUT_FOLDER / "global_emotion_lemma_statistics.csv",
    index=False,
    encoding="utf-8-sig",
)

# Separate top lists for easier use in tables and figures
global_top_frequency = (
    global_lemma_stats
    .sort_values(
        by=[
            "frequency",
            "total_emotional_contribution",
        ],
        ascending=[
            False,
            False,
        ],
    )
    .head(GLOBAL_TOP_N)
)

global_top_contribution = (
    global_lemma_stats
    .sort_values(
        by=[
            "total_emotional_contribution",
            "frequency",
        ],
        ascending=[
            False,
            False,
        ],
    )
    .head(GLOBAL_TOP_N)
)

global_top_mean_intensity = (
    global_lemma_stats
    .sort_values(
        by=[
            "mean_emotional_intensity",
            "frequency",
        ],
        ascending=[
            False,
            False,
        ],
    )
    .head(GLOBAL_TOP_N)
)

global_top_frequency.to_csv(
    OUTPUT_FOLDER / "top_lemmas_by_frequency.csv",
    index=False,
    encoding="utf-8-sig",
)

global_top_contribution.to_csv(
    OUTPUT_FOLDER
    / "top_lemmas_by_cumulative_emotional_contribution.csv",
    index=False,
    encoding="utf-8-sig",
)

global_top_mean_intensity.to_csv(
    OUTPUT_FOLDER
    / "top_lemmas_by_mean_emotional_intensity.csv",
    index=False,
    encoding="utf-8-sig",
)


# One top-lemma file per emotion
for emotion in EMOTIONS:
    emotion_subset = (
        global_emotion_lemma_stats[
            global_emotion_lemma_stats[
                "emotion"
            ] == emotion
        ]
        .sort_values(
            by=[
                "cumulative_contribution",
                "frequency",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .head(EMOTION_TOP_N)
    )

    emotion_subset.to_csv(
        OUTPUT_FOLDER
        / f"top_lemmas_{emotion}.csv",
        index=False,
        encoding="utf-8-sig",
    )


print("\nDONE ✔")
print(
    f"Word-statistics outputs saved to: "
    f"{OUTPUT_FOLDER.resolve()}"
)
print(
    f"POS reports saved to: "
    f"{POS_REPORT_FOLDER.resolve()}"
)
print(
    f"Console report saved to: "
    f"{TEXT_REPORT_PATH.resolve()}"
)
