import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lxml import etree
import re
import seaborn as sns

EMOTION_COLORS = {
    "anger": "#e74c3c",
    "anticipation": "#e68937",
    "neutral": "#95a5a6",
    "trust": "#29ca6c",
    "joy": "#f1c40f",
    "disgust": "#b267d0",
    "surprise": "#3498db",
    "sadness": "#2071c2",
    "fear": "#1f854a",
}

def _extract_all_passages(data):
    """
    Helper function to linearly extract all passages across chapters 
    to preserve structural continuity during chronological plotting.
    """
    all_passages = []
    for ch in data.get('chapters', []):
        ch_id = ch.get('chapter_id', 'unknown_ch')
        ch_title = ch.get('chapter_title', '')
        for p in ch.get('passages', []):
            p_mod = p.copy()
            p_mod['chapter_id'] = ch_id
            p_mod['chapter_title'] = ch_title
            all_passages.append(p_mod)
    return all_passages


def plot_all_emotions_flow(json_path, output_image_folder="plots/emotion_dynamics", smoothing_window=50):
    """
    Loads a single structural JSON analysis file and plots a comparative 
    timeline plus 8 individual filled subplots tracking Plutchik's core matrix.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    passages = _extract_all_passages(data)
    if not passages:
        print(f"[SKIP] File {data.get('file_name', json_path)} contains no structural passages.")
        return
        
    passage_ids = list(range(len(passages)))
    emotions = [emo for emo in EMOTION_COLORS.keys() if emo != 'neutral']
    
    # 1. Trajectory Data Extraction
    raw_trajectories = {emo: [] for emo in emotions}
    for p in passages:
        # Backward compatibility for old "skorovi" key variations
        scores = p.get('scores', p.get('skorovi_normalizovani', p.get('skorovi', {})))
        for emo in emotions:
            val = scores.get(emo, 0.0) if scores else 0.0
            raw_trajectories[emo].append(float(val))
            
    # 2. Moving Average Convolution (Line Smoothing)
    smoothed_trajectories = {}
    for emo, values in raw_trajectories.items():
        if len(values) > smoothing_window:
            box = np.ones(smoothing_window) / smoothing_window
            smoothed_trajectories[emo] = np.convolve(values, box, mode='same')
        else:
            smoothed_trajectories[emo] = np.array(values)

    # 3. Complex Multi-Grid Grid Visualization Setup
    num_emotions = len(emotions)
    fig = plt.figure(figsize=(15, 2 * num_emotions + 3))
    gs = fig.add_gridspec(num_emotions + 2, 1, hspace=0.6)
    
    # Top Comparative Main Plot Area
    ax_main = fig.add_subplot(gs[0:2, 0])
    for emo in emotions:
        ax_main.plot(passage_ids, smoothed_trajectories[emo], label=emo.capitalize(), 
                     color=EMOTION_COLORS[emo], linewidth=2, alpha=0.85)
    
    ax_main.set_title(f"Narrative Emotion Trajectory (Window={smoothing_window} p): {data['file_name']}", fontsize=14, pad=12, fontweight='bold')
    ax_main.set_ylabel("Intensity Intensity Score", fontsize=11)
    ax_main.grid(True, linestyle='--', alpha=0.3)
    ax_main.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='none', ncol=4, fontsize=10)
    
    # 4. Individual Isolated Subplots
    for idx, emo in enumerate(emotions):
        ax_sub = fig.add_subplot(gs[2 + idx, 0], sharex=ax_main)
        ax_sub.fill_between(passage_ids, smoothed_trajectories[emo], color=EMOTION_COLORS[emo], alpha=0.15)
        ax_sub.plot(passage_ids, smoothed_trajectories[emo], color=EMOTION_COLORS[emo], linewidth=1.8)
        ax_sub.set_ylabel(emo.capitalize(), color=EMOTION_COLORS[emo], fontsize=10, fontweight='bold')
        ax_sub.grid(True, linestyle=':', alpha=0.5)
        
        # Hide internal x-axis tick marks except for the bottom-most plot sequence
        if idx < num_emotions - 1:
            plt.setp(ax_sub.get_xticklabels(), visible=False)
            
    plt.xlabel("Sequential Order of Passages (Narrative Timeline)", fontsize=11)
    
    # 5. File System Safe Save
    if not os.path.exists(output_image_folder):
        os.makedirs(output_image_folder)
        
    base_name = os.path.basename(json_path)
    for suffix in ['_emo_analiza.json', '_emocije_analiza.json', '_analysis.json', '.json']:
        if base_name.endswith(suffix):
            base_name = base_name.replace(suffix, '')
            break
            
    output_filename = f"{base_name}_emotion_flow.png"
    save_path = os.path.join(output_image_folder, output_filename)
    
    plt.savefig(save_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"[SUCCESS] Exported high-res chart: {output_filename}")


def batch_generate_emotion_plots(json_folder, output_image_folder="plots/emotion_dynamics"):
    """
    Iterates automatically through target folders containing analysis documents 
    and exports sequential timeline visuals onto disk storage.
    """
    print(f"\n[START] Mass generating visual profiles from source folder: '{json_folder}'...")
    
    if not os.path.exists(json_folder):
        print(f"[ERROR] Source registry path '{json_folder}' does not exist.")
        return

    counter = 0
    for file in os.listdir(json_folder):
        if file.endswith('_emo_analiza.json') or file.endswith('_analysis.json'):
            full_path = os.path.join(json_folder, file)
            plot_all_emotions_flow(full_path, output_image_folder=output_image_folder)
            counter += 1
            
    print(f"[SUCCESS] Completed. Generated {counter} structural plots inside: '{output_image_folder}/'\n")


def plot_corpus_average_emotions(
    json_folder,
    resolution=20,
    output_image="plots/corpus_level/corpus_emotion_average.png",
):
    """
    Interpolates varying document lengths into static structural timeline milestones,
    aggregating overall emotional trends across the entire parsed novel corpus.
    """
    emotions = [emo for emo in EMOTION_COLORS.keys() if emo != 'neutral']
    corpus_records = {emo: [] for emo in emotions}
    file_counter = 0
    
    if not os.path.exists(json_folder):
        print(f"[ERROR] Specified data directory '{json_folder}' cannot be found.")
        return

    for file in os.listdir(json_folder):
        if file.endswith('_emo_analiza.json') or file.endswith('_analysis.json'):
            with open(os.path.join(json_folder, file), 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            passages = _extract_all_passages(data)
            if not passages:
                continue
                
            file_counter += 1
            temp_timeline = {emo: [] for emo in emotions}
            
            for p in passages:
                scores = p.get('scores', p.get('skorovi_normalizovani', p.get('skorovi', {})))
                for emo in emotions:
                    val = scores.get(emo, 0.0) if scores else 0.0
                    temp_timeline[emo].append(float(val))
            
            # Interpolate raw text arrays into uniform timeline phases
            x_old = np.linspace(0, 1, len(passages))
            x_new = np.linspace(0, 1, resolution)
            for emo in emotions:
                scaled_flow = np.interp(x_new, x_old, temp_timeline[emo])
                corpus_records[emo].append(scaled_flow)
                
    if file_counter == 0:
        print("[WARNING] Zero valid datasets located. Aborting corpus plot generation.")
        return
        
    plt.figure(figsize=(13, 6.5))
    timeline_percentages = np.linspace(0, 100, resolution)
    
    for emo in emotions:
        mean_trajectory = np.mean(corpus_records[emo], axis=0)
        plt.plot(timeline_percentages, mean_trajectory, label=emo.capitalize(),
                 color=EMOTION_COLORS[emo], linewidth=2.5, marker='o', markersize=4)
        
    plt.title(f"Corpus-Level Emotion Trends Across Narrative Progression (Total Novels: {file_counter})", fontsize=13, pad=15, fontweight='bold')
    plt.xlabel("Narrative Structural Progression (% of Book Length)", fontsize=11)
    plt.ylabel("Mean Emotion Intensity Score", fontsize=11)
    plt.xticks(np.arange(0, 101, 10), [f"{i}%" for i in range(0, 101, 10)])
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='none')
    plt.tight_layout()
    output_directory = os.path.dirname(output_image)
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)
    plt.savefig(output_image, dpi=300)
    plt.close()
    print(f"[SUCCESS] Corpus-level visualization successfully written to: '{output_image}'\n")


def load_corpus_summary(tsv_path):
    """Load and enrich the corpus-level emotion summary table."""
    data = pd.read_csv(tsv_path, sep="\t")
    data["emotional_density"] = (
        data["emotional_density"]
        .astype(str)
        .str.rstrip("%")
        .astype(float)
    )

    if data["emotional_density"].max() <= 1.0:
        data["emotional_density"] *= 100

    data["year"] = data["file_name"].str.extract(
        r"SRP(\d{4})"
    ).astype(float)
    data["decade"] = (
        (data["year"] // 10)
        .fillna(185)
        .astype(int)
        .mul(10)
        .astype(str)
        + "s"
    )
    data["author"] = data["file_name"].str.extract(
        r"SRP\d{5}_([^_]+)_"
    )
    data["title"] = data["file_name"].str.extract(
        r"SRP\d{5}_[^_]+_([^-]+)-"
    )
    data["short_name"] = (
        data["author"].fillna("Unknown")
        + "_"
        + data["title"].fillna("Novel")
    )
    return data


def plot_corpus_emotion_correlation_heatmap(
    corpus_data,
    output_directory,
):
    """Plot the corpus-level correlation matrix of the eight emotions."""
    emotions = [
        "anger",
        "anticipation",
        "disgust",
        "fear",
        "joy",
        "sadness",
        "surprise",
        "trust",
    ]
    os.makedirs(output_directory, exist_ok=True)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        corpus_data[emotions].corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5,
        cbar=True,
    )
    plt.title(
        "Correlation Matrix of Plutchik Emotions in the Corpus",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    output_path = os.path.join(
        output_directory,
        "corpus_emotion_correlation_heatmap.png",
    )
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def plot_corpus_macro_density_boxplot(
    corpus_data,
    output_directory,
):
    """Plot emotional-density variation across publication decades."""
    os.makedirs(output_directory, exist_ok=True)
    data = corpus_data.sort_values("year")

    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=data,
        x="decade",
        y="emotional_density",
        palette="Blues_r",
        linewidth=1.5,
        fliersize=4,
    )
    sns.stripplot(
        data=data,
        x="decade",
        y="emotional_density",
        color="#1B365D",
        alpha=0.4,
        size=4,
        jitter=0.2,
    )
    plt.title(
        "Macro Shift in Emotional Density Across Serbian Novel Decades",
        weight="bold",
        pad=15,
    )
    plt.xlabel("Historical Period (Decade)", labelpad=10)
    plt.ylabel("Text Emotional Density (%)", labelpad=10)
    plt.tight_layout()
    output_path = os.path.join(
        output_directory,
        "corpus_macro_density_boxplot.png",
    )
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def plot_corpus_pca_space(corpus_data, output_directory):
    """Plot the corpus novels in a two-dimensional emotion PCA space."""
    from adjustText import adjust_text
    from sklearn.decomposition import PCA

    emotions = [
        "anger",
        "anticipation",
        "disgust",
        "fear",
        "joy",
        "sadness",
        "surprise",
        "trust",
    ]
    os.makedirs(output_directory, exist_ok=True)
    data = corpus_data.copy()

    pca = PCA(n_components=2)
    transformed = pca.fit_transform(data[emotions].values)
    data["PCA1"] = transformed[:, 0]
    data["PCA2"] = transformed[:, 1]

    figure, axis = plt.subplots(figsize=(11, 8.5), dpi=300)
    sns.scatterplot(
        data=data.sort_values("decade"),
        x="PCA1",
        y="PCA2",
        hue="decade",
        style="decade",
        palette="turbo",
        s=130,
        alpha=0.9,
        edgecolor="black",
        linewidth=0.6,
        ax=axis,
    )

    distances = data["PCA1"] ** 2 + data["PCA2"] ** 2
    texts = []
    for index in distances.nlargest(6).index:
        texts.append(
            axis.text(
                data.loc[index, "PCA1"],
                data.loc[index, "PCA2"],
                data.loc[index, "short_name"],
                fontsize=9.5,
                weight="bold",
                bbox={
                    "facecolor": "white",
                    "alpha": 0.92,
                    "edgecolor": "#555555",
                    "boxstyle": "round,pad=0.25",
                    "lw": 0.7,
                },
            )
        )

    adjust_text(
        texts,
        x=data["PCA1"].values,
        y=data["PCA2"].values,
        force_points=0.2,
        force_text=0.3,
        expand_points=(2.5, 2.5),
        arrowprops={"arrowstyle": "-", "color": "#777777", "lw": 0.6},
    )

    axis.set_xlabel(
        f"Principal Component 1 "
        f"({pca.explained_variance_ratio_[0] * 100:.1f}% Variance)"
    )
    axis.set_ylabel(
        f"Principal Component 2 "
        f"({pca.explained_variance_ratio_[1] * 100:.1f}% Variance)"
    )
    axis.legend(
        title="Decade",
        loc="lower left",
        frameon=False,
    )
    plt.tight_layout()
    output_path = os.path.join(
        output_directory,
        "corpus_pca_space.png",
    )
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return output_path


def generate_corpus_level_plots(
    tsv_path,
    output_directory="plots/corpus_level",
):
    """Generate the active corpus-level correlation, density, and PCA plots."""
    corpus_data = load_corpus_summary(tsv_path)
    output_paths = [
        plot_corpus_emotion_correlation_heatmap(
            corpus_data,
            output_directory,
        ),
        plot_corpus_macro_density_boxplot(
            corpus_data,
            output_directory,
        ),
        plot_corpus_pca_space(
            corpus_data,
            output_directory,
        ),
    ]
    print(
        "[SUCCESS] Corpus-level correlation, density, and PCA plots "
        f"written to: '{output_directory}/'"
    )
    return output_paths


def top_passages_per_emotion(emotion, top_n=5, min_emo_words=4, json_folder='emotion_analysis_results'):
    """
    Queries across parsed records to return top narrative passages optimized by specific emotional spikes.
    Filters out narrative noise by enforcing a minimum emotional word count threshold per passage.
    """
    passage_list = []

    for file in os.listdir(json_folder):
        if file.endswith('_emo_analiza.json') or file.endswith('_analysis.json'):
            with open(os.path.join(json_folder, file), 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            for p in _extract_all_passages(data):
                # Fetch emotional words count for the current passage
                emotional_words_count = p.get('emotional_word_count', 0)
                
                # --- NOISE FILTER: Enforce the minimum emotional words condition ---
                if emotional_words_count >= min_emo_words:
                    scores = p.get('scores', p.get('skorovi_normalizovani', p.get('skorovi', {})))
                    raw_score = scores.get(emotion, 0.0) if scores else 0.0
                    
                    passage_list.append({
                        'file_name': data.get('file_name', file),
                        'chapter_id': p.get('chapter_id', 'unknown_ch'),
                        'passage_id': p.get('passage_id', 'unknown_p'),
                        'total_words': p.get('total_words', 0),
                        'emotional_words': emotional_words_count,
                        'density_%': p.get('emotional_density', 0.0),
                        'intensity_score': round(raw_score, 4)
                    })

    if not passage_list:
        print(f"[WARNING] No data found matching criteria parameters for emotion: {emotion} (with min_emo_words >= {min_emo_words})")
        return None

    df = pd.DataFrame(passage_list)
    top_p = df.sort_values(by='intensity_score', ascending=False).head(top_n)
    
    # Updated header printout to clearly state the applied minimum emotional word constraint
    print(f"\n--- TOP {top_n} PASSAGES FOR EMOTION: {emotion.upper()} (Criteria: Min {min_emo_words} emotional words) ---")
    print(top_p[['file_name', 'chapter_id', 'passage_id', 'total_words', 'emotional_words', 'density_%', 'intensity_score']].to_string(index=False))
    return top_p

def find_highest_emotional_charge_per_passage(json_folder, top_n=3):
    """Equivalent to Step 5: Locates passages with highest emotional keyword density."""
    for file in os.listdir(json_folder):
        if file.endswith('_analysis.json') or file.endswith('_emo_analiza.json'):
            with open(os.path.join(json_folder, file), 'r', encoding='utf-8') as f:
                data = json.load(f)
            passages = _extract_all_passages(data)
            if not passages:
                continue
            sorted_p = sorted(passages, key=lambda x: x.get('emotional_density', 0.0), reverse=True)
            print(f"\n===== NOVEL: {data['file_name']} =====")
            for i, p in enumerate(sorted_p[:top_n]):
                print(f"  {i+1}. Ch: {p['chapter_id']} | Passage: {p['passage_id']} -> Density: {p['emotional_density']}%")

def analyze_emotional_chapter_climax(json_folder, top_n=5):
    """Equivalent to Step 8: Maps macro chapters acting as emotional spikes."""
    chapter_list = []
    for file in os.listdir(json_folder):
        if file.endswith('_analysis.json') or file.endswith('_emo_analiza.json'):
            with open(os.path.join(json_folder, file), 'r', encoding='utf-8') as f:
                data = json.load(f)
            for ch in data.get('chapters', []):
                if ch.get('total_words', 0) > 100:
                    chapter_list.append({
                        'file_name': data['file_name'],
                        'chapter_id': ch['chapter_id'],
                        'chapter_title': ch['chapter_title'],
                        'density_%': ch['emotional_density']
                    })
    if chapter_list:
        df = pd.DataFrame(chapter_list).sort_values(by='density_%', ascending=False).head(top_n)
        print(f"\n--- TOP {top_n} CHAPTERS WITH HIGHEST EMOTIONAL CLIMAX ---")
        print(df.to_string(index=False))

def analyze_emotional_passage_climax(json_folder, top_n=5):
    """Prolazi kroz sve fajlove i pronalazi pasuse sa najvećom emocionalnom gustinom."""
    passage_list = []
    
    for file in os.listdir(json_folder):
        if file.endswith('_analysis.json') or file.endswith('_emo_analiza.json'):
            with open(os.path.join(json_folder, file), 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            file_name = data.get('file_name', file)
            
            # Spuštamo se kroz poglavlja do pasusa
            for ch in data.get('chapters', []):
                ch_title = ch.get('chapter_title', 'Unknown').replace('\n', ' ').strip()
                
                for p in ch.get('passages', []):
                    total_words = p.get('total_words', 0)
                    
                    # Filter: Ignorišemo prekratke pasuse da izbegnemo statistički šum
                    if total_words > 15: 
                        passage_list.append({
                            'file_name': file_name,
                            'chapter_title': ch_title,
                            'passage_id': p.get('passage_id', 'N/A'),
                            'words': total_words,
                            'density_%': p.get('emotional_density', 0)
                        })
                        
    if passage_list:
        df = pd.DataFrame(passage_list).sort_values(by='density_%', ascending=False).head(top_n)
        print(f"\n--- TOP {top_n} PARAGRAPHS WITH HIGHEST EMOTIONAL CLIMAX ---")
        print(df.to_string(index=False))
    else:
        print("No passages found matching the criteria.")

def analyze_emotional_sentence_climax(json_folder, top_n=5):
    """Spušta se do nivoa rečenica kroz sve JSON fajlove i pronalazi najveće emocionalne skokove."""
    sentence_list = []
    
    for file in os.listdir(json_folder):
        if file.endswith('_analysis.json') or file.endswith('_emo_analiza.json'):
            with open(os.path.join(json_folder, file), 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            file_name = data.get('file_name', file)
            
            # Trostruka petla: Chapters -> Passages -> Sentences
            for ch in data.get('chapters', []):
                for p in ch.get('passages', []):
                    for s in p.get('sentences', []):
                        total_words = s.get('total_words', 0)
                        
                        # Filter: Rečenica mora imati bar 5 reči
                        if total_words > 5:
                            # Rekonstruišemo delimičan tekst od pronađenih reči radi lakšeg konteksta
                            words_inside = [w.get('word', '') for w in s.get('word_details', [])]
                            words_preview = " ".join(words_inside)[:50] + "..." if words_inside else "N/A"
                            
                            sentence_list.append({
                                'file_name': file_name,
                                'sentence_id': s.get('sentence_id', 'N/A'),
                                'words_count': total_words,
                                'emo_words': s.get('emotional_word_count', 0),
                                'density_%': s.get('emotional_density', 0),
                                'words_preview': words_preview
                            })
                            
    if sentence_list:
        df = pd.DataFrame(sentence_list).sort_values(by='density_%', ascending=False).head(top_n)
        print(f"\n--- TOP {top_n} SENTENCES WITH HIGHEST EMOTIONAL CLIMAX ---")
        print(df.to_string(index=False))
    else:
        print("No sentences found matching the criteria.")


def analyze_emotional_sentence_climax_with_text(json_folder, xml_folder, top_n=5):
    """
    Pronalazi emotivne vrhunce u JSON-u, locira rečenicu u ELTeC XML-u,
    i rekonstruiše čist tekst iz ugnždenih <w> i <pc> tagova.
    """
    sentence_list = []
    
    # 1. Sakupljanje podataka iz JSON fajlova
    for file in os.listdir(json_folder):
        if file.endswith('_analysis.json') or file.endswith('_emo_analiza.json'):
            with open(os.path.join(json_folder, file), 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            xml_file_name = data.get('file_name') 
            if not xml_file_name:
                xml_file_name = file.replace('_analysis.json', '.xml').replace('_emo_analiza.json', '.xml')
            
            for ch in data.get('chapters', []):
                for p in ch.get('passages', []):
                    for s in p.get('sentences', []):
                        total_words = s.get('total_words', 0)
                        
                        if total_words > 5:
                            sentence_list.append({
                                'xml_file': xml_file_name,
                                'sentence_id': s.get('sentence_id', 'N/A'),
                                'words_count': total_words,
                                'emo_words': s.get('emotional_word_count', 0),
                                'density_%': s.get('emotional_density', 0)
                            })
                            
    if not sentence_list:
        print("Nema pronađenih rečenica koje zadovoljavaju kriterijum.")
        return

    # 2. Sortiranje i uzimanje TOP N
    df_top = pd.DataFrame(sentence_list).sort_values(by='density_%', ascending=False).head(top_n).copy()
    
    # 3. Rekonstrukcija rečenica iz ELTeC XML-a
    full_sentences_text = []
    
    for idx, row in df_top.iterrows():
        xml_path = os.path.join(xml_folder, row['xml_file'])
        s_id = row['sentence_id']
        text_found = "[Tekst nije pronađen u XML-u]"
        
        if os.path.exists(xml_path):
            try:
                with open(xml_path, 'r', encoding='utf-8', errors='ignore') as f:
                    xml_content = f.read()
                
                # Pronalazimo tačan <s>...</s> blok preko regexa (fleksibilno za xml:id)
                pattern = rf"<(?:tei:)?s\s+[^>]*?(?:xml:)?id=['\"]{s_id}['\"][^>]*?>(.*?)</(?:tei:)?s>"
                match = re.search(pattern, xml_content, re.DOTALL)
                
                if match:
                    inner_xml = match.group(1)
                    
                    # Da bismo ga lakše parsirali, omotavamo ga u privremeni koren
                    # i čistimo podrazumevani namespace da nas lxml ne muči
                    wrapped_xml = f"<root>{inner_xml}</root>"
                    
                    parser = etree.XMLParser(recover=True, encoding='utf-8')
                    s_tree = etree.fromstring(wrapped_xml, parser)
                    
                    words_tokens = []
                    # Prolazimo kroz sve elemente unutar rečenice (w i pc) redom kojim se pojavljuju
                    for el in s_tree.xpath("//w | //pc"):
                        token_text = el.text if el.text else ""
                        join_attr = el.get("join")
                        
                        # Ako element ima tekst, dodajemo ga
                        if token_text:
                            words_tokens.append((token_text, join_attr))
                    
                    # Rekonstrukcija rečenice sa pravilnim razmacima
                    sentence_bits = []
                    for i, (tok, join) in enumerate(words_tokens):
                        sentence_bits.append(tok)
                        # Ako nije poslednji token i nema oznaku da se lepi desno, dodajemo razmak
                        if i < len(words_tokens) - 1 and join != "right":
                            sentence_bits.append(" ")
                            
                    text_found = "".join(sentence_bits).strip()
                    # Popravka za česte slučajeve višestrukih razmaka
                    text_found = " ".join(text_found.split())
                    
            except Exception as e:
                text_found = f"[Greška tokom rekonstrukcije: {e}]"
        else:
            text_found = f"[XML fajl nedostaje: {row['xml_file']}]"
            
        full_sentences_text.append(text_found)
        
    df_top['full_sentence'] = full_sentences_text
    
    # Podešavanje pandas prikaza za čist terminalski ispis
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', 1000)
    
    df_top.to_csv('top_emotional_sentences_full.csv', sep='\t', index=False)  # Takođe čuvamo u CSV za dalje analize
    print(f"\n--- TOP {top_n} SENTENCES WITH HIGHEST EMOTIONAL CLIMAX (FULL TEXT) ---")
    columns_to_show = ['xml_file', 'sentence_id', 'density_%', 'full_sentence']
    print(df_top[columns_to_show].to_string(index=False))


def plot_emotion_trajectories_chapters(json_path, output_folder="plots/emotion_dynamics"):
    """Učitava JSON i crta linijski grafik kretanja emocija kroz poglavlja."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    title_of_work = data.get('file_name', os.path.basename(json_path)).replace('.xml', '').replace('_analysis.json', '')
    
    if 'chapters' not in data or not data['chapters']:
        print(f"Nema podataka o poglavljima u {json_path}")
        return
        
    # Ekstrakcija i "peglanje" podataka za poglavlja
    chapters_data = []
    for i, ch in enumerate(data['chapters']):
        row = {'Chapter': i + 1}
        scores = ch.get('scores', {})
        for emotion, val in scores.items():
            row[emotion] = val
        chapters_data.append(row)
        
    df_chapters = pd.DataFrame(chapters_data)
    
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    sns.set_theme(style="whitegrid")
    
    # Pratimo ključne Plutchik emocije za narativni tok
    emotions = [
        emotion for emotion in EMOTION_COLORS
        if emotion != "neutral"
    ]  # ['trust', 'joy', 'sadness', 'anger', 'fear']
    available_emotions = [e for e in emotions if e in df_chapters.columns]
    
    for emotion in available_emotions:
        ax.plot(
            df_chapters['Chapter'], 
            df_chapters[emotion], 
            marker='o', 
            linewidth=2, 
            markersize=6, 
            label=emotion.capitalize(), 
            color=EMOTION_COLORS.get(emotion, '#333333')
        )
        
    ax.set_title(f'Emotional Trajectories Across Chapters: {title_of_work}', weight='bold', fontsize=12, pad=15)
    ax.set_xlabel('Chapter Number', fontsize=10)
    ax.set_ylabel('Emotion Intensity', fontsize=10)
    ax.set_xticks(df_chapters['Chapter'])
    ax.legend(title='Emotions', loc='upper left', bbox_to_anchor=(1.02, 1), frameon=False)
    
    os.makedirs(output_folder, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"{title_of_work}_chapters_emotions.png"), bbox_inches='tight')
    plt.close()


def plot_emotion_heatmap_paragraphs(json_path, output_folder="plots/emotion_dynamics"):
    """Učitava ugnježdeni JSON, izvlači sve pasuse i pravi hronološku toplotnu mapu."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    title_of_work = data.get('file_name', os.path.basename(json_path)).replace('.xml', '').replace('_analysis.json', '')
    
    # Iteriramo kroz hijerarhiju da izvučemo sve pasuse hronološki
    passages_data = []
    p_idx = 1
    for ch in data.get('chapters', []):
        for p in ch.get('passages', []):
            row = {'Paragraph_Seq': p_idx}
            scores = p.get('scores', {})
            for emotion, val in scores.items():
                row[emotion] = val
            passages_data.append(row)
            p_idx += 1
            
    if not passages_data:
        print(f"Nema podataka o pasusima u {json_path}")
        return
        
    df_passages = pd.DataFrame(passages_data)
    emotions = [
        emotion for emotion in EMOTION_COLORS
        if emotion != "neutral"
    ]  # ['trust', 'anticipation', 'joy', 'surprise', 'sadness', 'fear', 'anger', 'disgust']
    available_emotions = [e for e in emotions if e in df_passages.columns]
    
    # Transponujemo da emocije budu na Y osi, a pasusi na X osi
    heatmap_data = df_passages[available_emotions].T
    
    fig, ax = plt.subplots(figsize=(12, 4.5), dpi=300)
    
    sns.heatmap(
        heatmap_data, 
        cmap='magma', 
        cbar_kws={'label': 'Intensity Score'},
        xticklabels=max(1, len(df_passages) // 15), # Prikaz oznaka na svakih 15 pasusa radi čitljivosti
        ax=ax
    )
    
    ax.set_title(f'Emotional Heatmaps Across Paragraphs: {title_of_work}', weight='bold', fontsize=12, pad=15)
    ax.set_xlabel('Paragraph Sequence (Timeline)', fontsize=10)
    ax.set_ylabel('Plutchik Emotions', fontsize=10)
    ax.set_yticklabels([t.get_text().capitalize() for t in ax.get_yticklabels()], rotation=0)
    
    os.makedirs(output_folder, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"{title_of_work}_paragraphs_heatmap.png"), bbox_inches='tight')
    plt.close()


def plot_emotion_curves_sentences(json_path, output_folder="plots/emotion_dynamics", window_size=40):
    """Prolazi kroz trostruko ugnježdenu strukturu, sakuplja sve rečenice i crta izglađene krive."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    title_of_work = data.get('file_name', os.path.basename(json_path)).replace('.xml', '').replace('_analysis.json', '')
    
    # Duboko otpakivanje hronološkog toka svih rečenica u delu
    sentences_data = []
    s_idx = 1
    for ch in data.get('chapters', []):
        for p in ch.get('passages', []):
            for s in p.get('sentences', []):
                row = {'Sentence_Seq': s_idx}
                scores = s.get('scores', {})
                for emotion, val in scores.items():
                    row[emotion] = val
                sentences_data.append(row)
                s_idx += 1
                
    if not sentences_data:
        print(f"Nema podataka o rečenicama u {json_path}")
        return
        
    df_sentences = pd.DataFrame(sentences_data)
    
    fig, ax = plt.subplots(figsize=(12, 5), dpi=300)
    sns.set_theme(style="white")
    
    focus_emotions = [
        emotion for emotion in EMOTION_COLORS
        if emotion != "neutral"
    ]  # ['joy', 'sadness', 'trust', 'fear']
    colors = EMOTION_COLORS
    
    for emotion in focus_emotions:
        if emotion in df_sentences.columns:
            # Primena kliznog proseka (Rolling Mean) na izvučene vrednosti
            smoothed_values = df_sentences[emotion].rolling(window=window_size, center=True, min_periods=1).mean()
            
            ax.plot(
                df_sentences['Sentence_Seq'], 
                smoothed_values, 
                label=emotion.capitalize(), 
                color=colors[emotion],
                linewidth=1.5
            )
            
    ax.set_title(f'Sentence Emotional Curves (Rolling Window = {window_size}): {title_of_work}', weight='bold', fontsize=12, pad=15)
    ax.set_xlabel('Sentence Sequence (Narrative Time)', fontsize=10)
    ax.set_ylabel('Smoothed Emotion Intensity', fontsize=10)
    ax.set_xlim(1, len(df_sentences))
    ax.legend(title='Emotions', loc='upper right', frameon=True)
    
    os.makedirs(output_folder, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"{title_of_work}_sentences_curves.png"), bbox_inches='tight')
    plt.close()


def cyrillic_to_latin(text):
    """
    Helper function to convert Serbian Cyrillic text to Latin.
    """
    if not isinstance(text, str):
        return text
    
    cyr_to_lat_map = {
        'А': 'A', 'Б': 'B', 'В': 'V', 'Г': 'G', 'Д': 'D', 'Ђ': 'Đ', 'Е': 'E', 'Ж': 'Ž', 'З': 'Z', 'И': 'I',
        'Ј': 'J', 'К': 'K', 'Л': 'L', 'Љ': 'Lj', 'М': 'M', 'Н': 'N', 'Њ': 'Nj', 'О': 'O', 'П': 'P', 'Р': 'R',
        'С': 'S', 'Т': 'T', 'Ћ': 'Ć', 'У': 'U', 'Ф': 'F', 'Х': 'H', 'Ц': 'C', 'Ч': 'Č', 'Џ': 'Dž', 'Ш': 'Š',
        'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'ђ': 'đ', 'е': 'e', 'ж': 'ž', 'з': 'z', 'и': 'i',
        'ј': 'j', 'к': 'k', 'л': 'l', 'љ': 'lj', 'м': 'm', 'н': 'n', 'њ': 'nj', 'о': 'o', 'п': 'p', 'р': 'r',
        'с': 's', 'т': 't', 'ћ': 'ć', 'у': 'u', 'ф': 'f', 'х': 'h', 'ц': 'c', 'ч': 'č', 'џ': 'dž', 'ш': 'š'
    }
    
    text_converted = ""
    for char in text:
        text_converted += cyr_to_lat_map.get(char, char)
    return text_converted

def get_clean_sentence_from_xml(xml_path, s_id):
    """
    Locates the sentence in the TEI XML file by ID and reconstructs 
    the original raw text using word/punctuation tokens and join rules.
    """
    if not os.path.exists(xml_path):
        return None
        
    try:
        with open(xml_path, 'r', encoding='utf-8', errors='ignore') as f:
            xml_content = f.read()
        
        # Regex extraction for the <s>...</s> block
        pattern = rf"<(?:tei:)?s\s+[^>]*?(?:xml:)?id=['\"]{s_id}['\"][^>]*?>(.*?)</(?:tei:)?s>"
        match = re.search(pattern, xml_content, re.DOTALL)
        
        if match:
            inner_xml = f"<root>{match.group(1)}</root>"
            parser = etree.XMLParser(recover=True, encoding='utf-8')
            s_tree = etree.fromstring(inner_xml, parser)
            
            words_tokens = []
            for el in s_tree.xpath("//w | //pc"):
                token_text = el.text if el.text else ""
                join_attr = el.get("join")
                if token_text:
                    words_tokens.append((token_text, join_attr))
            
            sentence_bits = []
            for i, (tok, join) in enumerate(words_tokens):
                sentence_bits.append(tok)
                if i < len(words_tokens) - 1 and join != "right":
                    sentence_bits.append(" ")
                    
            return "".join(sentence_bits).strip()
    except Exception:
        pass
    return None

def find_highest_emotional_climax(json_folder, xml_folder, top_n=5, min_words=25, min_emo_words=4):
    """
    Traverses all JSON files, filters passages using a composite emotional score,
    and pulls the authentic sentence text directly from the original ELTeC XML files.
    """
    all_filtered_passages = []
    
    for file in os.listdir(json_folder):
        if file.endswith('_analysis.json') or file.endswith('_emo_analiza.json'):
            file_path = os.path.join(json_folder, file)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Match XML filename dynamically
            xml_file_name = data.get('file_name')
            if not xml_file_name:
                xml_file_name = file.replace('_analysis.json', '.xml').replace('_emo_analiza.json', '.xml')
            
            for ch in data.get('chapters', []):
                ch_id = ch.get('chapter_id', 'Unknown_CH')
                
                for p in ch.get('passages', []):
                    total_words = p.get('total_words', 0)
                    emo_words = p.get('emotional_word_count', 0)
                    density = p.get('emotional_density', 0.0)
                    
                    if total_words >= min_words and emo_words >= min_emo_words:
                        climax_score = density * emo_words
                        
                        p_copy = p.copy()
                        p_copy['xml_file'] = xml_file_name
                        p_copy['chapter_id'] = ch_id
                        p_copy['climax_score'] = climax_score
                        all_filtered_passages.append(p_copy)
            
    sorted_passages = sorted(all_filtered_passages, key=lambda x: x['climax_score'], reverse=True)
    
    # FIXED: Header updated with the requested layout detailing specific boundaries
    print(f"\n========================================================================")
    print(f"TOP {top_n} PASSAGES WITH THE HIGHEST EMOTIONAL CLIMAX IN THE CORPUS")
    print(f"Criteria: Min {min_words} words (passage) & Min {min_emo_words} emotional words (passage), Min 3 words (sentence)")
    print(f"========================================================================")
    
    if not sorted_passages:
        print("   [INFO] No passages match the strict filtering criteria.")
        return
        
    for i, p in enumerate(sorted_passages[:top_n]):
        print(f"\nRANK {i+1}: Novel: {p['xml_file']} | Chapter ID: {p['chapter_id']}")
        print(f"   -> Passage ID: {p.get('passage_id', 'N/A')}")
        print(f"   -> Emotional Climax Score: {p['climax_score']:.2f}")
        print(f"   -> Density: {p['emotional_density']}% ({p['emotional_word_count']}/{p['total_words']} words)")
        
        sentences = p.get('sentences', [])
        valid_sentences = []
        for s in sentences:
            details = s.get('word_details', s.get('detalji_reci', []))
            if len(details) >= 3:  # Enforces Min 3 words condition
                valid_sentences.append(s)
        
        if valid_sentences:
            strongest_s = max(valid_sentences, key=lambda x: x.get('emotional_density', 0))
            s_id = strongest_s.get('sentence_id', 'N/A')
            print(f"   -> Key Sentence ({s_id}): Density {strongest_s.get('emotional_density', 0)}%")
            
            # FIXED: Extraction from original XML instead of raw word concatenation
            xml_path = os.path.join(xml_folder, p['xml_file'])
            original_text = get_clean_sentence_from_xml(xml_path, s_id)
            
            # Fallback if XML file is missing or sentence ID doesn't match
            if not original_text:
                details = strongest_s.get('word_details', strongest_s.get('detalji_reci', []))
                original_text = " ".join([d['word'] for d in details]) + " [JSON fallback]"
            
            latin_sentence_text = cyrillic_to_latin(original_text)
            print(f'      Text: "{latin_sentence_text}"')
            
            # Lemmas still pulled from JSON details for easy insight
            details = strongest_s.get('word_details', strongest_s.get('detalji_reci', []))
            latin_lemmas = [cyrillic_to_latin(d['lemma']) for d in details]
            print(f"      Emotional Lemmas: {', '.join(latin_lemmas[:15])}...")
        else:
            print("   -> Key Sentence: [No sentence in this passage met the minimum 3-word length requirement]")
            
        print("-" * 70)
