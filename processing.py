import os
import csv
import json
import xml.etree.ElementTree as ET
import cyrtranslit
from utils import EMOTION_COLORS

# --- STEP 1: LOAD EMOTIONAL LEXICON WITH POS MATCHING ---
def load_emolex(json_path):
    """
    Loads EmoLex.SR from JSON and maps unique lemma+PoS combinations 
    to their emotional weight vectors and primary labels.
    """
    lexicon = {}
    with open(json_path, mode='r', encoding='utf-8') as f:
        data = json.load(f)
        
    for entry in data.get('entries', []):
        lemma = entry['lemma'].strip().lower()
        pos = entry.get('pos', '').strip().lower()
        
        # Build composite unique key to guarantee grammatical accuracy (lowercase unified)
        lexicon_key = f"{lemma}_{pos}" if pos else lemma
        
        # Filter out neutral weights since they do not contribute to dynamic emotional charge
        weights_clean = {k: float(v) for k, v in entry['weights'].items() if k != 'neutral'}
        lexicon[lexicon_key] = {
            'weights': weights_clean,
            'label': entry['label']
        }
    return lexicon


# --- STEP 2: UTILITY METRIC FUNCTIONS ---
def initialize_emotions():
    """Returns a baseline zeroed dictionary for tracking Plutchik's 8 core emotions."""
    return {
        'anger': 0.0, 'anticipation': 0.0, 'disgust': 0.0, 'fear': 0.0,
        'joy': 0.0, 'surprise': 0.0, 'sadness': 0.0, 'trust': 0.0
    }


def normalize_scores(scores, emotional_word_count):
    """Normalizes raw aggregated intensity weights by dividing by total matched tokens."""
    if emotional_word_count == 0:
        return {k: 0.0 for k in scores.keys()}
    return {k: round(v / emotional_word_count, 4) for k, v in scores.items()}


def analyze_token(lemma_cyr, pos, lexicon):
    """Translates Cyrillic lemmas to Latin and runs a specific lemma+PoS lookup against the lexicon."""
    if not lemma_cyr:
        return None
    
    # Sigurna konverzija ćirilice u latinicu i svođenje na mala slova
    lemma_lat = cyrtranslit.to_latin(lemma_cyr.strip(), 'sr').lower()
    pos_clean = pos.strip().lower() if pos else ""
    
    # Attempt strict match using compound identifier (e.g., reka_noun)
    composite_key = f"{lemma_lat}_{pos_clean}" if pos_clean else lemma_lat
    if composite_key in lexicon:
        return lexicon[composite_key]
        
    # Fallback to lemma-only lookup if specific structural PoS tag variations miss
    return lexicon.get(lemma_lat, None)


def generate_css_gradient(weights):
    """Generates a dynamic multi-colored CSS linear-gradient for underlined micro-visualizations."""
    active_emotions = {k: v for k, v in weights.items() if v > 0}
    if not active_emotions:
        return "border-bottom: 2px solid #ccc;"
    
    total_weight = sum(active_emotions.values())
    gradient_parts = []
    current_percentage = 0.0
    
    for emo, weight in active_emotions.items():
        percentage = (weight / total_weight) * 100
        color = EMOTION_COLORS.get(emo, '#333')
        gradient_parts.append(f"{color} {current_percentage}%")
        current_percentage += percentage
        gradient_parts.append(f"{color} {current_percentage}%")
        
    gradient_str = ", ".join(gradient_parts)
    return f"border-bottom: 4px solid; border-image: linear-gradient(to right, {gradient_str}) 1;"


# --- STEP 3: TEI/XML STRUCTURAL ANALYSIS ENGINE ---
def analyze_emotion_xml(xml_path, lexicon, output_folder):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError:
        print(f"[ERROR] XML Parsing failed for file: {xml_path}")
        return None

    file_name = os.path.basename(xml_path)
    
    global_raw_scores = initialize_emotions()
    total_emotional_words_doc = 0
    total_words_doc = 0

    # DINAMIČKA DETEKCIJA NAMESPACE-A
    ns_uri = ""
    if root.tag.startswith('{'):
        ns_uri = root.tag.split('}')[0].strip('{')
    namespaces = {'ns': ns_uri} if ns_uri else {}

    def clean_tag(node):
        return node.tag.split('}')[-1] if '}' in node.tag else node.tag

    def get_xpath(tag_name, deep=True):
        prefix = ".//" if deep else "./"
        if ns_uri:
            return f"{prefix}ns:{tag_name}"
        return f"{prefix}{tag_name}"

    chapters_analysis = []
    html_content = []
    
    ch_idx = 0
    p_idx = 0
    s_idx = 0
    
    # Pronalaženje svih strukturnih div elemenata pod odgovarajućim namespace-om
    chapters = root.findall(get_xpath('div'), namespaces)
    if not chapters:
        chapters = [root]

    for el_div in chapters:
        if 'div' in clean_tag(el_div) or el_div == root:
            # Procesiramo samo poglavlja (ako XML ima eksplicitan type='chapter' ili ako nema div-ova uopšte)
            if el_div.get('type') == 'chapter' or len(chapters) == 1:
                current_chapter_raw_scores = initialize_emotions()
                current_chapter_emo_words = 0
                current_chapter_total_words = 0
                chapter_passages = []
                chapter_html_content = []
                
                ch_id = el_div.get('{http://www.w3.org/XML/1998/namespace}id') or el_div.get('xml:id') or f"ch_{ch_idx}"
                
                current_chapter_title = ""
                el_head = el_div.find(get_xpath('head', deep=False), namespaces)
                if el_head is not None:
                    current_chapter_title = "".join(el_head.itertext()).strip()
                else:
                    current_chapter_title = f"Chapter {ch_id}"
                
                #html_content.append(f"<h2 class='chapter-title'>{current_chapter_title} ({ch_id})</h2>")
                
                # Izolujemo pasuse koji se nalaze isključivo unutar ovog poglavlja
                passages = el_div.findall(get_xpath('p'), namespaces)
                
                for element in passages:
                    passage_id = element.get('{http://www.w3.org/XML/1998/namespace}id') or element.get('xml:id') or f"p_{p_idx}"
                    passage_raw_scores = initialize_emotions()
                    passage_emo_words = 0
                    passage_total_words = 0
                    sentences_analysis = []
                    
                    # html_content.append("  <p class='paragraph'>")
                    chapter_html_content.append("  <p class='paragraph'>")

                    # Izolujemo rečenice unutar trenutnog pasusa
                    sentences = element.findall(get_xpath('s'), namespaces)
                    for el_s in sentences:
                        sentence_id = el_s.get('{http://www.w3.org/XML/1998/namespace}id') or el_s.get('xml:id') or f"s_{s_idx}"
                        sentence_raw_scores = initialize_emotions()
                        sentence_emo_words = 0
                        sentence_total_words = 0
                        word_details = []

                        for el_child in el_s.findall('.//*'):
                            child_tag = clean_tag(el_child)
                            
                            if child_tag == 'w':
                                sentence_total_words += 1
                                lemma_cyr = el_child.get('lemma')
                                pos_tag = el_child.get('pos') or el_child.get('ana') or ""
                                word_text = el_child.text if el_child.text else ""
                                spacing = "" if el_child.get('join') == 'left' else " "
                                
                                emo_data = analyze_token(lemma_cyr, pos_tag, lexicon)
                                
                                if emo_data and any(v > 0 for v in emo_data['weights'].values()):
                                    sentence_emo_words += 1
                                    weights = emo_data['weights']
                                    
                                    css_style = generate_css_gradient(weights)
                                    component_breakdown = [f"{k}: {v}" for k, v in weights.items() if v > 0]
                                    tooltip = f"Lemma: {lemma_cyr} ({pos_tag}) | Profile: {', '.join(component_breakdown)}"
                                    
                                    # html_content.append(f"{spacing}<span class='emo-word' style='{css_style}' title='{tooltip}'>{word_text}</span>")
                                    chapter_html_content.append(
                                        f"{spacing}<span class='emo-word' style='{css_style}' "
                                        f"title='{tooltip}'>{word_text}</span>"
                                    )
                                    word_details.append({
                                        'word': word_text,
                                        'lemma': lemma_cyr,
                                        'pos': pos_tag,
                                        'weights': weights
                                    })
                                    
                                    for emo, weight in weights.items():
                                        sentence_raw_scores[emo] += weight
                                else:
                                    # html_content.append(f"{spacing}{word_text}")
                                    chapter_html_content.append(f"{spacing}{word_text}")
                                    
                            elif child_tag == 'pc':
                                pc_text = el_child.text if el_child.text else ""
                                spacing = "" if el_child.get('join') == 'right' else " "
                                # html_content.append(f"{spacing}{pc_text}")
                                chapter_html_content.append(f"{spacing}{pc_text}")

                        sentence_total_words_safe = sentence_total_words if sentence_total_words > 0 else 1
                        s_density = round((sentence_emo_words / sentence_total_words_safe) * 100, 2)
                        
                        sentences_analysis.append({
                            'sentence_id': sentence_id,
                            'scores': normalize_scores(sentence_raw_scores, sentence_emo_words),
                            'total_words': sentence_total_words,
                            'emotional_word_count': sentence_emo_words,
                            'emotional_density': s_density,
                            'word_details': word_details
                        })
                        
                        passage_emo_words += sentence_emo_words
                        passage_total_words += sentence_total_words
                        for emo in passage_raw_scores:
                            passage_raw_scores[emo] += sentence_raw_scores[emo]
                        s_idx += 1

                    # html_content.append("  </p>\n")
                    chapter_html_content.append("  </p>\n")

                    if passage_total_words > 0:
                        p_density = round((passage_emo_words / passage_total_words) * 100, 2)
                        chapter_passages.append({
                            'passage_id': passage_id,
                            'scores': normalize_scores(passage_raw_scores, passage_emo_words),
                            'total_words': passage_total_words,
                            'emotional_word_count': passage_emo_words,
                            'emotional_density': p_density,
                            'sentences': sentences_analysis
                        })
                        
                        current_chapter_emo_words += passage_emo_words
                        current_chapter_total_words += passage_total_words
                        for emo in current_chapter_raw_scores:
                            current_chapter_raw_scores[emo] += passage_raw_scores[emo]
                    p_idx += 1

                chapter_html = f"""
                <details class="chapter-details">
                    <summary class="chapter-summary">
                        <h2>{current_chapter_title} ({ch_id})</h2>
                    </summary>

                    <div class="chapter-content">
                        {"".join(chapter_html_content)}
                    </div>
                </details>
                """

                html_content.append(chapter_html)
                
                if current_chapter_total_words > 0:
                    chapters_analysis.append({
                        'chapter_id': ch_id,
                        'chapter_title': current_chapter_title,
                        'scores': normalize_scores(current_chapter_raw_scores, current_chapter_emo_words),
                        'total_words': current_chapter_total_words,
                        'emotional_word_count': current_chapter_emo_words,
                        'emotional_density': round((current_chapter_emo_words / current_chapter_total_words) * 100, 2),
                        'passages': chapter_passages
                    })
                ch_idx += 1

    # Globalna agregacija na nivou celog dokumenta
    for ch in chapters_analysis:
        total_words_doc += ch['total_words']
        total_emotional_words_doc += ch['emotional_word_count']
        for emo in global_raw_scores:
            global_raw_scores[emo] += (ch['scores'][emo] * ch['emotional_word_count']) if ch['emotional_word_count'] > 0 else 0

    global_density = round((total_emotional_words_doc / total_words_doc) * 100, 2) if total_words_doc > 0 else 0
    final_normalized_scores = normalize_scores(global_raw_scores, total_emotional_words_doc)

    # Čuvanje HTML vizuelizacije
    html_output_path = os.path.join(output_folder, file_name.replace('.xml', '_emotions.html'))
    save_html_report(html_output_path, file_name, html_content, total_words_doc, total_emotional_words_doc, global_density, final_normalized_scores)

    return {
        'file_name': file_name,
        'total_words': total_words_doc,
        'total_emotional_words': total_emotional_words_doc,
        'emotional_density': global_density,
        'scores': final_normalized_scores,
        'chapters': chapters_analysis
    }


def save_html_report(
    path,
    file_name,
    html_content,
    total,
    emotional,
    density,
    scores
):
    """
    Saves an interactive HTML report for emotion analysis.

    The generated report includes:
    - global corpus/document statistics;
    - normalized emotion scores;
    - an emotion-color legend;
    - collapsible chapter sections, provided that html_content
      contains <details> and <summary> elements;
    - highlighted emotional words with tooltip support.
    """

    styled_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">

    <title>Emotion Analysis: {file_name}</title>

    <style>
        body {{
            font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 30px;
            color: #333;
            background-color: #fcfcfc;
        }}

        h1 {{
            color: #2c3e50;
            border-bottom: 2px solid #eee;
            padding-bottom: 10px;
        }}

        /* Collapsible chapter container */
        .chapter-details {{
            background: #f4f6f7;
            margin-bottom: 15px;
            border-radius: 8px;
            border: 1px solid #eaeded;
            overflow: hidden;
        }}

        /* Clickable chapter header */
        .chapter-summary {{
            padding: 10px 20px;
            background: #e5e8e8;
            cursor: pointer;
            list-style-position: inside;
        }}

        .chapter-summary:hover {{
            background: #dfe4e6;
        }}

        .chapter-summary h2 {{
            display: inline-block;
            margin: 0;
            font-size: 1.3em;
            color: #2c3e50;
        }}

        /* Content displayed when a chapter is expanded */
        .chapter-content {{
            padding: 20px;
            background: #f4f6f7;
        }}

        /* Global metrics */
        .metrics-container {{
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 25px;
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
        }}

        .metric-box {{
            padding: 10px 15px;
            background: white;
            border-radius: 4px;
        }}

        /* Emotion legend */
        .legend {{
            margin-bottom: 30px;
            padding: 10px;
            background: #fff;
            border: 1px dashed #ccc;
            border-radius: 5px;
        }}

        .legend-item {{
            display: inline-block;
            padding: 3px 10px;
            margin: 3px 10px 3px 0;
            border-radius: 3px;
            font-weight: bold;
            font-size: 0.9em;
        }}

        /* Text passage / paragraph */
        .passage,
        .paragraph {{
            margin-bottom: 20px;
            text-align: justify;
            font-size: 1.1em;
            background: white;
            padding: 15px;
            border-radius: 5px;
        }}

        /* Highlighted emotion word */
        .emo-word {{
            border-bottom: 1px solid #999;
            cursor: help;
            border-radius: 3px;
            padding: 0 2px;
            font-weight: 500;
        }}
    </style>
</head>

<body>

    <h1>Visualization of Identified Emotions: {file_name}</h1>

    <div class="metrics-container">

        <div class="metric-box">
            <strong>Total Words:</strong> {total}
        </div>

        <div class="metric-box">
            <strong>Emotional Words:</strong> {emotional}
        </div>

        <div class="metric-box">
            <strong>Global Density:</strong> {density}%
        </div>

        <div class="metric-box">
            <strong>
                Average Intensity per Emotional Word
                (Normalized):
            </strong>
            <br>

            anger: {scores['anger']} |
            anticipation: {scores['anticipation']} |
            disgust: {scores['disgust']} |
            fear: {scores['fear']} |
            joy: {scores['joy']} |
            sadness: {scores['sadness']} |
            surprise: {scores['surprise']} |
            trust: {scores['trust']}
        </div>

    </div>

    <div class="legend">
        <strong>Legend:</strong>

        <span
            class="legend-item"
            style="background-color: {EMOTION_COLORS['anger']};">
            anger
        </span>

        <span
            class="legend-item"
            style="background-color: {EMOTION_COLORS['anticipation']};">
            anticipation
        </span>

        <span
            class="legend-item"
            style="background-color: {EMOTION_COLORS['disgust']};">
            disgust
        </span>

        <span
            class="legend-item"
            style="background-color: {EMOTION_COLORS['fear']};">
            fear
        </span>

        <span
            class="legend-item"
            style="background-color: {EMOTION_COLORS['joy']};">
            joy
        </span>

        <span
            class="legend-item"
            style="background-color: {EMOTION_COLORS['sadness']};">
            sadness
        </span>

        <span
            class="legend-item"
            style="background-color: {EMOTION_COLORS['surprise']};">
            surprise
        </span>

        <span
            class="legend-item"
            style="background-color: {EMOTION_COLORS['trust']};">
            trust
        </span>
    </div>

    <div class="text-content">
        {"".join(html_content)}
    </div>

</body>
</html>
"""

    with open(path, "w", encoding="utf-8") as file:
        file.write(styled_html)


# --- STEP 4: CORPORA RUNTIME PIPELINE ---
def run_emotion_analysis(corpus_folder, emolex_json_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    print("[INFO] Loading EmoLex JSON Lexicon Registry...")
    lexicon = load_emolex(emolex_json_path)
    print(f"[SUCCESS] Loaded {len(lexicon)} distinct emotional mapping components.\n")

    summary_records = []

    print("[START] Executing structural text parser across documents...")
    for file in os.listdir(corpus_folder):
        if file.endswith('.xml'):
            xml_path = os.path.join(corpus_folder, file)
            print(f"  Processing File: {file}...")
            
            document_results = analyze_emotion_xml(xml_path, lexicon, output_folder)
            
            if document_results:
                json_filename = file.replace('.xml', '_analysis.json')
                with open(os.path.join(output_folder, json_filename), 'w', encoding='utf-8') as jf:
                    json.dump(document_results, jf, indent=4, ensure_ascii=False)
                
                doc_scores = document_results['scores']
                summary_records.append({
                    'file_name': document_results['file_name'],
                    'total_words': document_results['total_words'],
                    'total_emotional_words': document_results['total_emotional_words'],
                    'emotional_density': f"{document_results['emotional_density']}%",
                    'anger': doc_scores['anger'],
                    'anticipation': doc_scores['anticipation'],
                    'disgust': doc_scores['disgust'],
                    'fear': doc_scores['fear'],
                    'joy': doc_scores['joy'],
                    'sadness': doc_scores['sadness'],
                    'surprise': doc_scores['surprise'],
                    'trust': doc_scores['trust']
                })

    tsv_output_path = os.path.join(output_folder, "corpus_global_emotion_report.tsv")
    if summary_records:
        columns = ['file_name', 'total_words', 'total_emotional_words', 'emotional_density',
                   'anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust']
        with open(tsv_output_path, mode='w', encoding='utf-8', newline='') as cf:
            writer = csv.DictWriter(cf, fieldnames=columns, delimiter='\t')
            writer.writeheader()
            writer.writerows(summary_records)
            
        print(f"\n[SUCCESS] Pipeline metrics and visual maps successfully generated inside: '{output_folder}/'")


if __name__ == "__main__":
    CORPUS_DIRECTORY = "./data/level2"
    EMOLEX_LEXICON_PATH = "./data/EmoLex.SR-v2.json"
    OUTPUT_FOLDER = "processed_data"
    run_emotion_analysis(CORPUS_DIRECTORY, EMOLEX_LEXICON_PATH, OUTPUT_FOLDER)
