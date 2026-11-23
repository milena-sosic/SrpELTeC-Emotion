import os
import xml.etree.ElementTree as ET


def analyze_tei_corpus_segments(corpus_folder):
    # Standard TEI namespace.
    ns = {'tei': 'http://www.tei-c.org/ns/1.0'}

    # Initialize all counters.
    total_novels = 0
    total_words = 0
    total_punctuation = 0
    total_sentences = 0
    total_pages = 0
    total_chapters = 0
    total_passages = 0
    total_poems = 0
    total_verse_lines = 0
    total_foreign_segments = 0
    total_mentioned_works = 0

    print(
        f"[START] Running detailed TEI analysis "
        f"(including words and sentences) in: '{corpus_folder}'...\n"
    )

    if not os.path.exists(corpus_folder):
        print(f"[ERROR] Folder '{corpus_folder}' does not exist!")
        return

    for file in os.listdir(corpus_folder):
        if file.endswith('.xml'):
            xml_path = os.path.join(corpus_folder, file)
            total_novels += 1

            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
            except ET.ParseError:
                print(f"   [ERROR] Failed to parse file: {file}")
                continue

            # --- 1. COUNT SENTENCES (<s>) ---
            file_sentence_count = len(root.findall('.//tei:s', ns))
            # If <s> is not used, check for <seg type="sentence"> or plain <seg>.
            if file_sentence_count == 0:
                file_sentence_count = len(root.findall('.//tei:seg', ns))
            total_sentences += file_sentence_count

            # --- 2. COUNT WORDS (<w> or a text split) ---
            file_word_count = len(root.findall('.//tei:w', ns))

            # Fallback for documents without tokenized <w> elements.
            if file_word_count == 0:
                # Extract plain text from the document body, excluding TEI header metadata.
                body = root.find('.//tei:text', ns)
                if body is not None:
                    body_text = "".join(body.itertext())
                    file_word_count = len(body_text.split())
                else:
                    # If there is no <text> element, use text from the entire root.
                    file_word_count = len("".join(root.itertext()).split())

            total_words += file_word_count

            # --- 3. OTHER TEI STRUCTURES ---
            total_pages += len(root.findall('.//tei:pb', ns))

            chapters = root.findall('.//tei:div[@type="chapter"]', ns)
            if len(chapters) == 0:
                chapters = root.findall('.//tei:div[@type="ch"]', ns)
            total_chapters += len(chapters)

            total_passages += len(root.findall('.//tei:p', ns))
            # total_poems += len(root.findall('.//tei:lg', ns))
            # total_verse_lines += len(root.findall('.//tei:l', ns))
            # A) Standard TEI tag for a group of verse lines.
            poem_groups = root.findall('.//tei:lg', ns)

            # B) Alternative: div elements explicitly marked as poems.
            poem_divisions = root.findall('.//tei:div[@type="poem"]', ns) or root.findall(
                './/tei:div[@type="poetry"]', ns
            )

            # C) Alternative: paragraphs containing verse lines.
            verse_paragraphs = root.findall('.//tei:p[@type="verse"]', ns) or root.findall(
                './/tei:p[@rend="verse"]', ns
            )

            # Add all detected poem forms.
            total_poems += len(poem_groups) + len(poem_divisions) + len(verse_paragraphs)

            # --- COUNT VERSE LINES ---
            verse_lines = root.findall('.//tei:l', ns)

            # If there are no standard <l> tags and poems use <p type="verse">,
            # treat the <s> elements inside those paragraphs as verse lines.
            if len(verse_lines) == 0 and len(verse_paragraphs) > 0:
                for verse_paragraph in verse_paragraphs:
                    verse_lines += verse_paragraph.findall('.//tei:s', ns)

            total_verse_lines += len(verse_lines)

            # --- COUNT PUNCTUATION (<pc>) ---
            total_punctuation += len(root.findall('.//tei:pc', ns))
            total_foreign_segments += len(root.findall('.//tei:foreign', ns))
            total_mentioned_works += len(root.findall('.//tei:title', ns))

    # --- FINAL DETAILED CORPUS REPORT ---
    print("=" * 50)
    print(" DETAILED TEI CORPUS SUMMARY")
    print("=" * 50)
    print(f"Novels:            {total_novels}")
    print(f"Words:             {total_words:,}".replace(",", "."))
    print(f"Punctuation:       {total_punctuation:,}".replace(",", "."))
    print(f"Sentences:         {total_sentences:,}".replace(",", "."))
    print(f"Pages:             {total_pages:,}".replace(",", "."))
    print(f"Chapters:          {total_chapters:,}".replace(",", "."))
    print(f"Passages:          {total_passages:,}".replace(",", "."))
    print(f"Poems:             {total_poems:,}".replace(",", "."))
    print(f"Verse lines:       {total_verse_lines:,}".replace(",", "."))
    print(f"Foreign segments:  {total_foreign_segments:,}".replace(",", "."))
    print(f"Mentioned works:   {total_mentioned_works:,}".replace(",", "."))
    print("=" * 50)


if __name__ == "__main__":
    # Set the path to the corpus XML files.
    CORPUS_PATH = "./data/level2"
    analyze_tei_corpus_segments(CORPUS_PATH)
