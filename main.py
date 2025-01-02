import argparse
from pathlib import Path
import re
from scripts.arxiv_fetcher import ArxivPaper
from scripts.pdf_processor import PDFProcessor
from scripts.note_creator import create_obsidian_note

def main():
    parser = argparse.ArgumentParser(description='Convert arXiv paper to Obsidian note')
    parser.add_argument('arxiv_id', help='arXiv paper ID (e.g., 2101.12345)')
    parser.add_argument('--output', '-o', help='Output directory', default='.')
    args = parser.parse_args()

    try:
        # Create output directory
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)

        # Fetch and download paper
        print(f"Fetching paper {args.arxiv_id} from arXiv...")
        paper = ArxivPaper(args.arxiv_id)
        metadata = paper.fetch_metadata()
        pdf_path = paper.download_pdf(output_path)

        if not pdf_path:
            raise Exception("Failed to download PDF")

        # Process PDF content
        print("Processing PDF content...")
        pdf_processor = PDFProcessor(pdf_path)
        sections = pdf_processor.identify_sections()

        # Create note
        print("Creating Obsidian note...")
        note_content = create_obsidian_note(metadata, sections)

        # Save note
        safe_title = re.sub(r'[<>:"/\\|?*]', '', metadata['title'])
        safe_title = safe_title.replace(' ', '_')
        note_file = output_path / f"{safe_title}.md"

        with open(note_file, 'w', encoding='utf-8') as f:
            f.write(note_content)

        print(f"Successfully created:")
        print(f"- Note: {note_file}")
        print(f"- PDF: {pdf_path}")

    except Exception as e:
        print(f"Error: {str(e)}")
        return 1

    return 0

if __name__ == "__main__":
    main()