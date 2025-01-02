# arXiv to Obsidian Converter

A tool to convert arXiv papers into Obsidian notes with automatic PDF processing and metadata extraction.

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/arxiv_to_obsidian.git
cd arxiv_to_obsidian
```

2. Install the required system dependencies:
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install tesseract-ocr poppler-utils

# macOS
brew install tesseract poppler
```

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Basic usage:
```bash
python main.py 2412.16003
```

Specify output directory:
```bash
python main.py 2412.16003 -o /path/to/output
```

## Features

- Fetches paper metadata from arXiv
- Downloads PDF files
- Extracts citation count from Semantic Scholar
- Processes PDF content with OCR support
- Creates well-formatted Obsidian notes
- Automatic section identification

## Project Structure

```
arxiv_to_obsidian/
├── requirements.txt
├── README.md
├── scripts/
│   ├── __init__.py
│   ├── arxiv_fetcher.py     # arXiv paper related functionality
│   ├── pdf_processor.py     # PDF processing
│   └── note_creator.py      # Obsidian note creation
├── main.py                  # Main execution file
└── tests/                   # Test files (optional)
    └── __init__.py
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.