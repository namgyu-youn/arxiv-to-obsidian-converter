# arXiv to Obsidian Converter
This is **certified obsidian-plugin**. I suffured from Bug, so made simple text extractor using OCR.
- See also : [obsidian-citation-plugin](https://github.com/hans/obsidian-citation-plugin), [arxiv-assistant](https://github.com/Puer-Hyun/arxiv-assistant).

Automatically converts arXiv papers into well-formatted Obsidian notes. It downloads papers using arXiv IDs, processes their content, and creates structured notes with metadata, summaries, and key points.

⬇️ Sample Output ⬇️

<img width="900" alt="image" src="https://github.com/Namgyu-Youn/arxiv-to-obsidian-converter/main/src/sample_output.png">

<img width="900" alt="image" src="https://github.com/Namgyu-Youn/arxiv-to-obsidian-converter/main/src/sample_terminal.png">

<img width="900" alt="image" src="https://github.com/Namgyu-Youn/arxiv-to-obsidian-converter/main/src/sample_obsidian.png">




## ✨ Features

- Downloads papers directly from arXiv using paper IDs
- Extracts citation counts from Semantic Scholar
- Processes PDF content with **OCR support**
- Creates well-structured **Obsidian notes** with:
    - Paper metadata
    - Citation information
    - Research field categorization
    - Key points section
    - Personal notes template
    - Formatted content with proper paragraph breaks



## ➕ Prerequisites

- Python 3.12 or higher
- Tesseract : OCR
- Poetry, Docker

## 🚩 How to use?
### Step 1. Clone the repository
```bash
git clone https://github.com/yourusername/arxiv-to-obsidian.git
cd arxiv-to-obsidian
```

### Step 2. Install system dependencies

``` bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install tesseract-ocr poppler-utils

# macOS
brew install tesseract poppler
Install Python dependencies
bashCopypip install -r requirements.txt
```

### Step 3: Using Docker
```bash
docker build .
```

### Step 4. Run poetry!
```
poetry run arxiv-to-obsidian 2304.08485 # Just Input arXiv ID!
```


```
📁 Project Structure
Copyarxiv_to_obsidian/
├── requirements.txt
├── README.md
├── scripts/
│   ├── __init__.py
│   ├── arxiv_fetcher.py     # arXiv paper functionality
│   ├── pdf_processor.py     # PDF processing
│   └── note_creator.py      # Obsidian note creation
├── main.py                  # Main execution file
└── tests/                   # Test files
    └── __init__.py
```