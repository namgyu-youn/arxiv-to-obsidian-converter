# arXiv to Obsidian Converter
This is **not certified obsidian-plugin**. I suffured from too many Bugs(🐛) using Obsidian, so made simple text extractor using OCR.
- See also : [obsidian-citation-plugin](https://github.com/hans/obsidian-citation-plugin), [arxiv-assistant](https://github.com/Puer-Hyun/arxiv-assistant).

Converts arXiv papers into well-formatted Obsidian notes. It downloads papers using arXiv IDs, processes their content, and creates structured notes with metadata, summaries, and key points.

With extracted content, you can do anything you want! (ex. Summarization using llm, Post somewhere, .. etc)

**⬇️ Sample Output ⬇️**

<img width="900" alt="image" src="https://github.com/Namgyu-Youn/arxiv-to-obsidian-converter/blob/main/src/sample_obsidian.png">

<img width="900" alt="image" src="https://github.com/Namgyu-Youn/arxiv-to-obsidian-converter/blob/main/src/sample_output.png">

<img width="900" alt="image" src="https://github.com/Namgyu-Youn/arxiv-to-obsidian-converter/blob/main/src/sample_terminal.png">




## ✨ Features

- Downloads papers directly **from arXiv** using paper IDs
- **Extracts citation** using Semantic Scholar
- Processes PDF content with **OCR support**
- Creates well-structured **Obsidian notes** with:
    - Paper metadata
    - Citation information
    - Research field categorization
    - Key points section
    - Personal notes template
    - Formatted content with proper paragraph breaks


## 🚩 How to use?
### Step 1. Clone the repository
```bash
git clone https://github.com/Namgyu-Youn/arxiv-to-obsidian.git
cd arxiv-to-obsidian
docker build .
```
### Step 2. Just run poetry!
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

## 👥 Contribution guide
Thanks for your interest. I always enjoy meaningful collaboration. <br/>
Do you have any question or bug?? Then please submit **ISSUE**!
