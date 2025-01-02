import re
from typing import Dict

def create_obsidian_note(arxiv_data: Dict, pdf_content: Dict) -> str:
    """Create Obsidian note with enhanced formatting."""
    authors = ", ".join(arxiv_data['authors'])
    pub_date = arxiv_data['published'].strftime("%Y-%m-%d")
    citation_display = "Null" if arxiv_data['citations'] is None else arxiv_data['citations']

    # Clean and format the abstract
    summary = re.sub(r'\s+', ' ', arxiv_data['summary'])
    summary = re.sub(r'([.!?])\s+([A-Z])', r'\1\n\2', summary)

    note = f"""# {arxiv_data['title']}

## 📚 Metadata
**Authors**: {authors}
**Publication Date**: {pub_date}
**arXiv ID**: {arxiv_data['links'][0].split('/')[-1]}
**Citations**: {citation_display}
**Research Fields**: {arxiv_data['categories']}
**PDF Link**: {arxiv_data['pdf_url']}

## 🎯 Key Points
### Research Objective
-
### Methodology
-
### Main Results
-
### Key Findings
-

## 💭 Personal Notes
### Strengths
-
### Limitations
-
### Future Research Directions
-

## 📝 Abstract
{summary.strip()}

## 📖 Content"""

    # Add sections with enhanced formatting
    for section_name, content in pdf_content.items():
        if content:
            note += f"\n### {section_name.title()}\n{content.strip()}"

    note += """

## 🔗 Related Papers
-

## 🏷️ Tags
#paper #arXiv"""

    # Add category tags
    categories = [cat.strip() for cat in arxiv_data['categories'].split(',')]
    for category in categories:
        if category:
            note += f" #{category.replace(' ', '')}"

    # Final cleanup to ensure consistent line breaks
    note = re.sub(r'\n{3,}', '\n\n', note)
    return note.strip()