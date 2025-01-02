import arxiv
import requests
import os
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List
import time

class ArxivPaper:
    def __init__(self, paper_id: str):
        self.paper_id = paper_id
        self.pdf_path = None
        self.metadata = {}
        self.citation_count = 0

    def fetch_metadata(self) -> Dict:
        """Fetch paper metadata from arXiv."""
        search = arxiv.Search(id_list=[self.paper_id])
        paper = next(search.results())

        # Get citation count from Semantic Scholar
        self.citation_count = self.get_citation_count(paper.title)

        # Process arXiv categories
        formatted_categories = self.format_categories(paper.categories)

        self.metadata = {
            'title': paper.title,
            'authors': [author.name for author in paper.authors],
            'published': paper.published,
            'summary': paper.summary,
            'pdf_url': paper.pdf_url,
            'categories': formatted_categories,
            'links': [link.href for link in paper.links],
            'citations': self.citation_count
        }
        return self.metadata

    def get_citation_count(self, title: str) -> Optional[int]:
        """Get citation count from Semantic Scholar."""
        try:
            time.sleep(1)  # API rate limiting

            encoded_title = requests.utils.quote(title)
            url = f"http://api.semanticscholar.org/graph/v1/paper/search"
            params = {
                "query": encoded_title,
                "fields": "citationCount,title",
                "limit": 1
            }
            headers = {
                "Accept": "application/json",
                "User-Agent": "Mozilla/5.0"
            }

            response = requests.get(url, params=params, headers=headers)
            if response.status_code == 200:
                data = response.json()
                if data["data"]:
                    return data["data"][0].get("citationCount")
            return None
        except Exception as e:
            print(f"Warning: Could not fetch citation count: {e}")
            return None

    def format_categories(self, categories: List[str]) -> str:
        """Format arXiv categories to be more readable."""
        category_names = {
            "cs.AI": "Artificial Intelligence",
            "cs.CL": "Computation and Language",
            "cs.CV": "Computer Vision",
            "cs.LG": "Machine Learning",
            "cs.NE": "Neural Computing",
            "cs.RO": "Robotics",
            "cs.SI": "Social Computing",
            "stat.ML": "Machine Learning (Statistics)",
        }

        formatted = []
        for cat in categories:
            if cat in category_names:
                formatted.append(category_names[cat])
            else:
                cat_parts = cat.split('.')
                if len(cat_parts) > 1:
                    formatted.append(cat_parts[1].upper())
                else:
                    formatted.append(cat)

        return ", ".join(formatted)

    def download_pdf(self, output_dir: str = '.') -> Optional[str]:
        """Download PDF file from arXiv."""
        if not self.metadata:
            self.fetch_metadata()

        pdf_url = self.metadata['pdf_url']
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        safe_title = re.sub(r'[<>:"/\\|?*]', '', self.metadata['title'])
        safe_title = safe_title.replace(' ', '_')
        pdf_path = output_dir / f"{safe_title}.pdf"

        response = requests.get(pdf_url)
        if response.status_code == 200:
            with open(pdf_path, 'wb') as f:
                f.write(response.content)
            self.pdf_path = str(pdf_path)
            return self.pdf_path
        return None