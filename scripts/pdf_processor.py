import pypdf
import re
from pathlib import Path
from typing import Dict, List
import pytesseract
from pdf2image import convert_from_path
import tempfile

class PDFProcessor:
    def __init__(self, pdf_path: str):
        self.pdf_path = Path(pdf_path)
        self.pages_text = []
        self.sections = {}

    def extract_text_with_ocr(self) -> List[str]:
        """Extract text from PDF using OCR."""
        print("Processing PDF content...")

        try:
            reader = pypdf.PdfReader(self.pdf_path)
            for page in reader.pages:
                # 페이지 레이아웃 처리
                text = ""
                def visitor_body(text_obj, cm, tm, fontDict, fontSize):
                    nonlocal text
                    # x 좌표에 따라 텍스트 정렬
                    x = tm[4] if tm is not None else 0
                    # text_obj.get_text()의 결과가 빈 문자열이 아닌 경우에만 처리
                    if text_obj.get_text().strip():
                        text += f"{{x:{x}}}{text_obj.get_text()} "

                page.extract_text(visitor_text=visitor_body)

                # x 좌표를 기준으로 텍스트 재정렬
                if text.strip():
                    lines = text.split('\n')
                    processed_lines = []

                    for line in lines:
                        # x 좌표 태그가 있는 텍스트들을 분리
                        parts = [p for p in line.split('{x:') if p]
                        if parts:
                            # x 좌표와 텍스트를 분리하고 정렬
                            sorted_parts = []
                            for part in parts:
                                if '}' in part:
                                    x_str, text_part = part.split('}', 1)
                                    try:
                                        x = float(x_str)
                                        sorted_parts.append((x, text_part))
                                    except ValueError:
                                        sorted_parts.append((0, part))
                                else:
                                    sorted_parts.append((0, part))

                            # x 좌표 순으로 정렬하고 텍스트만 결합
                            sorted_parts.sort(key=lambda x: x[0])
                            processed_line = ' '.join(text for _, text in sorted_parts)
                            processed_lines.append(processed_line)

                    self.pages_text.append('\n'.join(processed_lines))
                else:
                    raise Exception("Page text extraction failed")
        except Exception as e:
            print(f"Warning: PDF text extraction failed ({str(e)}), trying OCR...")
            with tempfile.TemporaryDirectory() as temp_dir:
                images = convert_from_path(self.pdf_path)
                for i, image in enumerate(images):
                    print(f"Processing page {i+1}/{len(images)}")
                    text = pytesseract.image_to_string(image)
                    self.pages_text.append(text)

        return self.pages_text

    def is_likely_table(self, text: str) -> bool:
        """Check if the text segment likely represents a table."""
        table_patterns = [
            r'\|\s*\|',              # | |
            r'\+[-+]+\+',            # +--+--+
            r'[\|\+][\-\=]+[\|\+]',  # |---|---|
            r'\s{2,}\|\s{2,}',       # 여러 공백으로 구분된 열
            r'^\s*\|.*\|\s*$',       # |로 시작하고 끝나는 행
            r'Table\s+\d+:',         # Table 1:
            r'^\s*[-\+]{3,}\s*$'     # 구분선
        ]

        return any(re.search(pattern, text, re.MULTILINE) for pattern in table_patterns)

    def process_math(self, text: str) -> str:
        """Convert mathematical expressions to LaTeX format."""
        try:
            # Basic math patterns
            math_patterns = [
                (r'(\d+)\^(\d+)', r'$\1^\2$'),
                (r'sqrt\((.*?)\)', r'$\\sqrt{\1}$'),
                (r'([a-zA-Z])\[([^\]]+)\]', r'$\1_{\2}$'),
                (r'sum\((.*?)\)', r'$\\sum{\1}$'),
                (r'int\((.*?)\)', r'$\\int{\1}$'),
            ]

            # Greek letters
            greek_letters = ['alpha', 'beta', 'gamma', 'theta', 'lambda', 'mu', 'sigma', 'omega']
            for letter in greek_letters:
                pattern = fr'\b{letter}\b'
                replacement = fr'$\\{letter}$'
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

            # Apply basic patterns
            for pattern, replacement in math_patterns:
                text = re.sub(pattern, replacement, text)

            # Mathematical operators
            text = re.sub(r'([=><])\s*(\d+(?:\.\d+)?)', r'$\1 \2$', text)
            text = re.sub(r'(\d+(?:\.\d+)?)\s*([×÷±∓∑∏∫])\s*(\d+(?:\.\d+)?)', r'$\1 \2 \3$', text)

            # Preserve existing LaTeX
            text = re.sub(r'\$\$(.*?)\$\$', r'$$\1$$', text)
            text = re.sub(r'\$(.*?)\$', r'$\1$', text)

            return text
        except Exception as e:
            print(f"Warning: Error processing math expressions: {str(e)}")
            return text

    def clean_text(self, text: str) -> str:
        """Clean and format text for better readability."""
        try:
            # 하이픈으로 나뉜 단어 처리
            text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)

            # 줄 단위로 처리
            lines = text.split('\n')
            cleaned_lines = []
            current_paragraph = []

            for line in lines:
                if self.is_likely_table(line):
                    if current_paragraph:
                        cleaned_lines.append(' '.join(current_paragraph))
                        current_paragraph = []
                    continue

                # 빈 줄을 만나면 새로운 문단 시작
                if not line.strip():
                    if current_paragraph:
                        cleaned_lines.append(' '.join(current_paragraph))
                        current_paragraph = []
                    cleaned_lines.append('')
                    continue

                # 수식 처리
                line = self.process_math(line)

                # 불필요한 공백 제거
                line = re.sub(r'\s+', ' ', line.strip())

                if line:
                    current_paragraph.append(line)

            # 마지막 문단 처리
            if current_paragraph:
                cleaned_lines.append(' '.join(current_paragraph))

            # 문단 사이 빈 줄 정리
            text = '\n\n'.join(line for line in cleaned_lines if line.strip())

            return text.strip()
        except Exception as e:
            print(f"Warning: Error cleaning text: {str(e)}")
            return text.strip()

    def identify_sections(self) -> Dict[str, str]:
        """Identify and process major sections in the paper."""
        if not self.pages_text:
            self.extract_text_with_ocr()

        full_text = "\n".join(self.pages_text)

        section_patterns = {
            'abstract': r'(?i)abstract.*?(?=\n\s*(?:introduction|1\.|2\.))',
            'introduction': r'(?i)(?:1\s*)?introduction.*?(?=\n\s*(?:2\.|method|background))',
            'methods': r'(?i)(?:2\s*)?(?:methods|methodology).*?(?=\n\s*(?:3\.|results|discussion))',
            'results': r'(?i)(?:3\s*)?results.*?(?=\n\s*(?:4\.|discussion|conclusion))',
            'discussion': r'(?i)(?:4\s*)?discussion.*?(?=\n\s*(?:5\.|conclusion))',
            'conclusion': r'(?i)(?:5\s*)?conclusion.*?(?=\n\s*(?:acknowledgments))'
        }

        sections = {}
        for section_name, pattern in section_patterns.items():
            match = re.search(pattern, full_text, re.DOTALL | re.IGNORECASE)
            if match:
                content = match.group(0).strip()
                content = self.clean_text(content)
                sections[section_name] = content

        return sections