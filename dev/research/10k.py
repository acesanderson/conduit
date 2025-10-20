import requests
import requests_cache
from pathlib import Path
from pydantic import BaseModel, Field, computed_field
from time import sleep
from functools import cached_property
from markdownify import markdownify as md
from rich.console import Console
from rich.markdown import Markdown

requests_cache.install_cache(".sec_cache")
console = Console()


class SECFiling(BaseModel):
    url: str = Field(..., description="The URL of the SEC filing")
    date: str = Field(..., description="The filing date of the SEC filing")

    @cached_property
    def content(self):
        headers = {"User-Agent": "Your Name your.email@example.com"}
        r = requests.get(self.url, headers=headers)
        return r.text

    def print(self):
        md_content = md(self.content)
        markdown = Markdown(md_content)
        console.print(markdown)


class Company(BaseModel):
    name: str = Field(..., description="The name of the company")
    cik: str = Field(..., description="The Central Index Key (CIK) of the company")

    @cached_property
    def filings(self) -> list[SECFiling]:
        url = f"https://data.sec.gov/submissions/CIK{self.cik}.json"
        headers = {"User-Agent": "Your Name your.email@example.com"}

        r = requests.get(url, headers=headers)
        data = r.json()

        # Zip the parallel lists into structured records
        records = zip(
            data["filings"]["recent"]["accessionNumber"],
            data["filings"]["recent"]["form"],
            data["filings"]["recent"]["filingDate"],
            data["filings"]["recent"]["primaryDocument"],
        )

        # Filter and view only 10-K/20-F filings
        filings = []
        for acc, form, date, doc in records:
            if form == "10-K" or form == "20-F":
                url = f"https://www.sec.gov/Archives/edgar/data/{int(self.cik)}/{acc.replace('-', '')}/{doc}"
                filings.append(SECFiling(url=url, date=date))
        if not filings:
            print(f"No 10-K or 20-F filings found for CIK {self.cik}")
        return filings


class Companies(BaseModel):
    companies: list[Company] = Field(
        ..., description="A list of companies with their CIKs"
    )

    @classmethod
    def from_markdown(cls, filepath: str | Path) -> "Companies":
        lines = Path(filepath).read_text().splitlines()
        companies = [line.split(",") for line in lines]
        company_objs = [Company(name=c[0], cik=c[1]) for c in companies]
        return cls(companies=company_objs)


if __name__ == "__main__":
    companies = Companies.from_markdown(Path(__file__).parent / "10k_companies.md")
    companies.companies[-1].filings[0].print()
