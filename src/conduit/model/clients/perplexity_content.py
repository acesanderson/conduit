from pydantic import BaseModel
from typing import override


class PerplexityCitation(BaseModel):
    title: str
    url: str
    date: str | None


class PerplexityContent(BaseModel):
    text: str
    citations: list[PerplexityCitation]

    @override
    def __str__(self) -> str:
        """
        Format the content and citations for display, in markdown format.
        """
        if not self.citations:
            return self.text
        citations_strs = []
        for citation in self.citations:
            citation_str = f"[{citation.title}]({citation.url})"
            if citation.date:
                citation_str += f" ({citation.date})"
            citations_strs.append(citation_str)
        citations_str = "\n".join(f"- {c}" for c in citations_strs)
        return f"{self.text}\n\n## Sources:\n{citations_str}"
