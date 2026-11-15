from __future__ import annotations

import re
from typing import Dict, List, Set, Tuple


class EntityRelationExtractor:
    """
    Lightweight, dependency-free heuristic extractor for entity recognition
    and (Subject, Relation, Object) triplet parsing from raw conversational text.
    """

    STOPWORDS = {"and", "or", "the", "a", "an", "on", "in", "at", "by", "for", "with", "to", "of"}

    RELATION_VERB_MAP: Dict[str, str] = {
        "uses": "USES",
        "is using": "USES",
        "utilized": "USES",
        "utilizes": "USES",
        "develops": "DEVELOPS",
        "is developing": "DEVELOPS",
        "built": "DEVELOPS",
        "builds": "DEVELOPS",
        "created": "DEVELOPS",
        "works at": "WORKS_AT",
        "works for": "WORKS_AT",
        "employed by": "WORKS_AT",
        "runs on": "DEPLOYED_ON",
        "deployed on": "DEPLOYED_ON",
        "hosted on": "DEPLOYED_ON",
        "is a": "IS_A",
        "is an": "IS_A",
        "participates in": "PARTICIPATES_IN",
        "competes in": "PARTICIPATES_IN",
    }

    KNOWN_TECH_TERMS: Set[str] = {
        "python", "pytorch", "tensorflow", "fastapi", "docker", "kubernetes",
        "sqlite", "postgresql", "redis", "langchain", "llamaindex", "vllm",
        "huggingface", "transformers", "cuda", "onnx", "triton", "ollama",
        "react", "vue", "nextjs", "typescript", "golang", "rust"
    }

    @classmethod
    def _clean_token(cls, token: str) -> str:
        words = [w for w in token.split() if w.lower() not in cls.STOPWORDS]
        return " ".join(words).title().strip()

    @classmethod
    def extract_entities(cls, text: str) -> List[str]:
        entities: Set[str] = set()

        # 1. Capitalized Proper Nouns / Acronyms
        cap_matches = re.findall(r"\b[A-Z][a-zA-Z0-9_]+\b", text)
        for m in cap_matches:
            if len(m) > 1 and m.lower() not in {"the", "this", "that", "user", "assistant", "system"}:
                entities.add(m.strip())

        # 2. Known Technical Acronyms (e.g. LLM, RAG, API, VLM, OCR, GPU, CPU)
        acronyms = re.findall(r"\b[A-Z]{2,6}\b", text)
        for a in acronyms:
            entities.add(a)

        # 3. Known technical keywords regardless of casing
        lower_text = f" {text.lower()} "
        for term in cls.KNOWN_TECH_TERMS:
            if re.search(rf"\b{re.escape(term)}\b", lower_text):
                entities.add(term.title())

        return sorted(list(entities))

    @classmethod
    def extract_relations(cls, text: str) -> List[Tuple[str, str, str]]:
        triplets: List[Tuple[str, str, str]] = []
        
        # Sort verbs by length descending to match multi-word verbs first
        sorted_verbs = sorted(cls.RELATION_VERB_MAP.keys(), key=len, reverse=True)
        verb_pattern_str = "|".join(re.escape(v) for v in sorted_verbs)
        
        # Pattern: ([Entity word]) [verb phrase] ([Entity word])
        pattern = re.compile(
            rf"\b([A-Z][a-zA-Z0-9_]+)\b\s+({verb_pattern_str})\s+\b([A-Z][a-zA-Z0-9_]+)\b",
            re.IGNORECASE,
        )

        for match in pattern.finditer(text):
            src, verb_phrase, tgt = match.groups()
            rel_name = cls.RELATION_VERB_MAP.get(verb_phrase.lower(), "RELATED_TO")
            src_clean = cls._clean_token(src)
            tgt_clean = cls._clean_token(tgt)
            if src_clean and tgt_clean and src_clean.lower() != tgt_clean.lower():
                triplets.append((src_clean, tgt_clean, rel_name))

        return triplets
