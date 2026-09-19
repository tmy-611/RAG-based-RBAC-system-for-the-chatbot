"""Guardrails: out-of-scope router (prompt-based, Groq)."""
import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
from presidio_anonymizer import AnonymizerEngine

load_dotenv()

ROUTER_MODEL = "openai/gpt-oss-20b"
OUT_OF_SCOPE_REPLY = "I can only answer questions related to FinSolve's internal data."
PII_ENTITIES = ["PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "US_SSN"]


@lru_cache(maxsize=1)
def get_router_llm() -> ChatGroq:
    """Cheap, fast LLM used only for routing, never for answering."""
    return ChatGroq(
        model=ROUTER_MODEL, temperature=0
    )


def is_in_scope(message:str) -> bool:
    """Return True when the question is about FinSolve internal company data."""
    verdict = get_router_llm().invoke(
        "You are a query router for FinSolve Technologies, a FinTech company. "
        "Reply with exactly one word: YES if the user question is about the "
        "company's internal data, business, employees, finance, engineering, "
        "marketing, HR, or policies. Reply NO for general knowledge, chit-chat, "
        f"coding help, or anything unrelated. Question: {message}"
    )
    return verdict.content.strip().upper().startswith("YES") 


@lru_cache(maxsize=1)
def get_analyzer() -> AnalyzerEngine:
    """Load the PII detector once, with custom HR recognizers registered."""
    return AnalyzerEngine()

@lru_cache(maxsize=1)
def get_anonymizer() -> AnonymizerEngine:
    """Load the PII masker once."""
    return AnonymizerEngine()

def mask_pii(text: str) -> str:
    """Detect spec PII + HR custom entities, replace with <ENTITY_TYPE>."""
    findings = get_analyzer().analyze(text=text, entities=PII_ENTITIES, language="en")
    return get_anonymizer().anonymize(text=text, analyzer_results=findings).text