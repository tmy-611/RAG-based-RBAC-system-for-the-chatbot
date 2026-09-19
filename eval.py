import sys
import types
import warnings

warnings.filterwarnings("ignore")

_vertex_mod = types.ModuleType("langchain_community.chat_models.vertexai")
_vertex_mod.ChatVertexAI = type("ChatVertexAI", (),{})
sys.modules["langchain_community.chat_models.vertexai"] = _vertex_mod

import langchain_community.llms as _llms

if not hasattr(_llms, "VertexAI"):
    _llms.VertexAI = type("VertexAI", (), {})

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from ragas import EvaluationDataset, RunConfig, SingleTurnSample, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness

from app.services.rag import answer_question, retrieve

ANSWER_MODEL = "openai/gpt-oss-120b"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

SAMPLES = [
    {
        "question": "What does the engineering document say about system architecture?",
        "role": "engineering",
        "reference": (
            "FinSolve's architecture is a microservices-based, cloud-native "
            "design emphasizing scalability, resilience, security, and modularity."
        ),
    },
    {
        "question": "How did FinSolve's revenue perform in 2024?",
        "role": "finance",
        "reference": (
            "Revenue grew by 25% in 2024 driven by global expansion in Asia "
            "and Europe, while net income increased by 12%."
        ),
    },
    {
        "question": "What was the ROI of FinSolve's digital campaigns in 2024?",
        "role": "marketing",
        "reference": (
            "The digital campaigns cost $5M and delivered a 3.5x ROI, "
            "generating $17.5M in revenue."
        ),
    },
]

def run_pipeline(sample: dict) -> SingleTurnSample:
    """Execute one query through the RBAC RAG pipeline for evaluation."""
    question, role = sample["question"], sample["role"]
    chunks = retrieve(question, role)
    return SingleTurnSample(
        user_input=question,
        response=answer_question(question, role),
        retrieved_contexts=[c.page_content for c in chunks],
        reference=sample["reference"],
    )

def main() -> None:
    """Evaluate the pipeline with 4 Ragas metrics and print the report."""
    dataset = EvaluationDataset(samples=[run_pipeline(s) for s in SAMPLES])

    llm = LangchainLLMWrapper(
        ChatGroq(model=ANSWER_MODEL, temperature=0, max_tokens=8192)
    )
    embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=EMBED_MODEL))

    result = evaluate(
        dataset=dataset,
        metrics=[Faithfulness(), AnswerRelevancy(strictness=1), ContextPrecision(), ContextRecall()],
        llm=llm,
        embeddings=embeddings,
        run_config=RunConfig(timeout=300, max_workers=2, max_retries=5),
        show_progress=True,
    )

    print("\n=== Ragas evaluation (3 queries) ===")
    print(result)
    result.to_pandas().to_csv("eval_results.csv", index=False)
    print("\nSaved per-sample scores to eval_results.csv")


if __name__ == "__main__":
    main()