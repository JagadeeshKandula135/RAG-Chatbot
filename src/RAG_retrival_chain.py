from typing import Any

from langchain_chroma import Chroma
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_core.prompts import PromptTemplate

from src.config import MEMORY_WINDOW, MMR_LAMBDA, RETRIEVAL_FETCH_K, RETRIEVAL_K


def build_retriever(vectordb: Chroma):
    """MMR retrieval: over-fetch candidates by cosine similarity, then
    re-select k results balancing relevance and diversity (less redundant
    context -> better answers with fewer prompt tokens)."""
    return vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": RETRIEVAL_K,
            "fetch_k": RETRIEVAL_FETCH_K,
            "lambda_mult": MMR_LAMBDA,
        },
    )


QA_PROMPT_TEMPLATE = """You are a precise research assistant.
Answer using the document context below. Quote concrete numbers and terms exactly.
If the answer is not in the context, say so explicitly before offering any general knowledge.
Use the chat history to resolve references like "it" or "that".

Chat History:
{chat_history}

Document Context:
{context}

Question:
{question}

Answer:"""

CONDENSE_PROMPT_TEMPLATE = """Rewrite the follow-up question as a standalone search query
using the chat history. Return only the rewritten question.

Chat History:
{chat_history}

Follow-up Question: {question}
Standalone Question:"""


def get_conversational_chain(vectordb: Chroma, llm: Any) -> ConversationalRetrievalChain:
    """Multi-turn RAG chain with bounded (windowed) memory per session."""
    memory = ConversationBufferWindowMemory(
        k=MEMORY_WINDOW,
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=build_retriever(vectordb),
        memory=memory,
        return_source_documents=True,
        condense_question_prompt=PromptTemplate.from_template(CONDENSE_PROMPT_TEMPLATE),
        combine_docs_chain_kwargs={
            "prompt": PromptTemplate(
                template=QA_PROMPT_TEMPLATE,
                input_variables=["chat_history", "context", "question"],
            )
        },
    )


def get_qa_chain(vectordb: Chroma, llm: Any) -> RetrievalQA:
    """Single-turn, strictly grounded QA chain."""
    prompt = PromptTemplate(
        template="""Answer the question **only using the context below**.
If the answer is not present, reply exactly: "I don't know."

Context:
{context}

Question:
{question}
""",
        input_variables=["context", "question"],
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=build_retriever(vectordb),
        input_key="query",
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
