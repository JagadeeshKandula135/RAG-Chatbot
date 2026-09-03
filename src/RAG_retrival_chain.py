from typing import Any
from typing import Any
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory



def get_conversational_chain(
    vectordb: Chroma, llm: Any
) -> ConversationalRetrievalChain:
    """Builds a multi-turn conversational RAG chain with session memory."""
    retriever = vectordb.as_retriever(
        search_type="similarity", search_kwargs={"k": 4}
    )

    # 1. Custom prompt for final response generation
    qa_prompt_template = """You are a helpful research assistant. 
Use the context below to answer the user's question. 
If the answer cannot be found in the context, clearly state that it is not in the document before answering using general knowledge.

Chat History:
{chat_history}

Document Context:
{context}

Question: 
{question}

Answer:"""

    QA_PROMPT = PromptTemplate(
        template=qa_prompt_template,
        input_variables=["chat_history", "context", "question"],
    )

    # 2. Memory buffer to keep track of conversation turns
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",  # Explicit output key to avoid ambiguity
    )

    # 3. Create ConversationalRetrievalChain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
        verbose=True,
    )

    return chain

def get_qa_chain(vectordb: Chroma, llm: Any) -> RetrievalQA:
    """Initializes a RetrievalQA chain using the Chroma retriever."""
    retriever = vectordb.as_retriever(
        search_type="similarity", search_kwargs={"k": 4}
    )

    prompt_template = """You are an assistant. Answer the question **only using the information provided in the context below**.  
Do not use any outside knowledge.  

Context: 
{context}

Question: 
{question}

Instructions:  
- If the answer can be found in the context, provide it concisely.  
- If the answer is not present in the context, reply exactly: "I don't know."
"""

    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        input_key="query",
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )

    return chain