import streamlit as st
import os
import time
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset

# --- PAGE CONFIG ---
st.set_page_config(page_title="Eval-First RAG System", layout="wide")
st.title("📄 Eval-First RAG System")
st.markdown("Upload a PDF, ask questions, and see exactly how the AI is performing.")

# --- SIDEBAR: API KEY ---
with st.sidebar:
    st.header("🔑 Configuration")
    groq_api_key = st.text_input("Enter Groq API Key", type="password")
    st.markdown("[Get a free Groq API key here](https://console.groq.com)")
    st.markdown("---")
    st.markdown("Built with LangChain, Groq, ChromaDB, and Ragas.")

if not groq_api_key:
    st.warning("👈 Please enter your Groq API Key in the sidebar to start.")
    st.stop()

os.environ["GROQ_API_KEY"] = groq_api_key

# --- CACHED FUNCTIONS ---
@st.cache_resource
def load_models():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)
    return embeddings, llm

embeddings, llm = load_models()

# --- FILE UPLOAD ---
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    with st.spinner("Processing PDF..."):
        loader = PyMuPDFLoader("temp.pdf")
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        
        vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
        retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 1, "fetch_k": 5})
        
    st.success(f"✅ PDF Processed! {len(chunks)} chunks created.")

    # --- RAG PIPELINE ---
    prompt = ChatPromptTemplate.from_template("""
    Answer the question based ONLY on the following context:
    {context}
    Question: {question}
    """)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # --- TABS ---
    tab1, tab2 = st.tabs(["💬 Chat with PDF", "📊 Evaluation Dashboard"])

    with tab1:
        query = st.text_input("Ask a question about the PDF:")
        if query:
            with st.spinner("Thinking..."):
                answer = rag_chain.invoke(query)
                st.markdown(f"**Answer:** {answer}")
                
                with st.expander("View Retrieved Context"):
                    docs = retriever.invoke(query)
                    for i, doc in enumerate(docs):
                        st.write(f"**Chunk {i+1}:**")
                        st.info(doc.page_content)

    with tab2:
        st.markdown("### Run Automated Evaluation (Ragas)")
        st.markdown("This uses a pre-built **golden dataset** of 2 questions and ground truths to evaluate the RAG pipeline.")
        
        if st.button("Run Evaluation"):
            with st.spinner("Running RAG chain and evaluating with Ragas... This takes ~2 minutes."):
                from ragas.llms import LangchainLLMWrapper
                from ragas.embeddings import LangchainEmbeddingsWrapper
                
                # Golden Dataset (2 questions to stay under rate limits)
                questions = [
                    "What is the BLEU score of the Transformer model on the English-to-German translation task?",
                    "How many attention heads are used in the base Transformer model?"
                ]
                
                ground_truths = [
                    "The Transformer achieves a BLEU score of 28.4 on the WMT 2014 English-to-German translation task.",
                    "The base Transformer model uses 8 parallel attention heads."
                ]
                
                answers = []
                contexts = []
                
                for q in questions:
                    ans = rag_chain.invoke(q)
                    answers.append(ans)
                    docs = retriever.invoke(q)
                    contexts.append([d.page_content for d in docs])
                    time.sleep(15)  # Sleep 15 seconds between questions to avoid rate limits
                
                data = {
                    "question": questions,
                    "answer": answers,
                    "contexts": contexts,
                    "ground_truth": ground_truths
                }
                eval_dataset = Dataset.from_dict(data)
                
                ragas_llm = LangchainLLMWrapper(llm)
                ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)
                
                result = evaluate(
                    dataset=eval_dataset,
                    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
                    llm=ragas_llm,
                    embeddings=ragas_embeddings,
                )
                
                st.success("✅ Evaluation Complete!")
                st.dataframe(result.to_pandas())
                
                st.markdown("---")
                st.markdown("### 📝 Answers Generated by your RAG Pipeline")
                for i, (q, a, gt) in enumerate(zip(questions, answers, ground_truths)):
                    st.markdown(f"**Q{i+1}:** {q}")
                    st.markdown(f"**AI Answer:** {a}")
                    st.markdown(f"**Ground Truth:** {gt}")
                    st.markdown("---")
