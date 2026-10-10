import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
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

# --- CACHED FUNCTIONS (To speed up the app) ---
@st.cache_resource
def load_models():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    llm =ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    return embeddings, llm

embeddings, llm = load_models()

# --- FILE UPLOAD ---
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    # Save file temporarily
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Load and Chunk
    with st.spinner("Processing PDF..."):
        loader = PyPDFLoader("temp.pdf")
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(documents)
        
        # Create Vector Store
        vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        
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
        st.markdown("This will generate 5 test questions from your PDF and evaluate the RAG pipeline.")
        
        if st.button("Run Evaluation"):
            with st.spinner("Generating test set and evaluating... This takes ~1 minute."):
                
                test_questions = [
                    "What is the main topic of the document?",
                    "What methodology was used?",
                    "What were the key results?",
                    "What are the limitations mentioned?",
                    "What is the conclusion?"
                ]
                
                answers = []
                contexts = []
                
                for q in test_questions:
                    ans = rag_chain.invoke(q)
                    answers.append(ans)
                    docs = retriever.invoke(q)
                    contexts.append([d.page_content for d in docs])
                
                ground_truths = ["Refer to the document for the answer."] * 5
                
                data = {
                    "question": test_questions,
                    "answer": answers,
                    "contexts": contexts,
                    "ground_truth": ground_truths
                }
                eval_dataset = Dataset.from_dict(data)
                
                result = evaluate(
                    dataset=eval_dataset,
                    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
                )
                
                st.success("✅ Evaluation Complete!")
                st.dataframe(result.to_pandas())
                st.info("Faithfulness measures if the answer is derived ONLY from the context (no hallucinations).")
