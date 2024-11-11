# RAG_Dengue_Fever_LLM
<h1>RAG-Based Dengue Drug Discovery</h1>

<p>This project applies a <strong>Retrieval-Augmented Generation (RAG)</strong> approach to explore drug discovery solutions targeting Dengue fever. Utilizing the power of machine learning, natural language processing, and large language models, this repository combines advanced text retrieval and generative models to identify potential protein targets and drug candidates.</p>

<h2>Project Overview</h2>
<p>The workflow includes:</p>
<ol>
    <li><strong>Document Preprocessing</strong>: Utilizing PyMuPDF for text extraction and preprocessing from a corpus of Dengue-related scientific papers.</li>
    <li><strong>Embedding Generation</strong>: Leveraging Sentence-BERT to create dense embeddings, enabling similarity-based document retrieval.</li>
    <li><strong>FAISS Indexing</strong>: Creating a FAISS index for efficient similarity search across document embeddings.</li>
    <li><strong>Generative Response Modeling</strong>: Using LLaMA and other causal language models to generate responses, augmented by document retrieval to improve relevance.</li>
</ol>

<h2>Key Technologies</h2>
<ul>
    <li><strong>Colab Integration</strong>: Google Colab for streamlined notebook execution.</li>
    <li><strong>Sentence Transformers</strong>: Embeddings for semantic similarity.</li>
    <li><strong>FAISS</strong>: High-performance similarity search and clustering.</li>
    <li><strong>Transformers Library</strong>: Loading, fine-tuning, and running large language models.</li>
</ul>

<h2>Getting Started</h2>
<ol>
    <li>Clone this repository and install the dependencies listed.</li>
    <li>Run the notebook in Google Colab or a local environment with GPU support.</li>
    <li>Customize queries to retrieve relevant insights into potential targets for Dengue treatment.</li>
</ol>

<h2>Example Query</h2>
<pre><code>query = "What are some proteins that could be targeted for dengue fever drug?"
rag_response = generate_response_with_rag(query)
pretrained_response = generate_response_pretrained(query)
print("RAG Response:", rag_response)
print("Pre-trained Model Response:", pretrained_response)
</code></pre>
