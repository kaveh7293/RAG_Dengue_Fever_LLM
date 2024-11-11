# RAG_Dengue_Fever_LLM
This project applies a Retrieval-Augmented Generation (RAG) approach to explore drug discovery solutions targeting Dengue fever. Utilizing the power of machine learning, natural language processing, and large language models, this repository combines advanced text retrieval and generative models to identify potential protein targets and drug candidates.

Project Overview
The workflow includes:

Document Preprocessing: Utilizing PyMuPDF for text extraction and preprocessing from a corpus of Dengue-related scientific papers.
Embedding Generation: Leveraging Sentence-BERT to create dense embeddings, enabling similarity-based document retrieval.
FAISS Indexing: Creating a FAISS index for efficient similarity search across document embeddings.
Generative Response Modeling: Using LLaMA and other causal language models to generate responses, augmented by document retrieval to improve relevance.
Key Technologies
Colab Integration: Google Colab for streamlined notebook execution.
Sentence Transformers: Embeddings for semantic similarity.
FAISS: High-performance similarity search and clustering.
Transformers Library: Loading, fine-tuning, and running large language models.
Getting Started
Clone this repository and install the dependencies listed.
Run the notebook in Google Colab or a local environment with GPU support.
Customize queries to retrieve relevant insights into potential targets for Dengue treatment.
This project applies a Retrieval-Augmented Generation (RAG) approach to explore drug discovery solutions targeting Dengue fever. Utilizing the power of machine learning, natural language processing, and large language models, this repository combines advanced text retrieval and generative models to identify potential protein targets and drug candidates.

Project Overview
The workflow includes:

Document Preprocessing: Utilizing PyMuPDF for text extraction and preprocessing from a corpus of Dengue-related scientific papers.
Embedding Generation: Leveraging Sentence-BERT to create dense embeddings, enabling similarity-based document retrieval.
FAISS Indexing: Creating a FAISS index for efficient similarity search across document embeddings.
Generative Response Modeling: Using LLaMA and other causal language models to generate responses, augmented by document retrieval to improve relevance.
Key Technologies
Colab Integration: Google Colab for streamlined notebook execution.
Sentence Transformers: Embeddings for semantic similarity.
FAISS: High-performance similarity search and clustering.
Transformers Library: Loading, fine-tuning, and running large language models.
Getting Started
Clone this repository and install the dependencies listed.
Run the notebook in Google Colab or a local environment with GPU support.
Customize queries to retrieve relevant insights into potential targets for Dengue treatment.
Example Query
python
Copy code
query = "What are some proteins that could be targeted for dengue fever drug?"
rag_response = generate_response_with_rag(query)
pretrained_response = generate_response_pretrained(query)
print("RAG Response:", rag_response)
print("Pre-trained Model Response:", pretrained_response)
