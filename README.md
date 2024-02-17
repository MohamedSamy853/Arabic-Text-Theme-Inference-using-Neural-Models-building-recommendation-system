# Arabic Text Theme Inference using Neural Models: Building Recommendation System

This project utilizes topic modeling techniques to identify topics in unlabelled Arabic text data. Various techniques such as Non-Negative Matrix Factorization (NMF), Truncated Singular Value Decomposition (SVD), Latent Dirichlet Allocation (LDA), and TF-IDF Vectorization are employed. Additionally, the project utilizes the Gensim library for efficient topic modeling and BERTopic, a topic modeling approach based on sentence transformers, for sentence embedding, dimension reduction, clustering, and vectorization. The project also incorporates the use of a large language model (LLM), specifically GPT-3.5 Turbo, through LangChain for topic labeling.

## Project Steps:

1. **Data Cleaning**: Cleanse the text data by removing missing rows, duplicates, normalizing text, eliminating numbers, non-Arabic words, URLs, punctuation, and white spaces.

2. **Data Analysis and Visualization**: Analyze the data and generate visualizations, including word clouds, to reveal important insights.

3. **Classical Techniques**: Utilize classical techniques such as LDA, Truncated SVD, NMF, and TF-IDF Vectorizer with KMeans. Address dimensionality issues by performing data preprocessing tasks such as lemmatization, stop words removal, and filtering based on part-of-speech tags. Visualize the results using word clouds.

4. **BERTopic Modeling**: Implement BERTopic for sentence embedding followed by dimension reduction using UMAP and clustering with HDBSCAN. Utilize KeyBert to extract important keywords. Employ LLM for topic labeling.

5. **Building Vector Store**: Create a vector store for the data using ChromaDB to enable topic-based recommendations by considering topic similarities.

6. **Web Application**: Develop a simple web application using Gradio that takes input text and the desired number of recommended topics, and returns the identified topic, key words, and relevant recommendation topics.

## Usage:

To use the application:
1. Clone the repository: `git clone https://github.com/MohamedSamy853/Arabic-Text-Theme-Inference-using-Neural-Models-building-recommendation-system`
2. Navigate to the project directory: `cd Arabic-Text-Theme-Inference-using-Neural-Models-building-recommendation-system`
3. Install requirements: `pip install -r requirements.txt`
4. Run the application: `python main.py`

## Additional Resources:

- [Screen Capture Video](https://github.com/MohamedSamy853/Arabic-Text-Theme-Inference-using-Neural-Models-building-recommendation-system/blob/main/screen-capture.webm)
- [LDA Model Results](https://github.com/MohamedSamy853/Arabic-Text-Theme-Inference-using-Neural-Models-building-recommendation-system/blob/main/results/lda/)
- [NMF Model Results](https://github.com/MohamedSamy853/Arabic-Text-Theme-Inference-using-Neural-Models-building-recommendation-system/blob/main/results/nmf/)
- [Truncated SVD Model Results](https://github.com/MohamedSamy853/Arabic-Text-Theme-Inference-using-Neural-Models-building-recommendation-system/blob/main/results/truncated_svd/)
- [BERTopic Visualization](https://github.com/MohamedSamy853/Arabic-Text-Theme-Inference-using-Neural-Models-building-recommendation-system/blob/main/figs/topic_visualization.html)
## demo video
[Watch the video](https://github.com/MohamedSamy853/Arabic-Text-Theme-Inference-using-Neural-Models-building-recommendation-system/blob/main/screen-capture.mp4)

