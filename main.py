import os
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import chromadb
import gradio  as gr 


#get embedding model

def load_embedding_model(embedding_id):
    embedding = SentenceTransformer(embedding_id)
    return embedding

def encode_text(text, embedding_model):
    encoded_text = embedding_model.encode(text)
    return encoded_text.tolist()

def load_bertopic_model(bertopic_dir , embedding_model_id):
    bertopic_model = BERTopic.load(path=bertopic_dir , embedding_model=embedding_model_id)
    return bertopic_model

def predict_topic(text , bertopic_model):
    topic_id = bertopic_model.transform([text])[0][0]
    topic_name = bertopic_model.get_topic_info(topic_id)['CustomName'].values[0]
    key_words = bertopic_model.get_topic_info(topic_id)['KeyBERT'].values[0]
    key_words = "\n".join(key_words)
    return topic_name , key_words

def active_vector_database(vector_databasse_dir ):
    os.system(f"chroma run --path {vector_databasse_dir}")
    chroma_client = chromadb.HttpClient(host="localhost", port=8000)
    collection = chroma_client.get_collection("ar_topics")
    return collection

def recommend_texts(collection , encoded_text , n_res ):
    res = collection.query(query_embeddings=encoded_text ,n_results=n_res)
    documents = res['documents'][0]
    return documents

def recommend(text , n_res):
        
        encoded_text = encode_text(text , embedding_model)
        
        topic_name , key_words = predict_topic(text , bertopic_model)
        
        recommend_docs = recommend_texts(collections , encoded_text ,int(n_res))
        
        recommend_docs = "\n\n".join(recommend_docs)
        
        return topic_name , key_words , recommend_docs

embedding_model_id = "sentence-transformers/LaBSE"

bert_topic_path = "./models/bertopic_model_safetensors/content/bertopic_model_safetensors/"

vector_path = "./vectorstore/chromadb-topics"

embedding_model = load_embedding_model(embedding_model_id)

bertopic_model = load_bertopic_model(bert_topic_path , embedding_model_id)

collections = active_vector_database(vector_path)

demo = gr.Interface(fn=recommend , 
                        inputs=[gr.Text(max_lines=100 ,label='your topic') , gr.Slider(minimum=1 ,maximum=5 , step=1)] , 
                        outputs=[gr.Textbox(label='topic name') , gr.Text(max_lines=10 , label='key words'),
                                 gr.TextArea(max_lines=300 ,label='recommended topics')])

if __name__ == '__main__':
    
    
    demo.launch()
    
        
        

    
    
    


    