import sys
import argparse
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

def embed_text(file_name, model):
    """
    Embed text in file_name using a pre-trained sentence transformer model.
    The text chunks are separated by '\n\n'.
    Create a data structure that contains id, embedding, and text for each chunk.
    Input: file_name (str), model (SentenceTransformer)
    Output: list of embeddings (dictionary with keys: id, embedding, text)
    """
    with open(file_name, 'r') as file:
        text = file.read()
    
    chunks = text.split('\n\n')
    embeddings = []
    for i, chunk in enumerate(chunks):
        embedding = model.encode(chunk, convert_to_tensor=False)  # Ensure it's a list for JSON serialization
        embeddings.append({
            'id': i,
            'embedding': embedding.tolist(),
            'text': chunk
        })

    return embeddings

def embed_user_query(query, model):
    """
    Embed a user query using a pre-trained sentence transformer model.
    Input: query (str), model (SentenceTransformer)
    Output: embedding of the query (list)
    """
    embedding = model.encode(query, convert_to_tensor=False)
    return embedding.tolist()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-load', '--load', help="Load vector database from file")
    parser.add_argument('-query', '--query', help="Search for similar text based on user query")
    args = parser.parse_args()

    # Initialize the SentenceTransformer model and Qdrant client
    model = SentenceTransformer('all-MiniLM-L6-v2')
    client = QdrantClient("localhost:6333")

    if args.load:
        # Embed the text in the file
        embeddings = embed_text(args.load, model)

        # if collection exists, delete it
        if 'reg_collection' in client.get_collections():
            client.delete_collection('reg_collection')
        
        print("Deleted collection reg_collection")
        print("List of collections:")
        print(client.get_collections())

        # Create a collection
        client.recreate_collection(
            collection_name='reg_collection',
            vectors_config=models.VectorParams(
                size=model.get_sentence_embedding_dimension(),
                distance=models.Distance.COSINE
            )
        )
        print("Created collection reg_collection")
        print("List of collections:")
        print(client.get_collections())

        # Upload data to the collection
        points = [
            models.PointStruct(
                id=embedding['id'], 
                vector=embedding['embedding'], 
                payload={'text': embedding['text']}
            ) for embedding in embeddings
        ]
        client.upload_points(collection_name='reg_collection', points=points)

    elif args.query:
        print("List of collections:")
        print(client.get_collections())
        # Embed the user query
        query_embedding = embed_user_query(args.query, model)

        hits = client.search(
            collection_name="reg_collection",
            query_vector=query_embedding,
            limit=1,
        )

        print("Search results:")
        for hit in hits:
            print(hit)

if __name__ == '__main__':
    main()
