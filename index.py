import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from qdrant_client.models import PointStruct

client = QdrantClient(path="./db") 


# Create a collection
if not client.collection_exists("my_collection"):
   client.create_collection(
      collection_name="my_collection",
      vectors_config=VectorParams(size=100, distance=Distance.COSINE),
   )
   print("Collection created")


# Insert vectors into a collection
vectors = np.random.rand(100, 100)
client.upsert(
   collection_name="my_collection",
   points=[
      PointStruct(
            id=idx,
            vector=vector.tolist(),
            payload={"color": "red", "rand_number": idx % 10}
      )
      for idx, vector in enumerate(vectors)
   ]
)

my_collection = client.get_collection("my_collection")

print(f"{my_collection.points_count} Vectors inserted")

# Search for similar vectors
query_vector = np.random.rand(100)
query_response = client.query_points(
   collection_name="my_collection",
   query=query_vector,
   limit=5  # Return 5 closest points
)
hits = query_response.points
print(f"Found {len(hits)} similar vectors")

# print(f"Found {len(hits)} similar vectors")

if client.delete_collection("my_collection"):
    print("Collection deleted")