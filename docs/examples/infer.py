import tritonclient.http as httpclient
import numpy as np
import json
import sys
TRITON_URL = "localhost:8000"
#MODEL_NAME = "semantic" #or tfidf
MODEL_NAME = sys.argv[1]

def query_triton(query_text):
    client = httpclient.InferenceServerClient(url=TRITON_URL)

    # Prepare input
    input_tensor = httpclient.InferInput("query", [1, 1], "BYTES")
    input_tensor.set_data_from_numpy(np.array([[query_text.encode("utf-8")]]))

    # Send inference request
    result = client.infer(
        model_name=MODEL_NAME,
        inputs=[input_tensor]
    )

    # Decode result
    output = result.as_numpy("top_docs")[0].decode("utf-8")
    top_docs = json.loads(output)
    return top_docs

if __name__ == "__main__":
    print("Enter your query. Type 'exit' to quit.")
    while True:
        query = input("\nYour query: ")
        if query.strip().lower() == "exit":
            break

        try:
            results = query_triton(query)
            print("\n--- Top 10 Recommendations ---")
            for i, doc in enumerate(results):
                print(f"{i+1}. Title: {doc['title']}")
                print(f"   ID: {doc['id']}")
                print(f"   Similarity: {doc['similarity']:.4f}")
                print("-" * 30)
        except Exception as e:
            print(f"Error: {e}")
