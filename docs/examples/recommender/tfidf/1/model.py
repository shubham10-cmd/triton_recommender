import json
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import triton_python_backend_utils as pb_utils
import os

class TritonPythonModel:
    def initialize(self, args):
        print("======INITIALIZING======= TFIDF",flush=True)
        # Load transformation weights
        with open(os.path.join(os.path.dirname(__file__), "doc_tower_kernel.json"),'r') as f:
            kernel = np.array(json.load(f), dtype=np.float32)
        with open(os.path.join(os.path.dirname(__file__),"doc_tower_bias.json"), "r") as f:
            bias = np.array(json.load(f), dtype=np.float32)

        self.kernel = torch.tensor(kernel.T, dtype=torch.float32)
        self.bias = torch.tensor(bias, dtype=torch.float32)

        # Rebuild TF-IDF vectorizer
        with open(os.path.join(os.path.dirname(__file__),"tfidf_vocabulary.json"), "r") as f:
            vocab = json.load(f)
        with open(os.path.join(os.path.dirname(__file__),"tfidf_idf.json"), "r") as f:
            idf = json.load(f)

        self.vectorizer = TfidfVectorizer()
        self.vectorizer.vocabulary_ = vocab
        self.vectorizer.idf_ = np.array(idf)
        self.vectorizer.fixed_vocabulary_ = True

        # Load document embeddings (64-dim)
        df = pd.read_csv(os.path.join(os.path.dirname(__file__),"document_embeddings.csv"), index_col="original_document_id")
        self.doc_embeddings = df.values
        self.doc_ids = df.index.tolist()

        # Title map
        meta = pd.read_csv(os.path.join(os.path.dirname(__file__),"su_docs_cleaned.csv"))
        self.title_map = dict(zip(meta["ids"], meta["title"]))

    def cosine_similarity(self, vec, mat):
        norm_vec = vec / np.linalg.norm(vec)
        norm_mat = mat / np.linalg.norm(mat, axis=1, keepdims=True)
        return np.dot(norm_mat, norm_vec)

    def execute(self, requests):
        responses = []
        for request in requests:
            query_tensor = pb_utils.get_input_tensor_by_name(request, "query")
            query = query_tensor.as_numpy()

            # Safely unwrap (1, 1) shape to a single value
            if isinstance(query, np.ndarray):
                query = query[0]  # (1, 1) → (1,)
                if isinstance(query, np.ndarray):
                    query = query[0]  # (1,) → scalar (bytes or str)

            # Decode if needed
            query_str = query if isinstance(query, str) else query.decode("utf-8")



            tfidf_vec = self.vectorizer.transform([query_str]).toarray()
            tfidf_tensor = torch.tensor(tfidf_vec, dtype=torch.float32)

            with torch.no_grad():
                transformed = F.linear(tfidf_tensor, self.kernel, self.bias)
                transformed = F.relu(transformed)
                query_emb = transformed.cpu().numpy()[0]

            sims = self.cosine_similarity(query_emb, self.doc_embeddings)
            top_idx = np.argsort(sims)[-10:][::-1]

            top_docs = [
                {
                    "id": str(self.doc_ids[i]),
                    "title": self.title_map.get(self.doc_ids[i], "Unknown"),
                    "similarity": float(sims[i])
                }
                for i in top_idx
            ]

            json_output = json.dumps(top_docs).encode("utf-8")
            output_tensor = pb_utils.Tensor("top_docs", np.array([json_output], dtype=np.object_))
            responses.append(pb_utils.InferenceResponse(output_tensors=[output_tensor]))
        return responses
