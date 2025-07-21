import json
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import triton_python_backend_utils as pb_utils
import os

class TritonPythonModel:
    def initialize(self, args):
        print("======INITIALIZING======= Semantic",flush=True)
        # Load projection weights
        with open(os.path.join(os.path.dirname(__file__), "doc_tower_kernel.json"), "r") as f:
            kernel = np.array(json.load(f), dtype=np.float32)
        with open(os.path.join(os.path.dirname(__file__), "doc_tower_bias.json"), "r") as f:
            bias = np.array(json.load(f), dtype=np.float32)

        self.kernel = torch.tensor(kernel.T, dtype=torch.float32)
        self.bias = torch.tensor(bias, dtype=torch.float32)

        # Sentence encoder
        print('downloading sentence transformer...',flush=True)
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        print("Finished dpwnloading",flush=True)
        # Load embeddings and metadata
        df = pd.read_csv(os.path.join(os.path.dirname(__file__), "document_embeddings.csv"), index_col="original_document_id")
        self.doc_embeddings = df.values
        self.doc_ids = df.index.tolist()

        meta = pd.read_csv(os.path.join(os.path.dirname(__file__), "su_docs_cleaned.csv"))
        self.title_map = dict(zip(meta["ids"], meta["title"]))
        self.text_map = dict(zip(meta["ids"], meta["article"]))  # required for reranker

        # Reranker setup
        print("downloading Qwen3 ...",flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Reranker-0.6B", padding_side='left')
        self.reranker = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-Reranker-0.6B").eval()

    def cosine_similarity(self, vec, mat):
        norm_vec = vec / np.linalg.norm(vec)
        norm_mat = mat / np.linalg.norm(mat, axis=1, keepdims=True)
        return np.dot(norm_mat, norm_vec)

    def rerank(self, query, candidate_ids):
        instruction = 'Given a web search query, retrieve relevant passages that answer the query'
        texts = [self.text_map.get(doc_id, "") for doc_id in candidate_ids]
        pairs = [f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {text}" for text in texts]
        max_length = 8192

        prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
        suffix_tokens = self.tokenizer.encode(suffix, add_special_tokens=False)

        inputs = self.tokenizer(
        pairs, padding=False, truncation='longest_first',
        return_attention_mask=False, max_length=max_length - len(prefix_tokens) - len(suffix_tokens)
        )


        for i, ele in enumerate(inputs['input_ids']):
            inputs['input_ids'][i] = prefix_tokens + ele + suffix_tokens
        inputs = self.tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=max_length)

        token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        token_true_id = self.tokenizer.convert_tokens_to_ids("yes")

        batch_scores = self.reranker(**inputs).logits[:, -1, :]
        print(batch_scores.shape)
        true_vector = batch_scores[:, token_true_id]
        false_vector = batch_scores[:, token_false_id]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        scores = batch_scores[:, 1].exp().tolist()

        # encodings = self.tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")
        # with torch.no_grad():
        #     outputs = self.reranker(**encodings, labels=encodings["input_ids"])
        #     losses = torch.nn.functional.cross_entropy(
        #         outputs.logits.view(-1, outputs.logits.size(-1)), 
        #         encodings["input_ids"].view(-1), 
        #         reduction='none'
        #     ).view(encodings["input_ids"].shape)

        #     # Use total loss per sequence as inverse relevance
        #     relevance_scores = -losses.sum(dim=1).cpu().numpy()

        # Sort based on scores
        reranked = sorted(zip(candidate_ids, scores), key=lambda x: x[1], reverse=True)
        return reranked[:10]

    def execute(self, requests):
        responses = []
        for request in requests:
            query_tensor = pb_utils.get_input_tensor_by_name(request, "query")
            query = query_tensor.as_numpy()

            # Unwrap and decode
            if isinstance(query, np.ndarray):
                query = query[0]
                if isinstance(query, np.ndarray):
                    query = query[0]
            query_str = query if isinstance(query, str) else query.decode("utf-8")

            with torch.no_grad():
                vec = self.encoder.encode(query_str, convert_to_tensor=True).unsqueeze(0)
                transformed = F.linear(vec, self.kernel, self.bias)
                transformed = F.relu(transformed)
                query_emb = transformed.cpu().numpy()[0]

            sims = self.cosine_similarity(query_emb, self.doc_embeddings)
            top_idx = np.argsort(sims)[-10:][::-1]
            candidate_ids = [self.doc_ids[i] for i in top_idx]

            # Rerank top 25
            reranked_docs = self.rerank(query_str, candidate_ids)

            top_docs = [
                {
                    "id": str(doc_id),
                    "title": self.title_map.get(doc_id, "Unknown"),
                    "similarity": float(score)
                }
                for doc_id, score in reranked_docs
            ]

            json_output = json.dumps(top_docs).encode("utf-8")
            output_tensor = pb_utils.Tensor("top_docs", np.array([json_output], dtype=np.object_))
            responses.append(pb_utils.InferenceResponse(output_tensors=[output_tensor]))

        return responses





























# import json
# import numpy as np
# import torch
# import torch.nn.functional as F
# import pandas as pd
# from sentence_transformers import SentenceTransformer
# import triton_python_backend_utils as pb_utils
# import os
# class TritonPythonModel:
#     def initialize(self, args):
#         print("=============Initilizing Semantic============",flush=True)
#         # Load trained projection weights
#         with open(os.path.join(os.path.dirname(__file__),"doc_tower_kernel.json"), "r") as f:
#             kernel = np.array(json.load(f), dtype=np.float32)
#         with open(os.path.join(os.path.dirname(__file__),"doc_tower_bias.json"), "r") as f:
#             bias = np.array(json.load(f), dtype=np.float32)

#         self.kernel = torch.tensor(kernel.T, dtype=torch.float32)
#         self.bias = torch.tensor(bias, dtype=torch.float32)

#         # Load SentenceTransformer
#         print("loading sentence SentenceTransformer" , flush=True)
#         self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
#         print("Done loading",flush = True)
#         # Load precomputed 64-dim document embeddings
#         df = pd.read_csv(os.path.join(os.path.dirname(__file__),"document_embeddings.csv"), index_col="original_document_id")
#         self.doc_embeddings = df.values  # shape: (N_docs, 64)
#         self.doc_ids = df.index.tolist()

#         # Map ID to title (from original metadata)
#         meta = pd.read_csv(os.path.join(os.path.dirname(__file__),"su_docs_cleaned.csv"))
#         self.title_map = dict(zip(meta["ids"], meta["title"]))

#     def cosine_similarity(self, vec, mat):
#         norm_vec = vec / np.linalg.norm(vec)
#         norm_mat = mat / np.linalg.norm(mat, axis=1, keepdims=True)
#         return np.dot(norm_mat, norm_vec)

#     def execute(self, requests):
#         responses = []
#         for request in requests:
#             query_tensor = pb_utils.get_input_tensor_by_name(request, "query")
#             query = query_tensor.as_numpy()

#             # Safely unwrap (1, 1) shape to a single value
#             if isinstance(query, np.ndarray):
#                 query = query[0]  # (1, 1) → (1,)
#                 if isinstance(query, np.ndarray):
#                     query = query[0]  # (1,) → scalar (bytes or str)

#             # Decode if needed
#             query_str = query if isinstance(query, str) else query.decode("utf-8")



#             with torch.no_grad():
#                 vec = self.encoder.encode(query_str, convert_to_tensor=True).unsqueeze(0)
#                 transformed = F.linear(vec, self.kernel, self.bias)
#                 transformed = F.relu(transformed)
#                 query_emb = transformed.cpu().numpy()[0]

#             sims = self.cosine_similarity(query_emb, self.doc_embeddings)
#             top_idx = np.argsort(sims)[-10:][::-1]

#             top_docs = [
#                 {
#                     "id": str(self.doc_ids[i]),
#                     "title": self.title_map.get(self.doc_ids[i], "Unknown"),
#                     "similarity": float(sims[i])
#                 }
#                 for i in top_idx
#             ]

#             json_output = json.dumps(top_docs).encode("utf-8")
#             output_tensor = pb_utils.Tensor("top_docs", np.array([json_output], dtype=np.object_))
#             responses.append(pb_utils.InferenceResponse(output_tensors=[output_tensor]))
#         return responses
