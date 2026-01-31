
import os
import gc
import logging
import unicodedata
import re
import collections 
from llama_index.core.schema import NodeWithScore # Explicitly import NodeWithScore for clarity
from typing import List, Dict, Any, Optional, Callable
import time
import random
import torch
import unicodedata
import re
from functools import lru_cache
from typing import Any, List, Optional
import torch
from weaviate.classes.init import Auth
from weaviate.agents.query import QueryAgent
from weaviate_agents.classes import QueryAgentCollectionConfig
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from huggingface_hub import login



# Chainlit
import chainlit as cl
from chainlit.input_widget import Select

# Lazy imports for heavy libraries -- imported inside startup
# from llama_index.core.settings import Settings
# from llama_index.core import Document, StorageContext, VectorStoreIndex
# from llama_index.vector_stores.weaviate import WeaviateVectorStore

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# -----------------------------
# Environment variables & defaults
# -----------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY", "")
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "")

# For CPU-only environments
os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv("CUDA_VISIBLE_DEVICES", "-1")


try:
    login(HUGGINGFACE_TOKEN)
except Exception as e:
    print(f"Warning: HuggingFace login failed: {e}")
    print("Continuing without authentication...")


import collections
from typing import List, Any, Optional
import uuid


import uuid # Added missing import
from llama_index.core.schema import TextNode, NodeWithScore # Added missing imports
# Use LlamaIndex standard schemas if available, otherwise simulate them
try:
    from llama_index.core.schema import NodeWithScore, TextNode
except ImportError:
    # Fallback if you don't have llama_index installed but want the structure
    class TextNode:
        def __init__(self, text: str, id_: str = None):
            self.text = text
            self.node_id = id_ or str(uuid.uuid4())

    class NodeWithScore:
        def __init__(self, node: TextNode, score: float = None):
            self.node = node
            self.score = score

        @property
        def text(self):
            return self.node.text

class WeaviateAgentRetriever:
    def __init__(self, agent):
        self.agent = agent

    def retrieve(self, queries: List[str], top_k=5, llm=None) -> List[NodeWithScore]:
        results = []
        # FIX 1: Ensure we handle single strings gracefully, though we expect a list
        if isinstance(queries, str):
            queries = [queries]

        for q in queries:
            # Call the agent
            response = self.agent.search(q, limit=top_k)

            # Parse results
            # Note: Adjust 'response.search_results.objects' based on actual Weaviate response structure
            if hasattr(response, 'search_results') and hasattr(response.search_results, 'objects'):
                iterator = response.search_results.objects
            else:
                iterator = [] # Handle empty/error cases safely

            for obj in iterator:
                text = obj.properties.get("text") or obj.properties.get("content")
                if text:
                    # FIX 2: Create a TextNode first
                    node = TextNode(text=text)

                    # FIX 3: Wrap it in NodeWithScore (default score to 1.0 or fetch from Weaviate metadata)
                    # Weaviate usually returns certainty or distance, usually mapped to score
                    # Corrected: Access certainty from obj._additional
                    score = obj.metadata.get("score", 1.0)

                    results.append(NodeWithScore(node=node, score=score))
        return results





# -----------------------------
# Utilities
# -----------------------------

def normalize_yoruba(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def free_memory():
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# -----------------------------
# Lightweight Embedding wrapper (lazy-created)
# -----------------------------
from llama_index.core.embeddings import BaseEmbedding

from typing import Any, List

import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel
class AfriBERTaEmbedding(BaseEmbedding):
    _model: Any = None
    _tokenizer: Any = None
    _device: Any = None

    def __init__(
        self,
        model_name: str = "Davlan/afro-xlmr-mini",
        **kwargs: Any
    ) -> None:
        super().__init__(model_name=model_name, **kwargs)

        # CPU device
        self._device = torch.device("cpu")

        # 2. Load tokenizer (use_fast=False to avoid tokenizer.json issues)
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=False
        )

        # 3. Load model on CPU
        self._model = AutoModel.from_pretrained(model_name).to(self._device)
        self._model.eval()

    def _mean_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = (
            attention_mask.unsqueeze(-1)
            .expand(token_embeddings.size())
            .float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def _embed(self, texts: List[str]) -> List[List[float]]:
        """Core embedding logic"""
        inputs = self._tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512   # important for RAG
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model(**inputs)

            # Pool
            embeddings = self._mean_pooling(
                outputs.last_hidden_state,
                inputs["attention_mask"]
            )

            # Normalize
            embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings.tolist()

    # --- LlamaIndex required methods ---
    def _get_query_embedding(self, query: str) -> List[float]:
        return self._embed([query])[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._embed([text])[0]

    def _get_text_embedding_batch(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)



# -----------------------------
# Safe LLM completion wrapper
# -----------------------------
# The original call_llm was a RAG function, let's redefine it here to be a simple LLM call
logger = logging.getLogger(__name__)
def safe_llm_complete(llm, prompt: str) -> Optional[str]:
    if llm is None:
        logger.warning("LLM is None — cannot complete prompt.")
        return None
    try:
        if hasattr(llm, "complete"):
            resp = llm.complete(prompt)
            if resp is None:
                return None
            if hasattr(resp, "text") and resp.text:
                return str(resp.text).strip()
            if hasattr(resp, "output_text") and resp.output_text:
                return str(resp.output_text).strip()
            return str(resp).strip()

        if hasattr(llm, "chat"):
            resp = llm.chat(messages=[{"role": "user", "content": prompt}])
            # Try common response shapes
            if hasattr(resp, "message") and hasattr(resp.message, "content"):
                return str(resp.message.content).strip()
            if hasattr(resp, "output_text"):
                return str(resp.output_text).strip()
            return str(resp).strip()

        # Generic fallback
        return str(llm(prompt)).strip()
    except Exception as e:
        logger.warning(f"Safe LLM call failed: {e}")
        return None

class PreRetrievalModule:
    """Expands query using HyDE (Hypothetical Document Embedding)."""
    def __init__(self, llm: Optional[Any] = None, enable_hyde: bool = True, enable_expansion: bool = True):
        self.llm = llm
        self.enable_hyde = enable_hyde
        self.enable_expansion = enable_expansion

    def process(self, query: str) -> List[str]:
        """Apply HyDE and query expansion to produce query variants."""
        queries = [query]

        if self.enable_hyde and self.llm:
            hyde_prompt = f"Write a short factual paragraph that could answer: '{query}'"
            hypo_doc = safe_llm_complete(self.llm, hyde_prompt)
            if hypo_doc:
                queries.append(hypo_doc)

        if self.enable_expansion and self.llm:
            expansion_prompt = f"Expand this question with related terms: '{query}'"
            expansion = safe_llm_complete(self.llm, expansion_prompt)
            if expansion:
                queries.append(expansion)

        return queries


class RetrievalModule:
    """Combines dense and sparse retrieval (hybrid)."""
    def __init__(self, dense_retriever: Any, sparse_retriever: Optional[Any] = None, hybrid_alpha: float = 0.5):
        self.dense = dense_retriever
        self.sparse = sparse_retriever
        self.hybrid_alpha = hybrid_alpha

    def retrieve(self, queries: List[str], top_k: int = 10) -> List[NodeWithScore]: # Adjusted return type hint
        all_results_raw = []  # Store raw NodeWithScore objects

        for q in queries:
            # Dense retrieval
            dense_results = self.dense.retrieve(q)
            all_results_raw.extend(dense_results)

            if self.sparse:
                # Sparse retrieval
                sparse_results = self.sparse.retrieve(q)
                all_results_raw.extend(sparse_results)

        # Use a dictionary to deduplicate based on node_id, preserving NodeWithScore
        unique_nodes_map = collections.OrderedDict()
        for node_with_score in all_results_raw:
            # Assuming node_with_score.node.node_id is unique enough
            unique_nodes_map[node_with_score.node.node_id] = node_with_score

        # Return list of NodeWithScore objects, limited by top_k
        return list(unique_nodes_map.values())[:top_k]


class PostRetrievalModule:
    """Applies reranking, filtering, and compression."""
    def __init__(self, embed_model):
        self.embed_model = embed_model

    def rerank(self, nodes, query, top_k=3):
        if not nodes:
            return []
        # Use _embed and convert to tensor
        query_emb = torch.tensor(self.embed_model._embed(query)[0])
        node_texts = [n.text for n in nodes]
        node_embs = torch.tensor(self.embed_model._embed(node_texts))

        scores = torch.nn.functional.cosine_similarity(query_emb.unsqueeze(0), node_embs)
        ranked = sorted(zip(nodes, scores.tolist()), key=lambda x: x[1], reverse=True)
        return [n for n, _ in ranked[:top_k]]

    def context_filter(self, nodes, diversity_threshold=0.8):
        unique_nodes, seen = [], []
        for n in nodes:
            text = n.text.strip()
            if not text:
                continue
            # Use _embed and convert to tensor
            emb = torch.tensor(self.embed_model._embed(text[:512])[0])
            if all(torch.nn.functional.cosine_similarity(emb, s, dim=0) < diversity_threshold for s in seen):
                unique_nodes.append(n)
                seen.append(emb)
        return unique_nodes

    def compress_contexts(self, nodes, max_len=1500):
        texts = []
        total_len = 0
        for n in nodes:
            t = n.text.strip()
            if total_len + len(t) > max_len:
                break
            texts.append(t)
            total_len += len(t)
        return "\n\n".join(texts)



class GenerationModule:
    def __init__(self, llm, verify: bool = True):
        self.llm = llm
        self.verify = verify

    def generate_with_fallback(self, query: str, domain: str = "General", context: str = "") -> str:
        fallback_prompt = f"""
        Ìwọ jẹ́ ọ̀jọ̀gbọ́n nínú {domain}. Fún ìdáhùn kan sí ìbéèrè yìí ní èdè Yorùbá:
        Ìbéèrè: {query}
        Jọwọ pèsè ìdáhùn tó dájú.
        Answer in clear Yoruba with proper paragraph spacing
        """
        resp = safe_llm_complete(self.llm, fallback_prompt)
        if resp:
            return resp 
        return "⚠️ Ko ṣee ṣe lati gba ìdáhùn lọwọ LLM."

    def generate(self, query: str, context: str, domain: str = "General") -> str:
        prompt = build_yoruba_prompt(query=query, context=context, domain=domain)
        raw = safe_llm_complete(self.llm, prompt)
        if raw is None:
            return "⚠️ Ko ṣee ṣe lati gba ìdáhùn lọwọ LLM."
        return raw.strip()
# -----------------------------
# Dummy retriever fallback
# -----------------------------
class DummyRetriever:
    def retrieve(self, queries, top_k=10):
        return []


class OrchestrationModule:
    """Coordinates all plug-in modules."""
    def __init__(self, pre, retriever, post, generator,llm_fallback_threshold: float = 0.3):
        self.pre = pre
        self.retriever = retriever
        self.post = post
        self.generator = generator
        self.llm_fallback_threshold = llm_fallback_threshold
    def modular_query(self, question: str, domain: str = "General", top_k: int = 5):
        query = normalize_yoruba(question)
        queries=self.pre.process(query)
        print(f"🔍 Expanded Queries: {queries}")
        nodes=[]
        if self.retriever:
            nodes= self.retriever.retrieve(queries, top_k=top_k)
        if not nodes:
            print("⚠️ No documents found. Switching to LLM Fallback.")
            return {
                "question": question,
                "answer": self.generator.generate_with_fallback(question, domain, ""),
                "context": "",
                "num_docs": 0,
                "expanded_queries": queries,
                "mode": "llm_fallback",
                "show_warning": True
            }
        print(f"📚 Retrieved {len(nodes)} documents")
         # Step 3: Post-Retrieval Filtering
        joined_query = " ".join(queries)
        nodes = self.post.rerank(nodes, joined_query, top_k=top_k)
        nodes = self.post.context_filter(nodes)
        context = self.post.compress_contexts(nodes)

        # Step 4: Generate
        answer = self.generator.generate(joined_query, context)

        # Step 5: Verification
        # The generate method handles internal verification based on self.generator.verify
        verified = self.generator.verify

        return {
            "question": question,
            "answer": answer,
            "verified": verified,
            "context": context,
            "num_docs": len(nodes),
            "expanded_queries": queries,
            "mode": "retrieval_augmented",
            "show_warning": False
        }

    


# -----------------------------
# Gemini loader (cached)
# -----------------------------
@lru_cache(maxsize=4)
def load_gemini_llm(api_key: str, model: str = "models/gemini-2.5-flash"):
    try:
        from llama_index.llms.gemini import Gemini

        llm = Gemini(model=model, api_key=api_key)
        logger.info("Gemini LLM loaded")
        return llm
    except Exception as e:
        logger.warning(f"Could not load Gemini LLM: {e}")
        return None




def build_yoruba_prompt(query: str, context: str = "", domain: str = "General") -> str:
    if context:
        prompt = f"""
        Ìwọ jẹ́ ọ̀jọ̀gbọ́n nínú {domain}. Fún ìdáhùn kan sí ìbéèrè yìí ní èdè Yorùbá nípa lílo àkíyèsí àwọn ìwé ìtọ́ni tí a fún ní ìsàlẹ̀.

        Àwọn ìwé ìtọ́ni (Context):
        {context}

        Ìbéèrè: {query}

        Jọwọ:
        1. Dáhùn ní èdè Yorùbá
        2. Bá ìbéèrè mu
        3. Tó o jẹ́ òtító
        4. Answer in clear Yoruba with proper paragraph spacing.

        Ìdáhùn:
        """
    else:
        prompt = f"""
        Ìwọ jẹ́ ọ̀jọ̀gbọ́n nínú {domain}. Fún ìdáhùn kan sí ìbéèrè yìí ní èdè Yorùbá:

        Ìbéèrè: {query}

        Jọwọ:
        1. Dáhùn ní èdè Yorùbá
        2. Bá ìbéèrè mu
        3. Tó o jẹ́ òtító
        4. Answer in clear Yoruba with proper paragraph spacing.

        Ìdáhùn:
        """
    return prompt

import chainlit as cl
from chainlit.types import ThreadDict
from chainlit.input_widget import Select
import asyncio
# -----------------------------
# STARTUP: Chainlit handlers
# -----------------------------
DOMAINS = ["Entertainment", "Current Affairs", "Social Life", "Culture", "Religion"]


@cl.on_chat_start
async def start():
    """Load heavy resources lazily here. This reduces top-level startup time and avoids Docker startup timeouts.
    """
    await cl.Message("🔧 Initializing application — this may take a few seconds...").send()

    
    state = {}

    # 1) Load embedding model lazily
    from llama_index.core import Settings
    try:
        embedder = AfriBERTaEmbedding()
        Settings.embed_model = embedder

        # Do not call embedder.load() synchronously to avoid long startup; but the first call will load it.
        state["embedder"] = Settings.embed_model
    except Exception as e:
        logger.warning(f"Embedding init failed: {e}")
        state["embedder"] = None


    # 2) Load or fallback LLM
    llm = None
    if GEMINI_API_KEY:
        Settings.llm = load_gemini_llm(GEMINI_API_KEY)
    else:
        logger.info("GEMINI_API_KEY not set. LLM will be None (fallback to safe messages).")

    # 3) Setup simple modules (no vector store unless weaviate configured)
    generation_module = GenerationModule(llm=Settings.llm, verify=True)

    # If weaviate is available, try to create a retriever
    retrieval_module = None
    try:
        if WEAVIATE_URL and WEAVIATE_API_KEY:
            import weaviate
            from llama_index.vector_stores.weaviate import WeaviateVectorStore
            from llama_index.core import StorageContext, VectorStoreIndex

            client = weaviate.connect_to_weaviate_cloud(
                cluster_url=WEAVIATE_URL,
                auth_credentials=weaviate.auth.AuthApiKey(WEAVIATE_API_KEY),
                skip_init_checks=True,
            )
            agent = QueryAgent(client=client,
                               collections=[
                                   QueryAgentCollectionConfig(
                                        name="Yoruba_rag",
                                        ),
                                        ],
                                        )
            


            # Build minimal vector store wrapper — this assumes index already exists in Weaviate
            vector_store = WeaviateVectorStore(weaviate_client=client, index_name="Yoruba_rag")
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            vector_index = VectorStoreIndex.from_vector_store(vector_store=vector_store, storage_context=storage_context, embed_model=embedder)
            retriever = WeaviateAgentRetriever(agent=agent)
            retrieval_module = RetrievalModule(dense_retriever=retriever,sparse_retriever=None)
            logger.info("Weaviate retriever initialized")
        else:
            logger.info("Weaviate not configured — continuing without a retriever.")
    except Exception as e:
        logger.warning(f"Weaviate initialization failed: {e}")
        retrieval_module = None
    
    # 3) Pre module
    pre = PreRetrievalModule(llm=Settings.llm, enable_hyde=True)
    # 4) Post module
    post_module = PostRetrievalModule(embed_model=state.get("embedder"))

    
    # 5) Orchestrator
    orchestrator = OrchestrationModule(pre=pre, retriever=retrieval_module, post=post_module, generator=generation_module)
    # Save into session
    cl.user_session.set("state", state)
    cl.user_session.set("orchestrator", orchestrator)
    cl.user_session.set("settings", {"domain": DOMAINS[0]})
    
  
    # Send a friendly welcome with domain selector
    settings = await cl.ChatSettings(
        [
            Select(id="domain", label="Select Domain (Ọ̀nà ìbéèrè)", values=DOMAINS, initial_index=0),
        ]
    ).send()
    
    # ... rest of your welcome message ...

    # Welcome message in both Yoruba and English
    welcome_message = """
## 🇳🇬 Kaabo! | Welcome to the Yorùbá Question Answering Assistant
This assistant helps you ask and receive answers in **Yorùbá**, powered by advanced artificial intelligence and curated knowledge sources.

Please help us improve by evaluating this response.
[👉 Click here to fill the form](https://forms.gle/owiYWgNgoeLtjr3N7)
---

### 📘 Bí o ṣe lè lo ìrànlọ́wọ́ yìí | How to Use This Assistant

1. 📁 **Yan apakan (Domain)** — Select a domain using the settings (⚙️).
2. 💬 **Beere ìbéèrè rẹ** — Ask your question in Yorùbá (or English if needed).
3. 🔍 **Gba ìdáhùn tó péye** — The system retrieves relevant information and generates a clear response.

---

### 🌍 Àwọn Apakan tí ó wà | Available Domains

- 🎬 **Entertainment** — Movies, music, sports, and popular culture  
- 📰 **Current Affairs** — News, politics, economy  
- 👥 **Social Life** — Relationships, community, etiquette  
- 🎭 **Culture** — Traditions, history, festivals  
- 🙏 **Religion** — Beliefs, practices, spirituality  

---

---

#### ✅ Bẹrẹ nípa yíyan apakan kan, lẹ́yìn náà beere ìbéèrè rẹ  
#### Start by selecting a domain and asking your question

"""
   
    await cl.Message(content=welcome_message).send()
    
    
# ============================================================
@cl.on_settings_update
async def load_thread(settings):
    pass  # not needed yet

@cl.on_message
async def main(message: cl.Message):
    question = message.content.strip()
    if not question:
        await cl.Message(content="⚠️ Jọwọ beere ìbéèrè kan — ask a question.").send()
        return

    orchestrator: OrchestrationModule = cl.user_session.get("orchestrator")
    settings = cl.user_session.get("settings") or {"domain": DOMAINS[0]}
    domain = settings.get("domain", DOMAINS[0])

    # Send an initial typing message and stream tokens
    msg = await cl.Message(content="🔄 Processing your question...").send()

    # If orchestrator not available, reply with fallback
    if orchestrator is None:
        await msg.update(content="⚠️ System not initialized. Try again shortly.")
        return

    # Run query (synchronous call inside async — if heavy, consider running in threadpool)
    try:
        result = await asyncio.to_thread(
    orchestrator.modular_query,
    question,
    domain,
    20
)
        
    except Exception as e:
        logger.exception("Query failed")
        await msg.update(content=f"⚠️ Query failed: {e}")
        return

    answer = result.get("answer", "⚠️ Ko ṣee ṣe lati gba ìdáhùn.")
    show_warning = result.get("show_warning", False)
    
    mode_text = "Retrieved from Database" if result["mode"] == "retrieval_augmented" else "LLM Knowledge Base (Fallback)"
    msg.content = f"**Answer:** {result['answer']}\n\n*Source: {mode_text}*"
    await msg.update()




# -----------------------------
# If run as script, expose entrypoint name (useful for local testing)
# -----------------------------
if __name__ == "__main__":
    print("This file is intended to be run with: chainlit run fixed_chainlit_app.py")
