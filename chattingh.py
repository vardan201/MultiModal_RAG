import fitz
import os
import uuid
import sqlite3
from pathlib import Path
from typing import List, Union, Dict, TypedDict, Annotated
from datetime import datetime

import torch
import numpy as np
from PIL import Image
from dotenv import load_dotenv

from transformers import CLIPProcessor, CLIPModel
from transformers import BlipProcessor, BlipForConditionalGeneration
import pytesseract

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

load_dotenv()


# ============ STATE DEFINITION ============
class GraphState(TypedDict):
    """State that flows through the LangGraph workflow"""
    question: str
    chat_history: str
    needs_retrieval: bool
    content_type: str  # NEW: "text", "image", or "both"
    text_documents: List[Document]  # NEW: Separate text docs
    image_documents: List[Document]  # NEW: Separate image docs
    documents: List[Document]
    rewritten_query: str
    answer: str
    session_id: str


# ============ MEMORY MANAGER ============
class MemoryManager:
    """SQLite-based short-term memory with trimming"""
    
    def __init__(self, db_path="chat_memory.db", max_messages=20):
        self.db_path = db_path
        self.max_messages = max_messages
        self.init_db()
    
    def init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                role TEXT,
                content TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()
    
    def add_message(self, session_id: str, role: str, content: str):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO chat_history (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, role, content)
        )
        conn.commit()
        conn.close()
        self.trim_memory(session_id)
    
    def trim_memory(self, session_id: str):
        """Keep only the last max_messages"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT COUNT(*) FROM chat_history WHERE session_id = ?",
            (session_id,)
        )
        count = cursor.fetchone()[0]
        
        if count > self.max_messages:
            cursor.execute("""
                DELETE FROM chat_history 
                WHERE session_id = ? 
                AND id NOT IN (
                    SELECT id FROM chat_history 
                    WHERE session_id = ? 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                )
            """, (session_id, session_id, self.max_messages))
            conn.commit()
        
        conn.close()
    
    def get_history(self, session_id: str, limit: int = None) -> List[Dict]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = """
            SELECT role, content FROM chat_history 
            WHERE session_id = ? 
            ORDER BY timestamp DESC
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        cursor.execute(query, (session_id,))
        rows = cursor.fetchall()
        conn.close()
        
        return [{"role": row[0], "content": row[1]} for row in reversed(rows)]
    
    def clear_session(self, session_id: str):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM chat_history WHERE session_id = ?", (session_id,))
        conn.commit()
        conn.close()


# ============ ENHANCED AGENTIC RAG PIPELINE ============
class AgenticRAGPipeline:
    """
    Enhanced Multimodal RAG with SEPARATE TEXT and IMAGE vector stores
    Smart routing decides which content type to use per query
    """

    def __init__(self, chunk_size=500, chunk_overlap=50, session_id=None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        if session_id is None:
            self.session_id = self.generate_session_id()
            print(f"[INFO] Auto-generated session ID: {self.session_id}")
        else:
            self.session_id = session_id
        
        # NEW: Separate vector stores for text and images
        self.text_vector_store = None
        self.image_vector_store = None
        self.text_retriever = None
        self.image_retriever = None
        
        self.processed_files = []
        self.memory = MemoryManager(max_messages=20)

        # -------- CLIP (OpenAI) --------
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        ).to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )

        # -------- BLIP for Image Captioning --------
        print("[INFO] Loading BLIP model for image captioning...")
        self.blip_processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        self.blip_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(self.device)
        print("[SUCCESS] BLIP model loaded")

        # -------- LLM Components --------
        self.router_llm = ChatGroq(
            model_name="llama-3.3-70b-versatile", temperature=0.1
        )
        self.content_router_llm = ChatGroq(  # NEW: For content type routing
            model_name="llama-3.3-70b-versatile", temperature=0.1
        )
        self.rewriter_llm = ChatGroq(
            model_name="llama-3.3-70b-versatile", temperature=0.3
        )
        self.generator_llm = ChatGroq(
            model_name="llama-3.3-70b-versatile", temperature=0.2
        )

        # -------- BUILD LANGGRAPH WORKFLOW --------
        self.workflow = self.build_graph()

        print(f"[SUCCESS] Enhanced Agentic RAG initialized for session: {self.session_id}")
    
    @staticmethod
    def generate_session_id() -> str:
        """Generate unique session ID based on timestamp and random UUID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        return f"session_{timestamp}_{unique_id}"
    
    @staticmethod
    def generate_session_from_files(file_paths: List[str]) -> str:
        """Generate a unique session ID based on file names"""
        import hashlib
        file_names = "_".join(sorted([Path(f).stem for f in file_paths]))
        file_hash = hashlib.md5(file_names.encode()).hexdigest()[:8]
        return f"session_files_{file_hash}"

    # ---------- TEXT PARSING ----------
    def parse_pdf(self, file_path: str) -> str:
        doc = fitz.open(file_path)
        text = "".join(page.get_text() for page in doc)
        doc.close()
        return text

    def parse_txt(self, file_path: str) -> str:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    def chunk_text(self, text: str) -> List[str]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ".", " "],
        )
        return splitter.split_text(text)

    # ---------- IMAGE EXTRACTION ----------
    def extract_images_from_pdf(
        self, file_path: str, output_dir="extracted_images"
    ) -> List[str]:
        session_dir = os.path.join(output_dir, self.session_id)
        os.makedirs(session_dir, exist_ok=True)
        
        doc = fitz.open(file_path)
        image_paths = []

        for page in doc:
            for img in page.get_images(full=True):
                xref = img[0]
                base = doc.extract_image(xref)
                img_bytes = base["image"]
                ext = base["ext"]
                name = f"{uuid.uuid4()}.{ext}"
                path = os.path.join(session_dir, name)

                with open(path, "wb") as f:
                    f.write(img_bytes)

                image_paths.append(path)

        doc.close()
        return image_paths

    # ---------- OCR WITH PYTESSERACT ----------
    def perform_ocr(self, image_path: str) -> str:
        """Extract text from image using PyTesseract OCR"""
        try:
            image = Image.open(image_path)
            ocr_text = pytesseract.image_to_string(image)
            return ocr_text.strip()
        except Exception as e:
            print(f"[WARNING] OCR failed for {image_path}: {e}")
            return ""

    # ---------- IMAGE CAPTIONING WITH BLIP ----------
    def generate_image_caption(self, image_path: str) -> str:
        """Generate a descriptive caption for an image using BLIP"""
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self.blip_processor(image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                out = self.blip_model.generate(**inputs, max_length=50)
            
            caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
            return caption
        except Exception as e:
            print(f"[WARNING] Caption generation failed for {image_path}: {e}")
            return "Image description unavailable"

    # ---------- COMPREHENSIVE IMAGE PROCESSING ----------
    def process_image_multimodal(self, image_path: str, source_file: str) -> Dict[str, str]:
        """Process a single image with OCR, captioning, and metadata"""
        print(f"[INFO] Processing image: {os.path.basename(image_path)}")
        
        ocr_text = self.perform_ocr(image_path)
        caption = self.generate_image_caption(image_path)
        
        try:
            img = Image.open(image_path)
            width, height = img.size
            format_type = img.format
            metadata = f"Format: {format_type}, Size: {width}x{height}"
        except:
            metadata = "Metadata unavailable"
        
        return {
            "ocr_text": ocr_text,
            "caption": caption,
            "metadata": metadata,
            "image_path": image_path,
            "source": source_file
        }

    # ---------- CREATE RICH MULTIMODAL CONTENT ----------
    def create_multimodal_content(self, image_data: Dict[str, str]) -> str:
        """Merge OCR, caption, and metadata into searchable content"""
        content_parts = []
        
        if image_data["caption"]:
            content_parts.append(f"Image Description: {image_data['caption']}")
        
        if image_data["ocr_text"]:
            content_parts.append(f"Text in Image (OCR): {image_data['ocr_text']}")
        
        content_parts.append(f"Image Details: {image_data['metadata']}")
        
        return "\n".join(content_parts)

    # ---------- CLIP EMBEDDINGS ----------
    def embed_text(self, text: str) -> np.ndarray:
        """Embed text with CLIP, handling max token length"""
        inputs = self.clip_processor(
            text=[text], 
            return_tensors="pt", 
            padding=True,
            truncation=True,
            max_length=77
        ).to(self.device)
        with torch.no_grad():
            emb = self.clip_model.get_text_features(**inputs)
        return emb.cpu().numpy()[0]

    def embed_image(self, image_path: str) -> np.ndarray:
        image = Image.open(image_path).convert("RGB")
        inputs = self.clip_processor(
            images=image, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            emb = self.clip_model.get_image_features(**inputs)
        return emb.cpu().numpy()[0]

    # ---------- ENHANCED FILE PROCESSING WITH SEPARATE STORES ----------
    def process_files(self, file_paths: List[str]) -> Dict[str, int]:
        """
        Process files into SEPARATE text and image vector stores
        Returns counts of text and image chunks processed
        """
        text_documents = []
        text_embeddings = []
        image_documents = []
        image_embeddings = []

        for file_path in file_paths:
            if file_path not in self.processed_files:
                self.processed_files.append(file_path)
            
            ext = Path(file_path).suffix.lower()

            if ext == ".pdf":
                # ========== TEXT EXTRACTION (PRIORITY 1) ==========
                print(f"\n[TEXT] Extracting text from: {os.path.basename(file_path)}")
                text = self.parse_pdf(file_path)
                
                if text.strip():  # Only process if there's actual text
                    chunks = self.chunk_text(text)
                    print(f"   -> Found {len(chunks)} text chunks")

                    for chunk in chunks:
                        text_documents.append(
                            Document(
                                page_content=chunk,
                                metadata={
                                    "type": "text", 
                                    "source": file_path,
                                    "session_id": self.session_id
                                },
                            )
                        )
                        text_embeddings.append(self.embed_text(chunk))
                else:
                    print(f"   -> No text found in PDF")

                # ========== IMAGE EXTRACTION (PRIORITY 2) ==========
                print(f"\n[IMAGE] Extracting images from: {os.path.basename(file_path)}")
                image_paths = self.extract_images_from_pdf(file_path)
                print(f"   -> Found {len(image_paths)} images")
                
                for img_path in image_paths:
                    image_data = self.process_image_multimodal(img_path, file_path)
                    multimodal_content = self.create_multimodal_content(image_data)
                    
                    # Store in IMAGE vector store
                    image_documents.append(
                        Document(
                            page_content=multimodal_content,
                            metadata={
                                "type": "image",
                                "image_path": img_path,
                                "source": file_path,
                                "session_id": self.session_id,
                                "has_ocr": bool(image_data["ocr_text"]),
                                "has_caption": bool(image_data["caption"])
                            },
                        )
                    )
                    # Use TEXT embedding for OCR/caption searchability
                    image_embeddings.append(self.embed_text(multimodal_content))
                    
                    print(f"   [OK] {os.path.basename(img_path)} "
                          f"(OCR: {bool(image_data['ocr_text'])}, "
                          f"Caption: {bool(image_data['caption'])})")

            elif ext == ".txt":
                text = self.parse_txt(file_path)
                chunks = self.chunk_text(text)

                for chunk in chunks:
                    text_documents.append(
                        Document(
                            page_content=chunk,
                            metadata={
                                "type": "text", 
                                "source": file_path,
                                "session_id": self.session_id
                            },
                        )
                    )
                    text_embeddings.append(self.embed_text(chunk))

        # ========== CREATE SEPARATE VECTOR STORES ==========
        try:
            from langchain_core.embeddings import Embeddings
        except ImportError:
            from langchain.embeddings.base import Embeddings
        
        class CLIPEmbeddings(Embeddings):
            def __init__(self, parent):
                self.parent = parent
            
            def embed_documents(self, texts):
                return [self.parent.embed_text(text) for text in texts]
            
            def embed_query(self, text):
                return self.parent.embed_text(text)
        
        clip_embeddings = CLIPEmbeddings(self)

        # Create/update TEXT vector store
        if text_documents:
            text_vectors = np.vstack(text_embeddings)
            text_pairs = [
                (doc.page_content, vector) 
                for doc, vector in zip(text_documents, text_vectors)
            ]
            
            if self.text_vector_store is None:
                self.text_vector_store = FAISS.from_embeddings(
                    text_pairs, clip_embeddings
                )
                for i, doc in enumerate(text_documents):
                    self.text_vector_store.docstore._dict[
                        self.text_vector_store.index_to_docstore_id[i]
                    ] = doc
            else:
                self.text_vector_store.add_embeddings(text_pairs)
                current_index = len(self.text_vector_store.index_to_docstore_id)
                for i, doc in enumerate(text_documents):
                    self.text_vector_store.docstore._dict[
                        self.text_vector_store.index_to_docstore_id[current_index + i]
                    ] = doc
            
            self.text_retriever = self.text_vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 5, "fetch_k": 15, "lambda_mult": 0.7}
            )

        # Create/update IMAGE vector store
        if image_documents:
            image_vectors = np.vstack(image_embeddings)
            image_pairs = [
                (doc.page_content, vector) 
                for doc, vector in zip(image_documents, image_vectors)
            ]
            
            if self.image_vector_store is None:
                self.image_vector_store = FAISS.from_embeddings(
                    image_pairs, clip_embeddings
                )
                for i, doc in enumerate(image_documents):
                    self.image_vector_store.docstore._dict[
                        self.image_vector_store.index_to_docstore_id[i]
                    ] = doc
            else:
                self.image_vector_store.add_embeddings(image_pairs)
                current_index = len(self.image_vector_store.index_to_docstore_id)
                for i, doc in enumerate(image_documents):
                    self.image_vector_store.docstore._dict[
                        self.image_vector_store.index_to_docstore_id[current_index + i]
                    ] = doc
            
            self.image_retriever = self.image_vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 5, "fetch_k": 15, "lambda_mult": 0.7}
            )
        
        print(f"\n{'='*60}")
        print(f"[SUCCESS] Processed documents for session: {self.session_id}")
        print(f"   - TEXT chunks: {len(text_documents)}")
        print(f"   - IMAGE chunks: {len(image_documents)}")
        print(f"   - TOTAL: {len(text_documents) + len(image_documents)}")
        print(f"{'='*60}\n")

        return {
            "text_chunks": len(text_documents),
            "image_chunks": len(image_documents),
            "total": len(text_documents) + len(image_documents)
        }

    # ========== LANGGRAPH NODES ==========
    
    def my_ai_assistant_node(self, state: GraphState) -> GraphState:
        """
        Node 1: My_AI_Assistant (Router)
        Decides if retrieval is needed AND what content type to retrieve
        """
        print("\n[My_AI_Assistant] Routing query...")
        
        history = self.memory.get_history(state["session_id"], limit=4)
        history_text = "\n".join(
            f"{msg['role']}: {msg['content']}" for msg in history
        )
        
        # First: Check if retrieval is needed
        prompt = PromptTemplate(
            input_variables=["question", "history"],
            template="""You are an AI assistant router. Decide if the question needs document retrieval.

Chat History:
{history}

Question: {question}

Reply with ONLY one word: 'RETRIEVE' or 'DIRECT'
- RETRIEVE: if the question needs information from documents
- DIRECT: if it's a general question or greeting

Decision:"""
        )
        
        chain = prompt | self.router_llm | StrOutputParser()
        decision = chain.invoke({
            "question": state["question"],
            "history": history_text
        }).strip()
        
        state["needs_retrieval"] = "RETRIEVE" in decision.upper()
        state["chat_history"] = history_text
        
        # Second: If retrieval needed, determine content type
        if state["needs_retrieval"]:
            content_prompt = PromptTemplate(
                input_variables=["question"],
                template="""Analyze the question and determine what type of content is needed.

Question: {question}

Reply with ONLY one word:
- TEXT: if asking about written content, explanations, definitions, text-based information
- IMAGE: if asking about visual content, diagrams, charts, pictures, screenshots, visual elements
- BOTH: if the answer could come from either text or images

Examples:
"What does the document say about XYZ?" -> TEXT
"Show me the diagram of XYZ" -> IMAGE
"Explain the architecture" -> BOTH (could be text explanation or diagram)
"What's in the chart?" -> IMAGE

Decision:"""
            )
            
            content_chain = content_prompt | self.content_router_llm | StrOutputParser()
            content_type = content_chain.invoke({"question": state["question"]}).strip().upper()
            
            if "TEXT" in content_type and "IMAGE" not in content_type:
                state["content_type"] = "text"
            elif "IMAGE" in content_type and "TEXT" not in content_type:
                state["content_type"] = "image"
            else:
                state["content_type"] = "both"
        else:
            state["content_type"] = "none"
        
        print(f"   -> Retrieval: {'YES' if state['needs_retrieval'] else 'NO'}")
        print(f"   -> Content Type: {state['content_type']}")
        
        return state
    
    def vector_retriever_node(self, state: GraphState) -> GraphState:
        """
        Node 2: Vector_Retriever
        Retrieves from TEXT or IMAGE stores based on content_type
        """
        print(f"\n[Vector_Retriever] Retrieving {state['content_type']} content...")
        
        state["text_documents"] = []
        state["image_documents"] = []
        state["documents"] = []
        
        content_type = state["content_type"]
        
        # Retrieve from TEXT store
        if content_type in ["text", "both"] and self.text_retriever:
            print("   -> Searching TEXT store...")
            text_docs = self.text_retriever.invoke(state["question"])
            state["text_documents"] = text_docs
            print(f"   -> Found {len(text_docs)} text chunks")
        
        # Retrieve from IMAGE store
        if content_type in ["image", "both"] and self.image_retriever:
            print("   -> Searching IMAGE store...")
            image_docs = self.image_retriever.invoke(state["question"])
            state["image_documents"] = image_docs
            print(f"   -> Found {len(image_docs)} image chunks")
        
        # Combine based on content type
        if content_type == "text":
            state["documents"] = state["text_documents"][:5]
        elif content_type == "image":
            state["documents"] = state["image_documents"][:5]
        else:  # both
            # Interleave text and image for balanced context
            combined = []
            max_len = max(len(state["text_documents"]), len(state["image_documents"]))
            for i in range(max_len):
                if i < len(state["text_documents"]):
                    combined.append(state["text_documents"][i])
                if i < len(state["image_documents"]):
                    combined.append(state["image_documents"][i])
            state["documents"] = combined[:7]
        
        print(f"   -> TOTAL retrieved: {len(state['documents'])} chunks")
        
        return state
    
    def query_rewriter_node(self, state: GraphState) -> GraphState:
        """Node 3: Query_Rewriter"""
        print("\n[Query_Rewriter] Rewriting query...")
        
        if not state["documents"]:
            state["rewritten_query"] = state["question"]
            return state
        
        context_preview = "\n".join(
            d.page_content[:100] for d in state["documents"][:3]
        )
        
        prompt = PromptTemplate(
            input_variables=["question", "history", "context"],
            template="""Rewrite the question to be more specific based on context and history.

Chat History:
{history}

Available Context:
{context}

Original Question: {question}

Rewritten Question:"""
        )
        
        chain = prompt | self.rewriter_llm | StrOutputParser()
        rewritten = chain.invoke({
            "question": state["question"],
            "history": state["chat_history"],
            "context": context_preview
        })
        
        state["rewritten_query"] = rewritten.strip()
        
        print(f"   -> Rewritten: {state['rewritten_query']}")
        
        return state
    
    def output_generator_node(self, state: GraphState) -> GraphState:
        """Node 4: Output_Generator"""
        print("\n[Output_Generator] Generating answer...")
        
        if state["needs_retrieval"] and state["documents"]:
            context = "\n\n".join(d.page_content for d in state["documents"])
            
            prompt = PromptTemplate(
                input_variables=["context", "history", "question", "rewritten", "content_type"],
                template="""You are a helpful AI assistant. Answer based on the context and chat history.

Content Type Retrieved: {content_type}
- If "text": context contains text from documents
- If "image": context contains OCR text and image descriptions
- If "both": context contains both text and image information

Chat History:
{history}

Context:
{context}

Original Question: {question}
Clarified Question: {rewritten}

Provide a clear, concise answer. If the context doesn't contain the answer, say so.
If answering from image content, mention that the information comes from images/diagrams.

Answer:"""
            )
            
            chain = prompt | self.generator_llm | StrOutputParser()
            answer = chain.invoke({
                "context": context,
                "history": state["chat_history"],
                "question": state["question"],
                "rewritten": state["rewritten_query"],
                "content_type": state["content_type"]
            })
        else:
            prompt = PromptTemplate(
                input_variables=["history", "question"],
                template="""You are a helpful AI assistant.

Chat History:
{history}

Question: {question}

Answer:"""
            )
            
            chain = prompt | self.generator_llm | StrOutputParser()
            answer = chain.invoke({
                "history": state["chat_history"],
                "question": state["question"]
            })
        
        state["answer"] = answer.strip()
        
        print("   -> Answer generated")
        
        return state
    
    # ========== ROUTING LOGIC ==========
    
    def route_after_assistant(self, state: GraphState) -> str:
        """Decide whether to retrieve or go directly to generator"""
        if state["needs_retrieval"] and (self.text_retriever or self.image_retriever):
            return "retriever"
        else:
            return "generator"
    
    # ========== BUILD LANGGRAPH ==========
    
    def build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(GraphState)
        
        workflow.add_node("assistant", self.my_ai_assistant_node)
        workflow.add_node("retriever", self.vector_retriever_node)
        workflow.add_node("rewriter", self.query_rewriter_node)
        workflow.add_node("generator", self.output_generator_node)
        
        workflow.set_entry_point("assistant")
        
        workflow.add_conditional_edges(
            "assistant",
            self.route_after_assistant,
            {
                "retriever": "retriever",
                "generator": "generator"
            }
        )
        
        workflow.add_edge("retriever", "rewriter")
        workflow.add_edge("rewriter", "generator")
        workflow.add_edge("generator", END)
        
        return workflow.compile()
    
    # ========== MAIN INTERFACE ==========
    
    def ask(self, question: str) -> dict:
        """Execute the LangGraph workflow"""
        print("\n" + "="*60)
        print("Starting Enhanced Agentic RAG Workflow")
        print("="*60)
        
        initial_state = GraphState(
            question=question,
            chat_history="",
            needs_retrieval=False,
            content_type="none",
            text_documents=[],
            image_documents=[],
            documents=[],
            rewritten_query=question,
            answer="",
            session_id=self.session_id
        )
        
        final_state = self.workflow.invoke(initial_state)
        
        self.memory.add_message(self.session_id, "human", question)
        self.memory.add_message(self.session_id, "ai", final_state["answer"])
        
        print("\n" + "="*60)
        print("[SUCCESS] Workflow Complete")
        print("="*60 + "\n")
        
        return {
            "answer": final_state["answer"],
            "documents": final_state["documents"],
            "needed_retrieval": final_state["needs_retrieval"],
            "content_type": final_state.get("content_type"),
            "text_docs_count": len(final_state.get("text_documents", [])),
            "image_docs_count": len(final_state.get("image_documents", [])),
            "rewritten_query": final_state.get("rewritten_query") if final_state.get("rewritten_query") != question else None
        }

    def clear_memory(self):
        """Clear chat history from SQLite"""
        self.memory.clear_session(self.session_id)
        print(f"[SUCCESS] Memory cleared for session: {self.session_id}")
    
    def get_session_info(self) -> Dict:
        """Get information about current session"""
        history = self.memory.get_history(self.session_id)
        return {
            "session_id": self.session_id,
            "processed_files": self.processed_files,
            "message_count": len(history),
            "has_text_retriever": self.text_retriever is not None,
            "has_image_retriever": self.image_retriever is not None
        }


# ============ EXAMPLE USAGE ============
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = AgenticRAGPipeline()
    
    # Process documents
    files = ["FEES 6TH SEM.pdf"]  # Your PDF with text AND images
    stats = pipeline.process_files(files)
    
    print(f"\nProcessing complete!")
    print(f"Text chunks: {stats['text_chunks']}")
    print(f"Image chunks: {stats['image_chunks']}")
    
    # Test queries
    print("\n" + "="*60)
    print("TESTING QUERIES")
    print("="*60)
    
    # Text-focused query
    result1 = pipeline.ask("What does the document say about machine learning?")
    print(f"\nAnswer: {result1['answer']}")
    print(f"Content type used: {result1['content_type']}")
    print(f"Text docs: {result1['text_docs_count']}, Image docs: {result1['image_docs_count']}")
    
    # Image-focused query
    result2 = pipeline.ask("Show me the diagram or chart about the architecture")
    print(f"\nAnswer: {result2['answer']}")
    print(f"Content type used: {result2['content_type']}")
    print(f"Text docs: {result2['text_docs_count']}, Image docs: {result2['image_docs_count']}")
    
    # Mixed query
    result3 = pipeline.ask("Explain the system architecture")
    print(f"\nAnswer: {result3['answer']}")
    print(f"Content type used: {result3['content_type']}")
    print(f"Text docs: {result3['text_docs_count']}, Image docs: {result3['image_docs_count']}")