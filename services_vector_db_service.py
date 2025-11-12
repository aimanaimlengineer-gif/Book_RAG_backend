"""
Vector Database Service - FAISS and Pinecone integration for similarity search
"""
import logging
import asyncio
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import json
import os
import pickle
from pathlib import Path

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None

try:
    import pinecone
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False
    pinecone = None

logger = logging.getLogger(__name__)

class VectorDBService:
    """Service for vector storage and similarity search using FAISS or Pinecone"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        self.provider = self.config.get("provider", "faiss")
        self.dimension = self.config.get("dimension", 384)
        self.index_path = self.config.get("faiss_index_path", "./data/vector_index")
        
        # Initialize components
        self.index = None
        self.metadata_store = {}
        self.vector_count = 0
        
        # Create directories
        Path(self.index_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize provider
        self._initialize_provider()
        
        logger.info(f"Vector DB service initialized with provider: {self.provider}")
    
    def _initialize_provider(self):
        """Initialize the vector database provider"""
        
        if self.provider == "faiss":
            if not FAISS_AVAILABLE:
                logger.error("FAISS not available, using basic vector storage")
                self.provider = "basic"
                self._initialize_basic_storage()
                return
            
            try:
                self._initialize_faiss()
                logger.info("FAISS index initialized successfully")
            except Exception as e:
                logger.error(f"FAISS initialization failed: {str(e)}")
                self.provider = "basic"
                self._initialize_basic_storage()
        
        elif self.provider == "pinecone":
            if not PINECONE_AVAILABLE:
                logger.error("Pinecone not available, falling back to FAISS")
                self.provider = "faiss"
                self._initialize_provider()
                return
            
            try:
                self._initialize_pinecone()
                logger.info("Pinecone index initialized successfully")
            except Exception as e:
                logger.error(f"Pinecone initialization failed: {str(e)}")
                self.provider = "faiss"
                self._initialize_provider()
        
        else:
            logger.warning(f"Unknown provider {self.provider}, using basic storage")
            self.provider = "basic"
            self._initialize_basic_storage()
    
    def _initialize_faiss(self):
        """Initialize FAISS index"""
        
        # Try to load existing index
        index_file = f"{self.index_path}.index"
        metadata_file = f"{self.index_path}.metadata"
        
        if os.path.exists(index_file) and os.path.exists(metadata_file):
            try:
                self.index = faiss.read_index(index_file)
                with open(metadata_file, 'rb') as f:
                    self.metadata_store = pickle.load(f)
                
                self.vector_count = self.index.ntotal
                logger.info(f"Loaded existing FAISS index with {self.vector_count} vectors")
                return
            
            except Exception as e:
                logger.warning(f"Failed to load existing index: {str(e)}")
        
        # Create new index
        # Using IndexFlatIP for cosine similarity (inner product on normalized vectors)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata_store = {}
        self.vector_count = 0
        
        logger.info(f"Created new FAISS index with dimension {self.dimension}")
    
    def _initialize_pinecone(self):
        """Initialize Pinecone index"""
        
        api_key = self.config.get("pinecone_api_key")
        environment = self.config.get("pinecone_environment", "us-east1-gcp")
        index_name = self.config.get("pinecone_index_name", "agentic-rag-vectors")
        
        if not api_key:
            raise ValueError("Pinecone API key not provided")
        
        # Initialize Pinecone
        pinecone.init(api_key=api_key, environment=environment)
        
        # Create or connect to index
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(
                name=index_name,
                dimension=self.dimension,
                metric="cosine"
            )
            logger.info(f"Created new Pinecone index: {index_name}")
        
        self.index = pinecone.Index(index_name)
        
        # Get vector count
        stats = self.index.describe_index_stats()
        self.vector_count = stats.total_vector_count
        
        logger.info(f"Connected to Pinecone index with {self.vector_count} vectors")
    
    def _initialize_basic_storage(self):
        """Initialize basic in-memory vector storage"""
        
        self.index = {
            "vectors": [],
            "metadata": []
        }
        self.vector_count = 0
        
        # Try to load from file
        storage_file = f"{self.index_path}.basic"
        if os.path.exists(storage_file):
            try:
                with open(storage_file, 'rb') as f:
                    data = pickle.load(f)
                    self.index = data["index"]
                    self.metadata_store = data["metadata"]
                    self.vector_count = len(self.index["vectors"])
                
                logger.info(f"Loaded basic storage with {self.vector_count} vectors")
            except Exception as e:
                logger.warning(f"Failed to load basic storage: {str(e)}")
    
    async def add_vectors(
        self,
        vectors: List[np.ndarray],
        metadata: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """Add vectors to the database"""
        
        if not vectors:
            return []
        
        if len(vectors) != len(metadata):
            raise ValueError("Vectors and metadata must have the same length")
        
        # Generate IDs if not provided
        if ids is None:
            ids = [f"vec_{self.vector_count + i}" for i in range(len(vectors))]
        
        # Ensure vectors are numpy arrays
        vectors = [np.array(v, dtype=np.float32) for v in vectors]
        
        # Normalize vectors for cosine similarity
        normalized_vectors = []
        for v in vectors:
            norm = np.linalg.norm(v)
            if norm > 0:
                normalized_vectors.append(v / norm)
            else:
                normalized_vectors.append(v)
        
        if self.provider == "faiss":
            return await self._add_vectors_faiss(normalized_vectors, metadata, ids)
        elif self.provider == "pinecone":
            return await self._add_vectors_pinecone(normalized_vectors, metadata, ids)
        else:  # basic
            return await self._add_vectors_basic(normalized_vectors, metadata, ids)
    
    async def _add_vectors_faiss(
        self,
        vectors: List[np.ndarray],
        metadata: List[Dict[str, Any]],
        ids: List[str]
    ) -> List[str]:
        """Add vectors to FAISS index"""
        
        # Convert to matrix
        vector_matrix = np.vstack(vectors)
        
        # Add to index
        self.index.add(vector_matrix)
        
        # Store metadata
        for i, (vector_id, meta) in enumerate(zip(ids, metadata)):
            internal_id = self.vector_count + i
            self.metadata_store[internal_id] = {
                "id": vector_id,
                "metadata": meta,
                "added_at": datetime.utcnow().isoformat()
            }
        
        self.vector_count += len(vectors)
        
        # Save index
        await self._save_faiss_index()
        
        return ids
    
    async def _add_vectors_pinecone(
        self,
        vectors: List[np.ndarray],
        metadata: List[Dict[str, Any]],
        ids: List[str]
    ) -> List[str]:
        """Add vectors to Pinecone index"""
        
        # Prepare data for Pinecone
        upsert_data = []
        for vector_id, vector, meta in zip(ids, vectors, metadata):
            upsert_data.append({
                "id": vector_id,
                "values": vector.tolist(),
                "metadata": {
                    **meta,
                    "added_at": datetime.utcnow().isoformat()
                }
            })
        
        # Upsert in batches
        batch_size = 100
        for i in range(0, len(upsert_data), batch_size):
            batch = upsert_data[i:i + batch_size]
            self.index.upsert(vectors=batch)
        
        self.vector_count += len(vectors)
        
        return ids
    
    async def _add_vectors_basic(
        self,
        vectors: List[np.ndarray],
        metadata: List[Dict[str, Any]],
        ids: List[str]
    ) -> List[str]:
        """Add vectors to basic storage"""
        
        for vector_id, vector, meta in zip(ids, vectors, metadata):
            self.index["vectors"].append(vector)
            self.index["metadata"].append({
                "id": vector_id,
                "metadata": meta,
                "added_at": datetime.utcnow().isoformat()
            })
        
        self.vector_count += len(vectors)
        
        # Save to file
        await self._save_basic_storage()
        
        return ids
    
    async def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors"""
        
        # Normalize query vector
        query_vector = np.array(query_vector, dtype=np.float32)
        norm = np.linalg.norm(query_vector)
        if norm > 0:
            query_vector = query_vector / norm
        
        if self.provider == "faiss":
            return await self._search_faiss(query_vector, top_k, filter_metadata)
        elif self.provider == "pinecone":
            return await self._search_pinecone(query_vector, top_k, filter_metadata)
        else:  # basic
            return await self._search_basic(query_vector, top_k, filter_metadata)
    
    async def _search_faiss(
        self,
        query_vector: np.ndarray,
        top_k: int,
        filter_metadata: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Search using FAISS index"""
        
        if self.vector_count == 0:
            return []
        
        # Search index
        query_matrix = query_vector.reshape(1, -1)
        similarities, indices = self.index.search(query_matrix, min(top_k * 2, self.vector_count))
        
        results = []
        for similarity, idx in zip(similarities[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue
            
            metadata_entry = self.metadata_store.get(idx, {})
            metadata = metadata_entry.get("metadata", {})
            
            # Apply filter if specified
            if filter_metadata and not self._matches_filter(metadata, filter_metadata):
                continue
            
            results.append({
                "id": metadata_entry.get("id", f"vec_{idx}"),
                "similarity": float(similarity),
                "metadata": metadata,
                "index": int(idx)
            })
            
            if len(results) >= top_k:
                break
        
        return results
    
    async def _search_pinecone(
        self,
        query_vector: np.ndarray,
        top_k: int,
        filter_metadata: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Search using Pinecone index"""
        
        # Prepare query
        query_params = {
            "vector": query_vector.tolist(),
            "top_k": top_k,
            "include_metadata": True
        }
        
        # Add filter if specified
        if filter_metadata:
            query_params["filter"] = filter_metadata
        
        # Execute search
        search_result = self.index.query(**query_params)
        
        results = []
        for match in search_result.matches:
            results.append({
                "id": match.id,
                "similarity": float(match.score),
                "metadata": match.metadata,
                "index": None  # Pinecone doesn't expose internal indices
            })
        
        return results
    
    async def _search_basic(
        self,
        query_vector: np.ndarray,
        top_k: int,
        filter_metadata: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Search using basic storage"""
        
        if not self.index["vectors"]:
            return []
        
        # Calculate similarities
        similarities = []
        for i, stored_vector in enumerate(self.index["vectors"]):
            similarity = np.dot(query_vector, stored_vector)
            
            metadata_entry = self.index["metadata"][i]
            metadata = metadata_entry.get("metadata", {})
            
            # Apply filter if specified
            if filter_metadata and not self._matches_filter(metadata, filter_metadata):
                continue
            
            similarities.append({
                "id": metadata_entry.get("id", f"vec_{i}"),
                "similarity": float(similarity),
                "metadata": metadata,
                "index": i
            })
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        return similarities[:top_k]
    
    def _matches_filter(self, metadata: Dict[str, Any], filter_metadata: Dict[str, Any]) -> bool:
        """Check if metadata matches filter criteria"""
        
        for key, value in filter_metadata.items():
            if key not in metadata:
                return False
            
            if isinstance(value, list):
                if metadata[key] not in value:
                    return False
            else:
                if metadata[key] != value:
                    return False
        
        return True
    
    async def delete_vectors(self, ids: List[str]) -> bool:
        """Delete vectors by ID"""
        
        if self.provider == "pinecone":
            return await self._delete_vectors_pinecone(ids)
        else:
            logger.warning(f"Vector deletion not implemented for provider: {self.provider}")
            return False
    
    async def _delete_vectors_pinecone(self, ids: List[str]) -> bool:
        """Delete vectors from Pinecone"""
        
        try:
            self.index.delete(ids=ids)
            return True
        except Exception as e:
            logger.error(f"Failed to delete vectors from Pinecone: {str(e)}")
            return False
    
    async def update_vector(
        self,
        vector_id: str,
        vector: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update vector and/or metadata"""
        
        if self.provider == "pinecone":
            return await self._update_vector_pinecone(vector_id, vector, metadata)
        else:
            logger.warning(f"Vector update not implemented for provider: {self.provider}")
            return False
    
    async def _update_vector_pinecone(
        self,
        vector_id: str,
        vector: Optional[np.ndarray],
        metadata: Optional[Dict[str, Any]]
    ) -> bool:
        """Update vector in Pinecone"""
        
        try:
            update_data = {"id": vector_id}
            
            if vector is not None:
                # Normalize vector
                norm = np.linalg.norm(vector)
                if norm > 0:
                    vector = vector / norm
                update_data["values"] = vector.tolist()
            
            if metadata is not None:
                update_data["metadata"] = {
                    **metadata,
                    "updated_at": datetime.utcnow().isoformat()
                }
            
            self.index.upsert(vectors=[update_data])
            return True
            
        except Exception as e:
            logger.error(f"Failed to update vector in Pinecone: {str(e)}")
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        
        stats = {
            "provider": self.provider,
            "dimension": self.dimension,
            "vector_count": self.vector_count,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if self.provider == "pinecone" and self.index:
            try:
                pinecone_stats = self.index.describe_index_stats()
                stats.update({
                    "total_vector_count": pinecone_stats.total_vector_count,
                    "index_fullness": pinecone_stats.index_fullness
                })
            except:
                pass
        
        return stats
    
    async def _save_faiss_index(self):
        """Save FAISS index to disk"""
        
        try:
            index_file = f"{self.index_path}.index"
            metadata_file = f"{self.index_path}.metadata"
            
            faiss.write_index(self.index, index_file)
            
            with open(metadata_file, 'wb') as f:
                pickle.dump(self.metadata_store, f)
            
            logger.debug(f"Saved FAISS index with {self.vector_count} vectors")
            
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {str(e)}")
    
    async def _save_basic_storage(self):
        """Save basic storage to disk"""
        
        try:
            storage_file = f"{self.index_path}.basic"
            
            data = {
                "index": self.index,
                "metadata": self.metadata_store
            }
            
            with open(storage_file, 'wb') as f:
                pickle.dump(data, f)
            
            logger.debug(f"Saved basic storage with {self.vector_count} vectors")
            
        except Exception as e:
            logger.error(f"Failed to save basic storage: {str(e)}")
    
    async def create_collection(self, collection_name: str, dimension: Optional[int] = None) -> bool:
        """Create a new collection/index"""
        
        if self.provider == "pinecone":
            try:
                dimension = dimension or self.dimension
                pinecone.create_index(
                    name=collection_name,
                    dimension=dimension,
                    metric="cosine"
                )
                return True
            except Exception as e:
                logger.error(f"Failed to create Pinecone collection: {str(e)}")
                return False
        else:
            logger.warning(f"Collection creation not implemented for provider: {self.provider}")
            return False
    
    async def list_collections(self) -> List[str]:
        """List available collections/indices"""
        
        if self.provider == "pinecone":
            try:
                return pinecone.list_indexes()
            except Exception as e:
                logger.error(f"Failed to list Pinecone collections: {str(e)}")
                return []
        else:
            return [self.config.get("collection_name", "default")]
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of vector database service"""
        
        health_status = {
            "status": "healthy",
            "provider": self.provider,
            "dimension": self.dimension,
            "vector_count": self.vector_count,
            "timestamp": datetime.utcnow()
        }
        
        # Test basic operations
        try:
            # Test vector addition and search
            test_vector = np.random.rand(self.dimension).astype(np.float32)
            test_id = f"health_check_{datetime.utcnow().timestamp()}"
            
            # Add test vector
            await self.add_vectors([test_vector], [{"test": True}], [test_id])
            
            # Search for test vector
            results = await self.search(test_vector, top_k=1)
            
            if results and len(results) > 0:
                health_status["search_test"] = "passed"
            else:
                health_status["search_test"] = "failed"
                health_status["status"] = "unhealthy"
            
            # Clean up test vector if possible
            await self.delete_vectors([test_id])
            
        except Exception as e:
            health_status["search_test"] = "failed"
            health_status["error"] = str(e)
            health_status["status"] = "unhealthy"
        
        return health_status
    
    async def close(self):
        """Close connections and save data"""
        
        if self.provider == "faiss":
            await self._save_faiss_index()
        elif self.provider == "basic":
            await self._save_basic_storage()
        
        logger.info("Vector DB service closed")
    
    def __del__(self):
        """Cleanup on destruction"""
        try:
            # Save data on destruction
            if self.provider == "faiss" and self.index:
                faiss.write_index(self.index, f"{self.index_path}.index")
            elif self.provider == "basic" and self.index:
                with open(f"{self.index_path}.basic", 'wb') as f:
                    pickle.dump({
                        "index": self.index,
                        "metadata": self.metadata_store
                    }, f)
        except:
            pass