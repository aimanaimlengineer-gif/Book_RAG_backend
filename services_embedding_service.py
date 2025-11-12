"""
Embedding Service - Semantic embeddings and similarity search
"""
import logging
import asyncio
import numpy as np
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import pickle
import os
from pathlib import Path

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    openai = None

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Service for generating and managing semantic embeddings"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        self.provider = self.config.get("provider", "sentence_transformers")
        self.model_name = self.config.get("model_name", "all-MiniLM-L6-v2")
        self.dimension = self.config.get("dimension", 384)
        self.cache_path = self.config.get("cache_path", "./cache/embeddings")
        
        # Initialize model
        self.model = None
        self.embedding_cache = {}
        
        # Create cache directory
        Path(self.cache_path).mkdir(parents=True, exist_ok=True)
        
        # Initialize embedding provider
        self._initialize_provider()
        
        # Load cache if exists
        self._load_embedding_cache()
        
        logger.info(f"Embedding service initialized with provider: {self.provider}")
    
    def _initialize_provider(self):
        """Initialize the embedding provider"""
        
        if self.provider == "sentence_transformers":
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                logger.error("Sentence Transformers not available, falling back to basic embeddings")
                self.provider = "basic"
                return
            
            try:
                self.model = SentenceTransformer(self.model_name)
                self.dimension = self.model.get_sentence_embedding_dimension()
                logger.info(f"Loaded Sentence Transformer model: {self.model_name}")
                
            except Exception as e:
                logger.error(f"Failed to load Sentence Transformer model: {str(e)}")
                self.provider = "basic"
        
        elif self.provider == "openai":
            if not OPENAI_AVAILABLE or not self.config.get("openai_api_key"):
                logger.error("OpenAI not available, falling back to basic embeddings")
                self.provider = "basic"
                return
            
            try:
                openai.api_key = self.config["openai_api_key"]
                self.model_name = self.config.get("openai_embedding_model", "text-embedding-ada-002")
                self.dimension = 1536  # OpenAI ada-002 dimension
                logger.info(f"Initialized OpenAI embeddings with model: {self.model_name}")
                
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI embeddings: {str(e)}")
                self.provider = "basic"
        
        elif self.provider == "basic":
            logger.info("Using basic hash-based embeddings (fallback)")
            self.dimension = self.config.get("dimension", 384)
        
        else:
            logger.warning(f"Unknown provider {self.provider}, using basic embeddings")
            self.provider = "basic"
            self.dimension = 384
    
    async def generate_embedding(self, text: str, normalize: bool = True) -> np.ndarray:
        """Generate embedding for a single text"""
        
        if not text or not text.strip():
            return np.zeros(self.dimension, dtype=np.float32)
        
        # Check cache first
        cache_key = self._get_cache_key(text)
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        try:
            if self.provider == "sentence_transformers":
                embedding = await self._generate_sentence_transformer_embedding(text)
            elif self.provider == "openai":
                embedding = await self._generate_openai_embedding(text)
            else:  # basic provider
                embedding = self._generate_basic_embedding(text)
            
            if normalize and np.linalg.norm(embedding) > 0:
                embedding = embedding / np.linalg.norm(embedding)
            
            # Cache the result
            self.embedding_cache[cache_key] = embedding
            
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {str(e)}")
            # Return zero vector as fallback
            return np.zeros(self.dimension, dtype=np.float32)
    
    async def generate_embeddings(self, texts: List[str], normalize: bool = True) -> List[np.ndarray]:
        """Generate embeddings for multiple texts"""
        
        embeddings = []
        
        # Check which texts are already cached
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text)
            if cache_key in self.embedding_cache:
                embeddings.append(self.embedding_cache[cache_key])
            else:
                embeddings.append(None)  # Placeholder
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Generate embeddings for uncached texts
        if uncached_texts:
            try:
                if self.provider == "sentence_transformers":
                    new_embeddings = await self._generate_sentence_transformer_embeddings(uncached_texts)
                elif self.provider == "openai":
                    new_embeddings = await self._generate_openai_embeddings(uncached_texts)
                else:  # basic provider
                    new_embeddings = [self._generate_basic_embedding(text) for text in uncached_texts]
                
                # Normalize if requested
                if normalize:
                    new_embeddings = [
                        emb / np.linalg.norm(emb) if np.linalg.norm(emb) > 0 else emb 
                        for emb in new_embeddings
                    ]
                
                # Insert new embeddings and cache them
                for idx, embedding in zip(uncached_indices, new_embeddings):
                    embeddings[idx] = embedding
                    cache_key = self._get_cache_key(texts[idx])
                    self.embedding_cache[cache_key] = embedding
                
            except Exception as e:
                logger.error(f"Failed to generate batch embeddings: {str(e)}")
                # Fill with zero vectors
                for idx in uncached_indices:
                    embeddings[idx] = np.zeros(self.dimension, dtype=np.float32)
        
        return embeddings
    
    async def _generate_sentence_transformer_embedding(self, text: str) -> np.ndarray:
        """Generate embedding using Sentence Transformers"""
        
        if not self.model:
            raise ValueError("Sentence Transformer model not initialized")
        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None, 
            lambda: self.model.encode([text], convert_to_numpy=True)[0]
        )
        
        return embedding.astype(np.float32)
    
    async def _generate_sentence_transformer_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings using Sentence Transformers (batch)"""
        
        if not self.model:
            raise ValueError("Sentence Transformer model not initialized")
        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: self.model.encode(texts, convert_to_numpy=True, batch_size=32, show_progress_bar=False)
        )
        
        return [emb.astype(np.float32) for emb in embeddings]
    
    async def _generate_openai_embedding(self, text: str) -> np.ndarray:
        """Generate embedding using OpenAI API"""
        
        try:
            response = await openai.Embedding.acreate(
                model=self.model_name,
                input=text
            )
            
            embedding = np.array(response['data'][0]['embedding'], dtype=np.float32)
            return embedding
            
        except Exception as e:
            logger.error(f"OpenAI embedding generation failed: {str(e)}")
            raise
    
    async def _generate_openai_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings using OpenAI API (batch)"""
        
        try:
            response = await openai.Embedding.acreate(
                model=self.model_name,
                input=texts
            )
            
            embeddings = [
                np.array(item['embedding'], dtype=np.float32) 
                for item in response['data']
            ]
            
            return embeddings
            
        except Exception as e:
            logger.error(f"OpenAI batch embedding generation failed: {str(e)}")
            raise
    
    def _generate_basic_embedding(self, text: str) -> np.ndarray:
        """Generate basic hash-based embedding (fallback)"""
        
        # Simple hash-based embedding for fallback
        import hashlib
        
        # Normalize text
        normalized_text = text.lower().strip()
        
        # Generate multiple hash values
        embedding = []
        
        for i in range(self.dimension // 32):  # 32 values per hash
            hash_input = f"{normalized_text}_{i}"
            hash_bytes = hashlib.md5(hash_input.encode()).digest()
            
            # Convert bytes to float values between -1 and 1
            hash_values = [
                ((byte - 128) / 128.0) 
                for byte in hash_bytes
            ]
            
            embedding.extend(hash_values[:32])
        
        # Pad if necessary
        while len(embedding) < self.dimension:
            embedding.append(0.0)
        
        return np.array(embedding[:self.dimension], dtype=np.float32)
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings"""
        
        try:
            # Ensure embeddings are normalized
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            normalized1 = embedding1 / norm1
            normalized2 = embedding2 / norm2
            
            similarity = np.dot(normalized1, normalized2)
            
            # Clamp to [-1, 1] to handle numerical precision issues
            similarity = max(-1.0, min(1.0, float(similarity)))
            
            return similarity
            
        except Exception as e:
            logger.error(f"Similarity calculation failed: {str(e)}")
            return 0.0
    
    def find_most_similar(
        self, 
        query_embedding: np.ndarray, 
        candidate_embeddings: List[np.ndarray],
        top_k: int = 5
    ) -> List[tuple]:
        """Find most similar embeddings to query"""
        
        similarities = []
        
        for i, candidate in enumerate(candidate_embeddings):
            similarity = self.calculate_similarity(query_embedding, candidate)
            similarities.append((i, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    async def semantic_search(
        self,
        query: str,
        documents: List[str],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Perform semantic search on documents"""
        
        # Generate embeddings
        query_embedding = await self.generate_embedding(query)
        document_embeddings = await self.generate_embeddings(documents)
        
        # Find most similar documents
        similar_indices = self.find_most_similar(query_embedding, document_embeddings, top_k)
        
        results = []
        for idx, similarity in similar_indices:
            results.append({
                "document_index": idx,
                "document": documents[idx],
                "similarity_score": similarity,
                "rank": len(results) + 1
            })
        
        return results
    
    def cluster_embeddings(
        self, 
        embeddings: List[np.ndarray], 
        n_clusters: int = 5
    ) -> List[int]:
        """Cluster embeddings using K-means"""
        
        try:
            from sklearn.cluster import KMeans
            
            # Stack embeddings
            embedding_matrix = np.stack(embeddings)
            
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embedding_matrix)
            
            return cluster_labels.tolist()
            
        except ImportError:
            logger.warning("scikit-learn not available, using basic clustering")
            return self._basic_clustering(embeddings, n_clusters)
        except Exception as e:
            logger.error(f"Clustering failed: {str(e)}")
            return [0] * len(embeddings)  # All in one cluster
    
    def _basic_clustering(self, embeddings: List[np.ndarray], n_clusters: int) -> List[int]:
        """Basic clustering without scikit-learn"""
        
        if not embeddings:
            return []
        
        # Simple clustering based on similarity threshold
        clusters = []
        cluster_centers = []
        
        for embedding in embeddings:
            best_cluster = -1
            best_similarity = -1.0
            
            # Find best matching cluster
            for i, center in enumerate(cluster_centers):
                similarity = self.calculate_similarity(embedding, center)
                if similarity > best_similarity and similarity > 0.7:  # Threshold
                    best_similarity = similarity
                    best_cluster = i
            
            if best_cluster == -1 and len(cluster_centers) < n_clusters:
                # Create new cluster
                cluster_centers.append(embedding)
                clusters.append(len(cluster_centers) - 1)
            elif best_cluster != -1:
                # Assign to existing cluster
                clusters.append(best_cluster)
            else:
                # Assign to first cluster
                clusters.append(0)
        
        return clusters
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        import hashlib
        return hashlib.md5(text.encode()).hexdigest()
    
    def _load_embedding_cache(self):
        """Load embedding cache from disk"""
        
        cache_file = os.path.join(self.cache_path, "embedding_cache.pkl")
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    self.embedding_cache = pickle.load(f)
                logger.info(f"Loaded {len(self.embedding_cache)} cached embeddings")
            except Exception as e:
                logger.warning(f"Failed to load embedding cache: {str(e)}")
                self.embedding_cache = {}
    
    def save_embedding_cache(self):
        """Save embedding cache to disk"""
        
        cache_file = os.path.join(self.cache_path, "embedding_cache.pkl")
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(self.embedding_cache, f)
            logger.info(f"Saved {len(self.embedding_cache)} embeddings to cache")
        except Exception as e:
            logger.error(f"Failed to save embedding cache: {str(e)}")
    
    def clear_cache(self):
        """Clear embedding cache"""
        self.embedding_cache = {}
        
        cache_file = os.path.join(self.cache_path, "embedding_cache.pkl")
        if os.path.exists(cache_file):
            os.remove(cache_file)
        
        logger.info("Embedding cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get embedding cache statistics"""
        
        return {
            "cached_embeddings": len(self.embedding_cache),
            "cache_size_mb": self._estimate_cache_size(),
            "provider": self.provider,
            "model": self.model_name,
            "dimension": self.dimension
        }
    
    def _estimate_cache_size(self) -> float:
        """Estimate cache size in MB"""
        
        if not self.embedding_cache:
            return 0.0
        
        # Rough estimation: dimension * 4 bytes (float32) per embedding
        bytes_per_embedding = self.dimension * 4
        total_bytes = len(self.embedding_cache) * bytes_per_embedding
        
        return round(total_bytes / (1024 * 1024), 2)
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of embedding service"""
        
        health_status = {
            "status": "healthy",
            "provider": self.provider,
            "model": self.model_name,
            "dimension": self.dimension,
            "cache_stats": self.get_cache_stats()
        }
        
        # Test embedding generation
        try:
            test_embedding = await self.generate_embedding("Health check test")
            
            if test_embedding is not None and len(test_embedding) == self.dimension:
                health_status["embedding_test"] = "passed"
            else:
                health_status["embedding_test"] = "failed"
                health_status["status"] = "unhealthy"
                
        except Exception as e:
            health_status["embedding_test"] = "failed"
            health_status["error"] = str(e)
            health_status["status"] = "unhealthy"
        
        return health_status
    
    def get_supported_models(self) -> List[str]:
        """Get list of supported models"""
        
        if self.provider == "sentence_transformers":
            return [
                "all-MiniLM-L6-v2",
                "all-MiniLM-L12-v2", 
                "all-mpnet-base-v2",
                "multi-qa-MiniLM-L6-cos-v1",
                "paraphrase-multilingual-MiniLM-L12-v2"
            ]
        elif self.provider == "openai":
            return [
                "text-embedding-ada-002",
                "text-similarity-davinci-001",
                "text-search-ada-doc-001"
            ]
        else:
            return ["basic-hash-embedding"]
    
    async def switch_model(self, new_model: str) -> bool:
        """Switch to a different embedding model"""
        
        try:
            old_model = self.model_name
            self.model_name = new_model
            
            # Clear cache since embeddings will be different
            self.clear_cache()
            
            # Re-initialize provider
            self._initialize_provider()
            
            # Test new model
            test_result = await self.health_check()
            
            if test_result["status"] == "healthy":
                logger.info(f"Successfully switched from {old_model} to {new_model}")
                return True
            else:
                # Rollback
                self.model_name = old_model
                self._initialize_provider()
                logger.error(f"Failed to switch to {new_model}, rolled back to {old_model}")
                return False
                
        except Exception as e:
            logger.error(f"Model switch failed: {str(e)}")
            return False
    
    def __del__(self):
        """Cleanup when service is destroyed"""
        # Disabled automatic cache saving on exit to prevent errors
        # Call save_embedding_cache() manually if needed
        pass