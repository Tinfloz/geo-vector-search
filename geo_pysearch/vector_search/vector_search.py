"""
Vector search module for genomic dataset discovery.

This module provides semantic search capabilities over genomic datasets using
pre-trained BioBERT models and FAISS for efficient similarity search.
"""

import faiss
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
from typing import List, Dict, Union, Optional, Literal
import logging
from contextlib import contextmanager

# Configure logging
logger = logging.getLogger(__name__)

# Dataset configurations
DATASET_CONFIGS = {
    'microarray': {
        'model': "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
        'vector_path': "geo_pysearch/data/vector_embeddings.npz",
        'index_path': "geo_pysearch/data/vector_index.faiss",
        'metadata_path': "geo_pysearch/data/vector_metadata.csv",
        'extra_metadata_path': "geo_pysearch/data/vector_corpus.cleaned.csv"
    },
    'rnaseq': {
        'model': "pritamdeka/S-BioBert-snli-multinli-stsb",
        'vector_path': "geo_pysearch/data/rnaseq_vector_embeddings.npz",
        'index_path': "geo_pysearch/data/rnaseq_vector_index.faiss.faiss",
        'metadata_path': "geo_pysearch/data/rnaseq_vector_metadata.csv",
        'extra_metadata_path': "geo_pysearch/data/rnaseq_vector_corpus.cleaned.csv"
    }
}

DatasetType = Literal['microarray', 'rnaseq']


class VectorSearchError(Exception):
    """Custom exception for VectorSearch operations."""
    pass


class VectorSearch:
    """
    Semantic search engine for genomic datasets using vector embeddings.
    
    This class provides efficient similarity search over genomic dataset metadata
    using pre-trained BioBERT models and FAISS indexing. It supports both
    microarray and RNA-seq datasets with specialized models for each type.
    
    Attributes:
        dataset_type: Type of genomic dataset ('microarray' or 'rnaseq')
        return_dataframe: Whether to return results as DataFrame or list of dicts
        base_path: Base directory containing data files
        
    Example:
        >>> searcher = VectorSearch(dataset_type='microarray')
        >>> results = searcher.search("cancer gene expression", top_k=10)
        >>> print(f"Found {len(results)} relevant datasets")
    """
    
    def __init__(
        self, 
        dataset_type: DatasetType = 'microarray',
        return_dataframe: bool = True,
        base_path: Optional[Path] = None
    ):
        """
        Initialize the VectorSearch instance.
        
        Args:
            dataset_type: Type of dataset to search ('microarray' or 'rnaseq')
            return_dataframe: If True, return results as pandas DataFrame,
                            otherwise as list of dictionaries
            base_path: Base directory containing data files. If None, uses
                      current working directory
                      
        Raises:
            ValueError: If dataset_type is not supported
            VectorSearchError: If required data files are not found
        """
        if dataset_type not in DATASET_CONFIGS:
            raise ValueError(f"Unsupported dataset type: {dataset_type}. "
                           f"Must be one of {list(DATASET_CONFIGS.keys())}")
        
        self.dataset_type = dataset_type
        self.return_dataframe = return_dataframe
        self.base_path = Path(base_path) if base_path else Path.cwd()
        
        # Get configuration for the specified dataset type
        self._config = DATASET_CONFIGS[dataset_type]
        
        # Construct file paths
        self._model_name = self._config['model']
        self._vector_path = self.base_path / self._config['vector_path']
        self._faiss_index_path = self.base_path / self._config['index_path']
        self._metadata_path = self.base_path / self._config['metadata_path']
        self._extra_metadata_path = self.base_path / self._config['extra_metadata_path']
        
        # Lazy-loaded components
        self._model: Optional[SentenceTransformer] = None
        self._faiss_index: Optional[faiss.Index] = None
        self._metadata: Optional[pd.DataFrame] = None
        self._extra_metadata: Optional[pd.DataFrame] = None

        # Validate that required files exist
        self._validate_data_files()
        
        logger.info(f"Initialized VectorSearch for {dataset_type} datasets")
    
    def _validate_data_files(self) -> None:
        """
        Validate that all required data files exist.
        
        Raises:
            VectorSearchError: If any required files are missing
        """
        required_files = [
            (self._faiss_index_path, "FAISS index"),
            (self._metadata_path, "metadata CSV"),
            (self._extra_metadata_path, "extra metadata CSV")
        ]
        
        missing_files = []
        for file_path, description in required_files:
            if not file_path.exists():
                missing_files.append(f"{description} at {file_path}")
        
        if missing_files:
            raise VectorSearchError(
                f"Missing required files: {', '.join(missing_files)}"
            )
    
    def _load_model(self) -> None:
        """
        Load the sentence transformer model for encoding queries.
        
        Raises:
            VectorSearchError: If model loading fails
        """
        if self._model is None:
            try:
                logger.debug(f"Loading model: {self._model_name}")
                self._model = SentenceTransformer(self._model_name)
                logger.debug("Model loaded successfully")
            except Exception as e:
                raise VectorSearchError(f"Failed to load model {self._model_name}: {e}")
    
    def _load_faiss_index(self) -> None:
        """
        Load the FAISS index for similarity search.
        
        Raises:
            VectorSearchError: If index loading fails
        """
        if self._faiss_index is None:
            try:
                logger.debug(f"Loading FAISS index from {self._faiss_index_path}")
                self._faiss_index = faiss.read_index(str(self._faiss_index_path))
                logger.debug(f"FAISS index loaded with {self._faiss_index.ntotal} vectors")
            except Exception as e:
                raise VectorSearchError(f"Failed to load FAISS index: {e}")
    
    def _load_metadata(self) -> None:
        """
        Load the metadata CSV file.
        
        Raises:
            VectorSearchError: If metadata loading fails
        """
        if self._metadata is None:
            try:
                logger.debug(f"Loading metadata from {self._metadata_path}")
                self._metadata = pd.read_csv(self._metadata_path)
                logger.debug(f"Metadata loaded with {len(self._metadata)} records")
            except Exception as e:
                raise VectorSearchError(f"Failed to load metadata: {e}")
            
    def _load_extra_metadata(self) -> None:
        """
        Load the extra metadata CSV file.
        
        Raises:
            VectorSearchError: If extra metadata loading fails
        """
        if self._extra_metadata is None:
            try:
                logger.debug(f"Loading extra metadata from {self._extra_metadata_path}")
                self._extra_metadata = pd.read_csv(self._extra_metadata_path)
                logger.debug(f"Extra metadata loaded with {len(self._extra_metadata)} records")
            except Exception as e:
                raise VectorSearchError(f"Failed to load metadata: {e}")
    
    def _ensure_components_loaded(self) -> None:
        """Ensure all required components are loaded."""
        self._load_model()
        self._load_faiss_index()
        self._load_metadata()
        self._load_extra_metadata()
    
    def _encode_query(self, query: str) -> np.ndarray:
        """
        Encode a text query into a vector representation.
        
        Args:
            query: Text query to encode
            
        Returns:
            Normalized vector embedding of the query
            
        Raises:
            VectorSearchError: If encoding fails
        """
        try:
            # Encode query and normalize embeddings for cosine similarity
            embedding = self._model.encode([query], normalize_embeddings=True)
            return embedding.astype("float32")
        except Exception as e:
            raise VectorSearchError(f"Failed to encode query '{query}': {e}")
    
    def _validate_search_params(self, query: str, top_k: int) -> None:
        """
        Validate search parameters.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Raises:
            ValueError: If parameters are invalid
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty or whitespace only")
        
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        
        if top_k > 10000:  # Reasonable upper limit
            raise ValueError("top_k cannot exceed 10,000")
    
    def search(
        self, 
        query: str, 
        top_k: int = 50
    ) -> Union[pd.DataFrame, List[Dict]]:
        """
        Search for datasets similar to the given query.
        
        Args:
            query: Natural language description of desired datasets
            top_k: Maximum number of results to return (default: 50)
            
        Returns:
            Search results as DataFrame or list of dictionaries (based on 
            return_dataframe setting). Results include all metadata columns
            plus a 'similarity' column with cosine similarity scores.
            
        Raises:
            ValueError: If query is empty or top_k is invalid
            VectorSearchError: If search operation fails
            
        Example:
            >>> results = searcher.search("breast cancer gene expression", top_k=5)
            >>> print(results[['title', 'similarity']].head())
        """
        # Validate inputs
        self._validate_search_params(query, top_k)
        
        # Ensure all components are loaded
        self._ensure_components_loaded()
        
        try:
            # Encode the query
            query_vector = self._encode_query(f"{query} control vs disease expression")
            
            # Perform similarity search
            logger.debug(f"Searching for top {top_k} results for query: '{query[:50]}...'")
            similarity_scores, indices = self._faiss_index.search(query_vector, top_k)
            
            # Extract results from metadata
            result_indices = indices[0]  # Get indices from first (and only) query
            result_scores = similarity_scores[0]  # Get scores from first query
            
            # Filter out invalid indices (FAISS returns -1 for insufficient results)
            valid_mask = result_indices >= 0
            result_indices = result_indices[valid_mask]
            result_scores = result_scores[valid_mask]
            
            if len(result_indices) == 0:
                logger.warning("No valid results found for the query")
                empty_result = pd.DataFrame() if self.return_dataframe else []
                return empty_result
            
            # Get metadata for matching datasets
            results_df = self._metadata.iloc[result_indices].copy()
            results_df['similarity'] = result_scores
            if self._extra_metadata is not None:
                results_df = results_df.merge(self._extra_metadata[["gse", "cleaned_text"]], on="gse", how="left")
            
            # Reset index to avoid confusion
            results_df = results_df.reset_index(drop=True)
            
            logger.info(f"Found {len(results_df)} results for query")
            
            # Return in requested format
            if self.return_dataframe:
                return results_df
            else:
                return results_df.to_dict(orient="records")
                
        except Exception as e:
            raise VectorSearchError(f"Search operation failed: {e}")
    
    def batch_search(
        self, 
        queries: List[str], 
        top_k: int = 50
    ) -> List[Union[pd.DataFrame, List[Dict]]]:
        """
        Perform batch search for multiple queries efficiently.
        
        Args:
            queries: List of query strings
            top_k: Maximum number of results per query
            
        Returns:
            List of search results, one per query
            
        Raises:
            ValueError: If queries list is empty or contains invalid queries
            VectorSearchError: If batch search fails
        """
        if not queries:
            raise ValueError("Queries list cannot be empty")
        
        # Validate all queries
        for i, query in enumerate(queries):
            try:
                self._validate_search_params(query, top_k)
            except ValueError as e:
                raise ValueError(f"Invalid query at index {i}: {e}")
        
        # Ensure components are loaded
        self._ensure_components_loaded()
        
        logger.info(f"Performing batch search for {len(queries)} queries")
        
        try:
            # Encode all queries at once for efficiency
            query_vectors = []
            for query in queries:
                query_vector = self._encode_query(query)
                query_vectors.append(query_vector[0])  # Remove batch dimension
            
            query_matrix = np.vstack(query_vectors).astype("float32")
            
            # Perform batch similarity search
            similarity_scores, indices = self._faiss_index.search(query_matrix, top_k)
            
            # Process results for each query
            results = []
            for i, (query_indices, query_scores) in enumerate(zip(indices, similarity_scores)):
                # Filter valid indices
                valid_mask = query_indices >= 0
                valid_indices = query_indices[valid_mask]
                valid_scores = query_scores[valid_mask]
                
                if len(valid_indices) == 0:
                    empty_result = pd.DataFrame() if self.return_dataframe else []
                    results.append(empty_result)
                    continue
                
                # Get metadata and add similarity scores
                query_results = self._metadata.iloc[valid_indices].copy()
                query_results['similarity'] = valid_scores
                if self._extra_metadata is not None:
                    query_results = query_results.merge(self._extra_metadata[["gse", "cleaned_text"]], on="gse", how="left")
                query_results = query_results.reset_index(drop=True)
                
                # Convert to requested format
                if self.return_dataframe:
                    results.append(query_results)
                else:
                    results.append(query_results.to_dict(orient="records"))
            
            logger.info("Batch search completed successfully")
            return results
            
        except Exception as e:
            raise VectorSearchError(f"Batch search failed: {e}")
    
    def get_stats(self) -> Dict:
        """
        Get statistics about the loaded dataset.
        
        Returns:
            Dictionary containing dataset statistics
        """
        self._ensure_components_loaded()
        
        return {
            'dataset_type': self.dataset_type,
            'model_name': self._model_name,
            'total_datasets': len(self._metadata),
            'vector_dimension': self._faiss_index.d,
            'index_type': type(self._faiss_index).__name__,
            'metadata_columns': list(self._metadata.columns)
        }
    
    @contextmanager
    def _temp_return_format(self, return_dataframe: bool):
        """Temporarily change the return format."""
        original = self.return_dataframe
        self.return_dataframe = return_dataframe
        try:
            yield
        finally:
            self.return_dataframe = original
    
    def close(self) -> None:
        """
        Clean up loaded resources to free memory.
        
        Note: After calling this method, the search functionality will still work
        but components will need to be reloaded on the next search operation.
        """
        self._model = None
        self._faiss_index = None
        self._metadata = None
        logger.info("VectorSearch resources cleaned up")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with automatic cleanup."""
        self.close()
    
    def __repr__(self) -> str:
        """String representation of the VectorSearch instance."""
        return (f"VectorSearch(dataset_type='{self.dataset_type}', "
                f"return_dataframe={self.return_dataframe})")