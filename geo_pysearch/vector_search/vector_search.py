"""
Vector search module for genomic dataset discovery with Hugging Face Hub integration.

This module provides semantic search capabilities over genomic datasets using
pre-trained BioBERT models and FAISS for efficient similarity search.
Files are automatically downloaded from Hugging Face Hub and cached locally.
"""

import faiss
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
from typing import List, Dict, Union, Optional, Literal
import logging
from contextlib import contextmanager
import hashlib
import os
import requests
from urllib.parse import urljoin
import tempfile
import shutil
from datetime import datetime, timedelta
import json

# Configure logging
logger = logging.getLogger(__name__)

# Dataset configurations with Hugging Face Hub paths
DATASET_CONFIGS = {
    'microarray': {
        'model': "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
        'hf_repo': "Tinfloz/geo_pysearch-data",
        'files': {
            #'vector_embeddings': "vector_embeddings.npz",
            'faiss_index': "vector_index.faiss",
            'metadata': "vector_metadata.csv",
            'extra_metadata': "vector_corpus.cleaned.csv"
        }
    },
    'rnaseq': {
        'model': "pritamdeka/S-BioBert-snli-multinli-stsb",
        'hf_repo': "Tinfloz/geo_pysearch-data",  
        'files': {
            #'vector_embeddings': "rnaseq_vector_embeddings.npz",
            'faiss_index': "rnaseq_vector_index.faiss",
            'metadata': "rnaseq_vector_metadata.csv",
            'extra_metadata': "rnaseq_vector_corpus.cleaned.csv"
        }
    }
}

DatasetType = Literal['microarray', 'rnaseq']


class VectorSearchError(Exception):
    """Custom exception for VectorSearch operations."""
    pass


class CacheManager:
    """
    Manages local caching of downloaded files from Hugging Face Hub.
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize cache manager.
        
        Args:
            cache_dir: Directory for caching files. If None, uses system temp dir.
        """
        if cache_dir is None:
            # Use system cache directory or temp directory
            if os.name == 'nt':  # Windows
                base_cache = Path(os.environ.get('LOCALAPPDATA', tempfile.gettempdir()))
            else:  # Unix-like
                base_cache = Path(os.environ.get('XDG_CACHE_HOME', 
                                               Path.home() / '.cache'))
            self.cache_dir = base_cache / 'genomic_vector_search'
        else:
            self.cache_dir = Path(cache_dir)
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.cache_dir / 'cache_metadata.json'
        logger.debug(f"Cache directory: {self.cache_dir}")
    
    def _get_cache_metadata(self) -> Dict:
        """Load cache metadata from disk."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load cache metadata: {e}")
        return {}
    
    def _save_cache_metadata(self, metadata: Dict) -> None:
        """Save cache metadata to disk."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        except IOError as e:
            logger.warning(f"Failed to save cache metadata: {e}")
    
    def _get_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of a file."""
        hash_sha256 = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except IOError:
            return ""
    
    def get_cached_file_path(self, repo_id: str, filename: str) -> Path:
        """Get the local cache path for a file."""
        # Create a safe filename from repo_id and filename
        safe_repo = repo_id.replace('/', '_')
        return self.cache_dir / safe_repo / filename
    
    def is_file_cached(self, repo_id: str, filename: str, 
                      max_age_days: int = 30) -> bool:
        """
        Check if a file is cached and not too old.
        
        Args:
            repo_id: Hugging Face repository ID
            filename: Name of the file
            max_age_days: Maximum age of cached file in days
            
        Returns:
            True if file is cached and fresh
        """
        cache_path = self.get_cached_file_path(repo_id, filename)
        if not cache_path.exists():
            return False
        
        # Check age
        metadata = self._get_cache_metadata()
        file_key = f"{repo_id}/{filename}"
        
        if file_key in metadata:
            cached_time = datetime.fromisoformat(metadata[file_key]['cached_at'])
            if datetime.now() - cached_time > timedelta(days=max_age_days):
                logger.debug(f"Cached file {file_key} is too old")
                return False
        
        return True
    
    def cache_file(self, repo_id: str, filename: str, source_path: Path) -> Path:
        """
        Cache a file and update metadata.
        
        Args:
            repo_id: Hugging Face repository ID
            filename: Name of the file
            source_path: Path to the source file to cache
            
        Returns:
            Path to the cached file
        """
        cache_path = self.get_cached_file_path(repo_id, filename)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy file to cache
        shutil.copy2(source_path, cache_path)
        
        # Update metadata
        metadata = self._get_cache_metadata()
        file_key = f"{repo_id}/{filename}"
        metadata[file_key] = {
            'cached_at': datetime.now().isoformat(),
            'file_hash': self._get_file_hash(cache_path),
            'file_size': cache_path.stat().st_size
        }
        self._save_cache_metadata(metadata)
        
        logger.debug(f"Cached file: {file_key}")
        return cache_path
    
    def clear_cache(self, repo_id: Optional[str] = None) -> None:
        """
        Clear cache files.
        
        Args:
            repo_id: If specified, only clear files for this repository.
                    If None, clear all cached files.
        """
        if repo_id is None:
            # Clear entire cache
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Cleared entire cache")
        else:
            # Clear specific repository cache
            repo_cache_dir = self.cache_dir / repo_id.replace('/', '_')
            if repo_cache_dir.exists():
                shutil.rmtree(repo_cache_dir)
            
            # Update metadata
            metadata = self._get_cache_metadata()
            keys_to_remove = [k for k in metadata.keys() if k.startswith(f"{repo_id}/")]
            for key in keys_to_remove:
                del metadata[key]
            self._save_cache_metadata(metadata)
            
            logger.info(f"Cleared cache for repository: {repo_id}")
    
    def get_cache_info(self) -> Dict:
        """Get information about cached files."""
        metadata = self._get_cache_metadata()
        total_size = 0
        file_count = 0
        
        for file_info in metadata.values():
            total_size += file_info.get('file_size', 0)
            file_count += 1
        
        return {
            'cache_dir': str(self.cache_dir),
            'total_files': file_count,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'files': metadata
        }


class HuggingFaceDownloader:
    """
    Downloads files from Hugging Face Hub with caching support.
    """
    
    def __init__(self, cache_manager: CacheManager):
        """
        Initialize downloader.
        
        Args:
            cache_manager: Cache manager instance
        """
        self.cache_manager = cache_manager
        self.hf_base_url = "https://huggingface.co"
    
    def _get_download_url(self, repo_id: str, filename: str, 
                         revision: str = "main") -> str:
        """Get the download URL for a file from Hugging Face Hub."""
        return f"{self.hf_base_url}/datasets/{repo_id}/resolve/{revision}/{filename}"
    
    def _download_file(self, url: str, target_path: Path, 
                      chunk_size: int = 8192) -> None:
        """
        Download a file from URL to target path with progress logging.
        
        Args:
            url: URL to download from
            target_path: Local path to save the file
            chunk_size: Size of chunks to download at a time
        """
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Downloading {url}")
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Get file size from headers if available
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(target_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Log progress every 10MB
                        if total_size > 0 and downloaded % (10 * 1024 * 1024) == 0:
                            progress = (downloaded / total_size) * 100
                            logger.debug(f"Download progress: {progress:.1f}%")
            
            logger.info(f"Successfully downloaded: {target_path.name}")
            
        except requests.RequestException as e:
            if target_path.exists():
                target_path.unlink()  # Clean up partial download
            raise VectorSearchError(f"Failed to download {url}: {e}")
    
    def get_file(self, repo_id: str, filename: str, 
                revision: str = "main", force_download: bool = False) -> Path:
        """
        Get a file from Hugging Face Hub, using cache if available.
        
        Args:
            repo_id: Hugging Face repository ID
            filename: Name of the file to download
            revision: Git revision (branch/tag/commit)
            force_download: If True, download even if cached
            
        Returns:
            Path to the local file (cached or downloaded)
        """
        # Check cache first
        if not force_download and self.cache_manager.is_file_cached(repo_id, filename):
            cache_path = self.cache_manager.get_cached_file_path(repo_id, filename)
            logger.debug(f"Using cached file: {filename}")
            return cache_path
        
        # Download to temporary location first
        download_url = self._get_download_url(repo_id, filename, revision)
        
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
        
        try:
            self._download_file(download_url, tmp_path)
            
            # Move to cache
            cached_path = self.cache_manager.cache_file(repo_id, filename, tmp_path)
            return cached_path
            
        finally:
            # Clean up temp file
            if tmp_path.exists():
                tmp_path.unlink()


class VectorSearch:
    """
    Semantic search engine for genomic datasets using vector embeddings.
    
    This class provides efficient similarity search over genomic dataset metadata
    using pre-trained BioBERT models and FAISS indexing. It supports both
    microarray and RNA-seq datasets with specialized models for each type.
    
    Files are automatically downloaded from Hugging Face Hub and cached locally.
    
    Attributes:
        dataset_type: Type of genomic dataset ('microarray' or 'rnaseq')
        return_dataframe: Whether to return results as DataFrame or list of dicts
        cache_dir: Directory for caching downloaded files
        
    Example:
        >>> searcher = VectorSearch(dataset_type='microarray')
        >>> results = searcher.search("cancer gene expression", top_k=10)
        >>> print(f"Found {len(results)} relevant datasets")
    """
    
    def __init__(
        self, 
        dataset_type: DatasetType = 'microarray',
        return_dataframe: bool = True,
        cache_dir: Optional[Path] = None,
        force_download: bool = False
    ):
        """
        Initialize the VectorSearch instance.
        
        Args:
            dataset_type: Type of dataset to search ('microarray' or 'rnaseq')
            return_dataframe: If True, return results as pandas DataFrame,
                            otherwise as list of dictionaries
            cache_dir: Directory for caching files. If None, uses system cache dir
            force_download: If True, re-download files even if cached
                      
        Raises:
            ValueError: If dataset_type is not supported
            VectorSearchError: If required data files cannot be obtained
        """
        if dataset_type not in DATASET_CONFIGS:
            raise ValueError(f"Unsupported dataset type: {dataset_type}. "
                           f"Must be one of {list(DATASET_CONFIGS.keys())}")
        
        self.dataset_type = dataset_type
        self.return_dataframe = return_dataframe
        self.force_download = force_download
        
        # Initialize cache manager and downloader
        self.cache_manager = CacheManager(cache_dir)
        self.downloader = HuggingFaceDownloader(self.cache_manager)
        
        # Get configuration for the specified dataset type
        self._config = DATASET_CONFIGS[dataset_type]
        
        # Model and repository info
        self._model_name = self._config['model']
        self._repo_id = self._config['hf_repo']
        self._files = self._config['files']
        
        # Lazy-loaded components
        self._model: Optional[SentenceTransformer] = None
        self._faiss_index: Optional[faiss.Index] = None
        self._metadata: Optional[pd.DataFrame] = None
        self._extra_metadata: Optional[pd.DataFrame] = None
        
        # File paths (will be set when files are downloaded)
        self._file_paths: Dict[str, Path] = {}
        
        logger.info(f"Initialized VectorSearch for {dataset_type} datasets")
        logger.info(f"Cache directory: {self.cache_manager.cache_dir}")
    
    def _ensure_files_available(self) -> None:
        """
        Ensure all required files are available locally (download if needed).
        
        Raises:
            VectorSearchError: If files cannot be downloaded
        """
        if self._file_paths:  # Already downloaded
            return
        
        logger.info("Ensuring required files are available...")
        
        try:
            for file_key, filename in self._files.items():
                logger.debug(f"Getting file: {filename}")
                file_path = self.downloader.get_file(
                    self._repo_id, 
                    filename, 
                    force_download=self.force_download
                )
                self._file_paths[file_key] = file_path
                
            logger.info("All required files are available")
            
        except Exception as e:
            raise VectorSearchError(f"Failed to obtain required files: {e}")
    
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
                faiss_path = self._file_paths['faiss_index']
                logger.debug(f"Loading FAISS index from {faiss_path}")
                self._faiss_index = faiss.read_index(str(faiss_path))
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
                metadata_path = self._file_paths['metadata']
                logger.debug(f"Loading metadata from {metadata_path}")
                self._metadata = pd.read_csv(metadata_path)
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
                extra_metadata_path = self._file_paths['extra_metadata']
                logger.debug(f"Loading extra metadata from {extra_metadata_path}")
                self._extra_metadata = pd.read_csv(extra_metadata_path)
                logger.debug(f"Extra metadata loaded with {len(self._extra_metadata)} records")
            except Exception as e:
                raise VectorSearchError(f"Failed to load extra metadata: {e}")
    
    def _ensure_components_loaded(self) -> None:
        """Ensure all required components are loaded."""
        self._ensure_files_available()
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
            >>> print(results[['gse', 'similarity']].head())
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
                results_df = results_df.merge(
                    self._extra_metadata[["gse", "cleaned_text"]], 
                    on="gse", 
                    how="left"
                )
            
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
                    query_results = query_results.merge(
                        self._extra_metadata[["gse", "cleaned_text"]], 
                        on="gse", 
                        how="left"
                    )
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
            'repository': self._repo_id,
            'total_datasets': len(self._metadata),
            'vector_dimension': self._faiss_index.d,
            'index_type': type(self._faiss_index).__name__,
            'metadata_columns': list(self._metadata.columns),
            'cache_info': self.cache_manager.get_cache_info()
        }
    
    def clear_cache(self) -> None:
        """
        Clear cached files for this dataset type.
        This will force re-download on next use.
        """
        self.cache_manager.clear_cache(self._repo_id)
        self._file_paths = {}  # Reset file paths
        logger.info(f"Cleared cache for {self.dataset_type} dataset")
    
    def get_cache_info(self) -> Dict:
        """Get information about cached files."""
        return self.cache_manager.get_cache_info()
    
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
        Cached files are preserved.
        """
        self._model = None
        self._faiss_index = None
        self._metadata = None
        self._extra_metadata = None
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
                f"return_dataframe={self.return_dataframe}, "
                f"cache_dir='{self.cache_manager.cache_dir}')")


# Utility functions for CLI/SDK
def clear_all_cache(cache_dir: Optional[Path] = None) -> None:
    """
    Clear all cached files.
    
    Args:
        cache_dir: Cache directory to clear. If None, uses default cache location.
    """
    cache_manager = CacheManager(cache_dir)
    cache_manager.clear_cache()
    print(f"Cleared all cached files from {cache_manager.cache_dir}")


def get_cache_info(cache_dir: Optional[Path] = None) -> Dict:
    """
    Get information about cached files.
    
    Args:
        cache_dir: Cache directory to inspect. If None, uses default cache location.
        
    Returns:
        Dictionary with cache information
    """
    cache_manager = CacheManager(cache_dir)
    return cache_manager.get_cache_info()


def print_cache_info(cache_dir: Optional[Path] = None) -> None:
    """
    Print formatted cache information.
    
    Args:
        cache_dir: Cache directory to inspect. If None, uses default cache location.
    """
    info = get_cache_info(cache_dir)
    print(f"Cache Directory: {info['cache_dir']}")
    print(f"Total Files: {info['total_files']}")
    print(f"Total Size: {info['total_size_mb']} MB")
    print("\nCached Files:")
    for file_path, file_info in info['files'].items():
        cached_time = datetime.fromisoformat(file_info['cached_at'])
        size_mb = round(file_info['file_size'] / (1024 * 1024), 2)
        print(f"  {file_path}: {size_mb} MB (cached: {cached_time.strftime('%Y-%m-%d %H:%M')})")