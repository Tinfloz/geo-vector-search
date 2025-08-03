from typing import Optional, Literal, Dict
from pathlib import Path
import pandas as pd
import logging
from geo_pysearch.vector_search.vector_search import VectorSearch
from geo_pysearch.vector_search.gpt_filter import GPTFilter

logger = logging.getLogger(__name__)

DatasetType = Literal['microarray', 'rnaseq']


def search_datasets(
    query: str,
    dataset_type: DatasetType = 'microarray',
    top_k: int = 50,
    use_gpt_filter: bool = False,
    confidence_threshold: float = 0.6,
    return_all_gpt_results: bool = False,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
    cache_dir: Optional[Path] = None,
    force_download: bool = False,
    **kwargs
) -> pd.DataFrame:
    """
    Search for genomic datasets with optional GPT filtering.

    Args:
        query: Natural language query/disease (e.g. "breast cancer gene expression")
        dataset_type: 'microarray' or 'rnaseq'
        top_k: Number of top results to retrieve
        use_gpt_filter: Whether to apply GPT filtering for differential expression suitability
        confidence_threshold: GPT confidence threshold (0.0-1.0)
        return_all_gpt_results: If True, return all GPT responses
        api_key: OpenAI API key (optional)
        api_url: OpenAI API base URL (optional)
        cache_dir: Directory for caching downloaded files (optional)
        force_download: If True, re-download files even if cached
        **kwargs: Additional arguments for VectorSearch or GPTFilter

    Returns:
        DataFrame with search results and similarity scores

    Note:
        Files are automatically downloaded from Hugging Face Hub and cached locally.
        First run may take longer due to file downloads.
    """
    if not query.strip():
        raise ValueError("Query cannot be empty")

    # Split kwargs for different components
    search_kwargs = {
        k: v for k, v in kwargs.items() 
        if k in ['return_dataframe']  # Remove base_path as it's no longer used
    }
    search_kwargs.update({
        'cache_dir': cache_dir,
        'force_download': force_download
    })
    
    gpt_kwargs = {
        "model": kwargs.get("model", "gpt-4"),
        "max_workers": kwargs.get("max_workers", 4),
        "temperature": kwargs.get("temperature", 0.3),
        "timeout": kwargs.get("timeout", 30),
        "api_key": api_key,
        "api_url": api_url,
    }

    logger.info(f"Searching for: '{query}' (dataset_type={dataset_type})")

    # Step 1: Semantic search (files will be downloaded and cached automatically)
    search_engine = VectorSearch(dataset_type=dataset_type, **search_kwargs)
    results = search_engine.search(query=query, top_k=top_k)

    if results.empty:
        logger.warning("No results found from semantic search")
        return results

    logger.info(f"Found {len(results)} results from semantic search")

    # Step 2: Optional GPT filtering
    if use_gpt_filter:
        logger.info(f"Applying GPT filtering for query: '{query}'")
        gpt_filter = GPTFilter(**gpt_kwargs)
        results = gpt_filter.filter(
            data=results,
            disease=query,
            confidence_threshold=confidence_threshold,
            return_all=return_all_gpt_results
        )
        logger.info(f"GPT filtering completed: {len(results)} results")

    return results


def search_microarray(
    query: str,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
    cache_dir: Optional[Path] = None,
    force_download: bool = False,
    **kwargs
) -> pd.DataFrame:
    """
    Search microarray datasets.
    
    Args:
        query: Natural language query/disease
        api_key: OpenAI API key (optional)
        api_url: OpenAI API base URL (optional)
        cache_dir: Directory for caching downloaded files (optional)
        force_download: If True, re-download files even if cached
        **kwargs: Additional search parameters
    
    Returns:
        DataFrame with search results
    """
    return search_datasets(
        query=query, 
        dataset_type='microarray', 
        api_key=api_key, 
        api_url=api_url,
        cache_dir=cache_dir,
        force_download=force_download,
        **kwargs
    )


def search_rnaseq(
    query: str,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
    cache_dir: Optional[Path] = None,
    force_download: bool = False,
    **kwargs
) -> pd.DataFrame:
    """
    Search RNA-seq datasets.
    
    Args:
        query: Natural language query/disease
        api_key: OpenAI API key (optional)
        api_url: OpenAI API base URL (optional)
        cache_dir: Directory for caching downloaded files (optional)
        force_download: If True, re-download files even if cached
        **kwargs: Additional search parameters
    
    Returns:
        DataFrame with search results
    """
    return search_datasets(
        query=query, 
        dataset_type='rnaseq', 
        api_key=api_key, 
        api_url=api_url,
        cache_dir=cache_dir,
        force_download=force_download,
        **kwargs
    )


def search_with_gpt(
    query: str,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
    cache_dir: Optional[Path] = None,
    force_download: bool = False,
    **kwargs
) -> pd.DataFrame:
    """
    Search with GPT filtering enabled.
    
    Args:
        query: Natural language query/disease
        api_key: OpenAI API key (optional)
        api_url: OpenAI API base URL (optional)
        cache_dir: Directory for caching downloaded files (optional)
        force_download: If True, re-download files even if cached
        **kwargs: Additional search parameters
    
    Returns:
        DataFrame with filtered search results
    """
    return search_datasets(
        query=query, 
        use_gpt_filter=True, 
        api_key=api_key, 
        api_url=api_url,
        cache_dir=cache_dir,
        force_download=force_download,
        **kwargs
    )


# New utility functions for cache management
def get_cache_info(cache_dir: Optional[Path] = None) -> Dict:
    """
    Get information about cached files.
    
    Args:
        cache_dir: Cache directory to inspect (optional)
    
    Returns:
        Dictionary with cache information including size and file details
    """
    from geo_pysearch.vector_search.vector_search import get_cache_info as _get_cache_info
    return _get_cache_info(cache_dir)


def clear_cache(cache_dir: Optional[Path] = None, dataset_type: Optional[DatasetType] = None) -> None:
    """
    Clear cached files.
    
    Args:
        cache_dir: Cache directory to clear (optional)
        dataset_type: If specified, only clear cache for this dataset type
    
    Example:
        >>> clear_cache()  # Clear all cached files
        >>> clear_cache(dataset_type='microarray')  # Clear only microarray cache
    """
    if dataset_type is not None:
        # Clear cache for specific dataset type
        search_engine = VectorSearch(dataset_type=dataset_type, cache_dir=cache_dir)
        search_engine.clear_cache()
        logger.info(f"Cleared cache for {dataset_type} dataset")
    else:
        # Clear all cache
        from geo_pysearch.vector_search.vector_search import clear_all_cache
        clear_all_cache(cache_dir)
        logger.info("Cleared all cached files")


def print_cache_info(cache_dir: Optional[Path] = None) -> None:
    """
    Print formatted cache information to console.
    
    Args:
        cache_dir: Cache directory to inspect (optional)
    """
    from geo_pysearch.vector_search.vector_search import print_cache_info as _print_cache_info
    _print_cache_info(cache_dir)


def preload_datasets(
    dataset_types: Optional[list] = None,
    cache_dir: Optional[Path] = None,
    force_download: bool = False
) -> None:
    """
    Pre-download and cache datasets to avoid delays during first search.
    
    Args:
        dataset_types: List of dataset types to preload. If None, preloads all types.
        cache_dir: Directory for caching files (optional)
        force_download: If True, re-download files even if cached
    
    Example:
        >>> preload_datasets()  # Preload all dataset types
        >>> preload_datasets(['microarray'])  # Preload only microarray
    """
    if dataset_types is None:
        dataset_types = ['microarray', 'rnaseq']
    
    for dataset_type in dataset_types:
        logger.info(f"Preloading {dataset_type} dataset...")
        try:
            search_engine = VectorSearch(
                dataset_type=dataset_type,
                cache_dir=cache_dir,
                force_download=force_download
            )
            # Just initialize to trigger file downloads
            search_engine._ensure_components_loaded()
            search_engine.close()  # Free memory
            logger.info(f"Successfully preloaded {dataset_type} dataset")
        except Exception as e:
            logger.error(f"Failed to preload {dataset_type} dataset: {e}")
            raise