"""
GeoDatasetFinder SDK - Simple interface for genomic dataset discovery.
"""

import logging
from typing import Optional, Literal
import pandas as pd

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
        **kwargs: Additional arguments for VectorSearch or GPTFilter

    Returns:
        DataFrame with search results and similarity scores
    """
    if not query.strip():
        raise ValueError("Query cannot be empty")

    # Split kwargs for different components
    search_kwargs = {k: v for k, v in kwargs.items() if k in ['base_path']}
    gpt_kwargs = {
        "model": kwargs.get("model", "gpt-4"),
        "max_workers": kwargs.get("max_workers", 4),
        "temperature": kwargs.get("temperature", 0.3),
        "timeout": kwargs.get("timeout", 30),
        "api_key": api_key,
        "api_url": api_url,
    }

    logger.info(f"Searching for: '{query}' (dataset_type={dataset_type})")

    # Step 1: Semantic search
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
    **kwargs
) -> pd.DataFrame:
    """
    Search microarray datasets.
    """
    return search_datasets(query=query, dataset_type='microarray', api_key=api_key, api_url=api_url, **kwargs)


def search_rnaseq(
    query: str,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Search RNA-seq datasets.
    """
    return search_datasets(query=query, dataset_type='rnaseq', api_key=api_key, api_url=api_url, **kwargs)


def search_with_gpt(
    query: str,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Search with GPT filtering enabled.
    """
    return search_datasets(query=query, use_gpt_filter=True, api_key=api_key, api_url=api_url, **kwargs)