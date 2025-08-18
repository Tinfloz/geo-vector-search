from typing import Optional, Literal, Dict, List, Union
from pathlib import Path
import pandas as pd
import logging
from geo_pysearch.vector_search.vector_search import VectorSearch
from geo_pysearch.vector_search.gpt_filter import GPTFilter
from geo_pysearch.vector_search.tiered_gpt_filter import EnhancedGPTFilter

logger = logging.getLogger(__name__)

DatasetType = Literal['microarray', 'rnaseq']
GPTFilterType = Literal['basic', 'enhanced']


def search_datasets(
    query: str,
    dataset_type: DatasetType = 'microarray',
    top_k: int = 50,
    use_gpt_filter: bool = False,
    gpt_filter_type: GPTFilterType = 'enhanced',
    confidence_threshold: float = 0.6,
    tier_filter: Optional[List[int]] = None,
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
        gpt_filter_type: Type of GPT filter to use ('basic' or 'enhanced')
        confidence_threshold: GPT confidence threshold (0.0-1.0)
        tier_filter: List of acceptable tiers for enhanced filter (default: [1] for Tier 1 only)
        return_all_gpt_results: If True, return all GPT responses
        api_key: OpenAI API key (optional)
        api_url: OpenAI API base URL (optional)
        cache_dir: Directory for caching downloaded files (optional)
        force_download: If True, re-download files even if cached
        **kwargs: Additional arguments for VectorSearch or GPTFilter

    Returns:
        DataFrame with search results and similarity scores.
        
        Enhanced GPT filter adds these columns:
        - gpt_tier: Tier classification (1/2/3)
        - gpt_disease_samples, gpt_control_samples: Sample presence
        - gpt_tissue_type: Tissue/model type
        - gpt_anatomical_relevance: Anatomical relevance level
        - gpt_study_design: Study design type
        - gpt_reason: Detailed explanation
        - gpt_key_limitations: Key limitations identified
        - gpt_confidence: Confidence score (0.0-1.0)
        - gpt_raw_response: Full GPT response
        - gpt_error: Error message if processing failed

    Note:
        Files are automatically downloaded from Hugging Face Hub and cached locally.
        First run may take longer due to file downloads.
        
    Examples:
        >>> # Basic search without filtering
        >>> results = search_datasets("breast cancer", top_k=20)
        
        >>> # Enhanced GPT filtering for Tier 1 datasets only
        >>> results = search_datasets(
        ...     "alzheimer disease", 
        ...     use_gpt_filter=True,
        ...     tier_filter=[1],
        ...     confidence_threshold=0.8
        ... )
        
        >>> # Get Tier 1 and 2 datasets for broader analysis
        >>> results = search_datasets(
        ...     "diabetes", 
        ...     use_gpt_filter=True,
        ...     tier_filter=[1, 2],
        ...     confidence_threshold=0.6
        ... )
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
        "model": kwargs.get("model", "gpt-4o"),
        "max_workers": kwargs.get("max_workers", 4),
        "temperature": kwargs.get("temperature", 0.3),
        "timeout": kwargs.get("timeout", 45 if gpt_filter_type == 'enhanced' else 30),
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
        logger.info(f"Applying {gpt_filter_type} GPT filtering for query: '{query}'")
        
        if gpt_filter_type == 'enhanced':
            gpt_filter = EnhancedGPTFilter(**gpt_kwargs)
            results = gpt_filter.filter(
                data=results,
                disease=query,
                tier_filter=tier_filter,
                confidence_threshold=confidence_threshold,
                return_all=return_all_gpt_results
            )
        else:  # basic filter
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
    Search with GPT filtering enabled (uses enhanced filter by default).
    
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
        gpt_filter_type='enhanced',
        api_key=api_key, 
        api_url=api_url,
        cache_dir=cache_dir,
        force_download=force_download,
        **kwargs
    )


def search_with_enhanced_gpt(
    query: str,
    tier_filter: Optional[List[int]] = None,
    confidence_threshold: float = 0.6,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
    cache_dir: Optional[Path] = None,
    force_download: bool = False,
    **kwargs
) -> pd.DataFrame:
    """
    Search with enhanced GPT filtering for differential expression analysis.
    
    This function uses the enhanced 3-tier evaluation system specifically designed
    for differential expression analysis suitability assessment.
    
    Args:
        query: Natural language query/disease
        tier_filter: List of acceptable tiers (default: [1] for Tier 1 only)
                    - Tier 1: Directly suitable (high confidence)
                    - Tier 2: Conditionally suitable (moderate utility)
                    - Tier 3: Not suitable (exclude)
        confidence_threshold: Minimum confidence score (0.0-1.0)
        api_key: OpenAI API key (optional)
        api_url: OpenAI API base URL (optional)
        cache_dir: Directory for caching downloaded files (optional)
        force_download: If True, re-download files even if cached
        **kwargs: Additional search parameters
    
    Returns:
        DataFrame with enhanced GPT assessment columns
    
    Examples:
        >>> # Get only Tier 1 datasets with high confidence
        >>> results = search_with_enhanced_gpt(
        ...     "breast cancer",
        ...     tier_filter=[1],
        ...     confidence_threshold=0.8
        ... )
        
        >>> # Get Tier 1 and 2 datasets for broader analysis
        >>> results = search_with_enhanced_gpt(
        ...     "alzheimer disease",
        ...     tier_filter=[1, 2],
        ...     confidence_threshold=0.6
        ... )
    """
    return search_datasets(
        query=query, 
        use_gpt_filter=True,
        gpt_filter_type='enhanced',
        tier_filter=tier_filter,
        confidence_threshold=confidence_threshold,
        api_key=api_key, 
        api_url=api_url,
        cache_dir=cache_dir,
        force_download=force_download,
        **kwargs
    )


def analyze_gpt_results(results_df: pd.DataFrame) -> Dict:
    """
    Analyze GPT filtering results and provide summary statistics.
    
    Args:
        results_df: DataFrame with GPT filtering results
        
    Returns:
        Dictionary with detailed analysis
        
    Example:
        >>> results = search_with_enhanced_gpt("cancer", return_all_gpt_results=True)
        >>> stats = analyze_gpt_results(results)
        >>> print(f"Tier 1 datasets: {stats['tier_distribution']['tier_1']}")
    """
    if results_df.empty:
        return {"message": "No data to analyze"}
    
    # Check if this is enhanced GPT results
    if 'gpt_tier' in results_df.columns:
        analysis = {
            'total_datasets': len(results_df),
            'tier_distribution': {
                'tier_1': len(results_df[results_df['gpt_tier'] == 1]),
                'tier_2': len(results_df[results_df['gpt_tier'] == 2]),
                'tier_3': len(results_df[results_df['gpt_tier'] == 3])
            },
            'avg_confidence': results_df['gpt_confidence'].mean(),
            'confidence_distribution': {
                'high_confidence_0.8+': len(results_df[results_df['gpt_confidence'] >= 0.8]),
                'medium_confidence_0.6-0.8': len(results_df[
                    (results_df['gpt_confidence'] >= 0.6) & (results_df['gpt_confidence'] < 0.8)
                ]),
                'low_confidence_<0.6': len(results_df[results_df['gpt_confidence'] < 0.6])
            }
        }
        
        # Add tissue type and study design analysis if available
        if 'gpt_tissue_type' in results_df.columns:
            analysis['tissue_types'] = results_df['gpt_tissue_type'].value_counts().to_dict()
        
        if 'gpt_study_design' in results_df.columns:
            analysis['study_designs'] = results_df['gpt_study_design'].value_counts().to_dict()
        
        if 'gpt_error' in results_df.columns:
            analysis['error_rate'] = len(results_df[results_df['gpt_error'].notna()]) / len(results_df)
    
    else:
        # Basic GPT results
        analysis = {
            'total_datasets': len(results_df),
            'avg_confidence': results_df.get('gpt_confidence', pd.Series([])).mean(),
            'message': "Basic GPT filtering results - use enhanced filtering for detailed analysis"
        }
    
    return analysis


def print_gpt_summary(results_df: pd.DataFrame) -> None:
    """
    Print a formatted summary of GPT filtering results.
    
    Args:
        results_df: DataFrame with GPT filtering results
        
    Example:
        >>> results = search_with_enhanced_gpt("diabetes")
        >>> print_gpt_summary(results)
    """
    stats = analyze_gpt_results(results_df)
    
    if "message" in stats and "No data" in stats["message"]:
        print("No data to summarize")
        return
    
    print("\n" + "="*50)
    print("GPT FILTERING SUMMARY")
    print("="*50)
    
    print(f"Total Datasets: {stats['total_datasets']}")
    
    if 'tier_distribution' in stats:
        print(f"\nTier Distribution:")
        print(f"  Tier 1 (Directly Suitable): {stats['tier_distribution']['tier_1']}")
        print(f"  Tier 2 (Conditionally Suitable): {stats['tier_distribution']['tier_2']}")
        print(f"  Tier 3 (Not Suitable): {stats['tier_distribution']['tier_3']}")
        
        print(f"\nAverage Confidence: {stats['avg_confidence']:.3f}")
        
        if 'confidence_distribution' in stats:
            print(f"\nConfidence Distribution:")
            print(f"  High (≥0.8): {stats['confidence_distribution']['high_confidence_0.8+']}")
            print(f"  Medium (0.6-0.8): {stats['confidence_distribution']['medium_confidence_0.6-0.8']}")
            print(f"  Low (<0.6): {stats['confidence_distribution']['low_confidence_<0.6']}")
        
        if 'tissue_types' in stats and stats['tissue_types']:
            print(f"\nTop Tissue Types:")
            for tissue, count in list(stats['tissue_types'].items())[:5]:
                print(f"  {tissue}: {count}")
        
        if 'study_designs' in stats and stats['study_designs']:
            print(f"\nStudy Designs:")
            for design, count in list(stats['study_designs'].items())[:5]:
                print(f"  {design}: {count}")
        
        if 'error_rate' in stats:
            print(f"\nError Rate: {stats['error_rate']:.1%}")
    
    else:
        print(f"Average Confidence: {stats.get('avg_confidence', 'N/A')}")
        if 'message' in stats:
            print(f"Note: {stats['message']}")
    
    print("="*50)


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