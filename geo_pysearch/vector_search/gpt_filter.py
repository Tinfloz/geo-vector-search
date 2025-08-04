"""
This module implements semantic query generation and evidence extraction strategies
inspired by the following works:

- Deka, P., Jurek-Loughrey, A., & others. (2022). "Evidence Extraction to Validate Medical Claims in Fake News Detection".
  International Conference on Health Information Science, pp. 3–15.

- Deka, P., & Jurek-Loughrey, A. (2021). "Unsupervised Keyword Combination Query Generation from Online Health Related Content for Evidence-Based Fact Checking".
  The 23rd International Conference on Information Integration and Web Intelligence, pp. 267–277.
"""

"""
GPT-based filtering module for genomic dataset curation.

This module provides intelligent filtering of genomic datasets using OpenAI's GPT models
to assess dataset suitability for differential expression analysis.
"""

import os
import requests
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Union, Optional, Literal
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
import time
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# Supported GPT models
GPTModel = Literal['gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo', 'gpt-3.5-turbo']


@dataclass
class GPTResponse:
    """Structured response from GPT API."""
    answer: Optional[str] = None
    reason: str = ""
    confidence: float = 0.0
    error: Optional[str] = None


class GPTFilterError(Exception):
    """Custom exception for GPTFilter operations."""
    pass


class GPTFilter:
    """
    Intelligent dataset filtering using GPT models for biomedical research.
    
    This class uses OpenAI's GPT models to assess whether genomic datasets
    are suitable for differential expression analysis for specific diseases.
    It processes datasets in parallel for efficiency.
    
    Attributes:
        model: GPT model to use for filtering
        max_workers: Maximum number of parallel API requests
        temperature: GPT temperature parameter (0.0-1.0)
        
    Example:
        >>> filter_engine = GPTFilter(model='gpt-4o', max_workers=4)
        >>> filtered_data = filter_engine.filter(datasets, disease='cancer')
        >>> print(f"Filtered to {len(filtered_data)} suitable datasets")
    """
    
    def __init__(
        self,
        model: GPTModel = 'gpt-4o',
        max_workers: int = 4,
        temperature: float = 0.3,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        timeout: int = 30
    ):
        """
        Initialize the GPTFilter instance.
        
        Args:
            model: GPT model to use for filtering
            max_workers: Maximum number of concurrent API requests (default: 4)
            temperature: GPT temperature parameter for response variability (default: 0.3)
            api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var
            api_url: OpenAI API URL. If None, reads from OPENAI_API_URL env var
            timeout: Request timeout in seconds (default: 30)
            
        Raises:
            GPTFilterError: If API credentials are not provided or invalid
            ValueError: If parameters are invalid
        """
        # Validate parameters
        if max_workers <= 0:
            raise ValueError("max_workers must be positive")
        if not 0.0 <= temperature <= 1.0:
            raise ValueError("temperature must be between 0.0 and 1.0")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        
        self.model = model
        self.max_workers = max_workers
        self.temperature = temperature
        self.timeout = timeout
        
        # Get API credentials
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self._api_key:
            raise GPTFilterError(
                "OpenAI API key not provided. Set OPENAI_API_KEY environment variable "
            )
        
        self._api_url = api_url or os.getenv("OPENAI_API_URL")
        if not self._api_url:
            raise GPTFilterError(
                "OpenAI API URL not provided. Set OPENAI_API_URL environment variable "
            )
        
        # Setup headers
        self._headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json"
        }
        
        # Rate limiting tracking
        self._request_times = []
        
        logger.info(f"Initialized GPTFilter with model={model}, max_workers={max_workers}")
    
    def _build_prompt(self, record: Dict, disease: str) -> str:
        """
        Build the prompt for GPT assessment.
        
        Args:
            record: Dataset metadata record
            disease: Target disease for analysis
            
        Returns:
            Formatted prompt string
        """
        return (
            f"You are a biomedical data curation assistant helping researchers identify "
            f"high-quality datasets for gene expression studies.\n\n"
            f"The researcher is studying **'{disease}'** and wants to perform "
            f"**Differential Expression (DE) analysis** — comparing gene expression "
            f"between disease and control samples.\n\n"
            f"You are given metadata for a GEO dataset. Your task is to assess whether "
            f"this dataset is **suitable** for DE analysis **for this specific disease**.\n\n"
            f"--- Dataset Metadata ---\n"
            f"GSE ID: {record.get('gse', 'N/A')}\n"
            f"Summary: {record.get('cleaned_text', 'N/A')}\n"
            f"------------------------\n\n"
            f"A dataset is suitable for DE analysis **only if**:\n"
            f"- It includes **both disease and control samples** (explicitly or clearly implied).\n"
            f"- The disease context matches or is closely related to **'{disease}'**.\n"
            f"- The design is appropriate for DE (e.g., not just time series, "
            f"pharmacological response, or single-condition studies).\n"
            f"- The sample size is not trivially small (ideally ≥5 per group).\n\n"
            f"If the dataset clearly meets the above criteria, answer 'Yes'. "
            f"If it clearly does not, answer 'No'. If uncertain due to lack of detail, answer 'No'.\n\n"
            f"Respond in **this exact format**:\n"
            f"Answer: <Yes/No>\n"
            f"Reason: <concise reason, ≤2 sentences>\n"
            f"Confidence: <a float between 0 and 1, where 1 is very confident>\n"
        )
    
    def _rate_limit_check(self) -> None:
        """
        Simple rate limiting to avoid overwhelming the API.
        Implements a basic sliding window rate limiter.
        """
        current_time = time.time()
        
        # Clean old requests (older than 60 seconds)
        self._request_times = [t for t in self._request_times if current_time - t < 60]
        
        # If we have too many recent requests, wait
        if len(self._request_times) >= 50:  # 50 requests per minute limit
            sleep_time = 60 - (current_time - self._request_times[0])
            if sleep_time > 0:
                logger.debug(f"Rate limiting: sleeping for {sleep_time:.1f} seconds")
                time.sleep(sleep_time)
        
        self._request_times.append(current_time)
    
    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type((requests.exceptions.RequestException, requests.exceptions.Timeout)),
        reraise=True
    )
    def _call_gpt(self, prompt: str) -> GPTResponse:
        """
        Make API call to GPT model with retry logic.
        
        Args:
            prompt: The prompt to send to GPT
            
        Returns:
            Structured GPT response
            
        Raises:
            GPTFilterError: If API call fails after retries
        """
        # Apply rate limiting
        self._rate_limit_check()
        
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are an expert biomedical research assistant."},
                {"role": "user", "content": prompt}
            ],
            "temperature": self.temperature,
            "max_tokens": 200  # Reasonable limit for our structured response
        }
        
        try:
            response = requests.post(
                self._api_url, 
                headers=self._headers, 
                json=body, 
                timeout=self.timeout
            )
            response.raise_for_status()
            
            response_data = response.json()
            
            # Check for API errors in response
            if 'error' in response_data:
                raise GPTFilterError(f"API error: {response_data['error']}")
            
            content = response_data["choices"][0]["message"]["content"]
            return self._parse_response(content)
            
        except requests.exceptions.Timeout:
            logger.warning(f"Request timeout after {self.timeout} seconds")
            raise
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request failed: {e}")
            raise
        except KeyError as e:
            raise GPTFilterError(f"Unexpected API response format: missing {e}")
        except Exception as e:
            raise GPTFilterError(f"Unexpected error in API call: {e}")
    
    def _parse_response(self, response: str) -> GPTResponse:
        """
        Parse GPT response into structured format.
        
        Args:
            response: Raw GPT response text
            
        Returns:
            Structured GPT response object
        """
        lines = response.strip().splitlines()
        result = GPTResponse()
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.lower().startswith("answer:"):
                answer = line.split(":", 1)[1].strip()
                # Normalize answer
                if answer.lower() in ['yes', 'y', 'true', '1']:
                    result.answer = 'Yes'
                elif answer.lower() in ['no', 'n', 'false', '0']:
                    result.answer = 'No'
                else:
                    result.answer = answer  # Keep original if unclear
                    
            elif line.lower().startswith("reason:"):
                result.reason = line.split(":", 1)[1].strip()
                
            elif line.lower().startswith("confidence:"):
                try:
                    confidence_str = line.split(":", 1)[1].strip()
                    # Remove any trailing text after the number
                    confidence_str = confidence_str.split()[0]
                    result.confidence = float(confidence_str)
                    # Clamp to valid range
                    result.confidence = max(0.0, min(1.0, result.confidence))
                except (ValueError, IndexError):
                    logger.warning(f"Could not parse confidence from: {line}")
                    result.confidence = 0.0
        
        # Validation
        if result.answer is None:
            logger.warning("GPT response missing 'Answer' field")
            result.answer = 'No'  # Default to No if unclear
            result.confidence = 0.0
        
        return result
    
    def _process_single(self, record: Dict, disease: str) -> Dict:
        """
        Process a single dataset record.
        
        Args:
            record: Dataset metadata record
            disease: Target disease
            
        Returns:
            Record with GPT assessment added
        """
        try:
            prompt = self._build_prompt(record, disease)
            gpt_result = self._call_gpt(prompt)
            
            # Create result record
            final_record = record.copy()
            final_record.update({
                'gpt_answer': gpt_result.answer,
                'gpt_reason': gpt_result.reason,
                'gpt_confidence': gpt_result.confidence,
                'gpt_error': None
            })
            
        except Exception as e:
            logger.error(f"Error processing record {record.get('gse', 'unknown')}: {e}")
            # Create error record
            final_record = record.copy()
            final_record.update({
                'gpt_answer': 'Error',
                'gpt_reason': f"Processing failed: {str(e)[:100]}...",
                'gpt_confidence': 0.0,
                'gpt_error': str(e)
            })
        
        return final_record
    
    def filter(
        self,
        data: Union[pd.DataFrame, List[Dict]],
        disease: str,
        confidence_threshold: float = 0.5,
        return_all: bool = False
    ) -> pd.DataFrame:
        """
        Filter datasets using GPT assessment for differential expression suitability.
        
        Args:
            data: Input datasets as DataFrame or list of dictionaries
            disease: Target disease for differential expression analysis
            confidence_threshold: Minimum confidence score to include dataset (0.0-1.0)
            return_all: If True, return all records with GPT assessments.
                       If False, return only suitable datasets above threshold.
            
        Returns:
            Filtered DataFrame with GPT assessment columns added:
            - gpt_answer: 'Yes'/'No'/'Error'
            - gpt_reason: Explanation for the decision
            - gpt_confidence: Confidence score (0.0-1.0)
            - gpt_error: Error message if processing failed
            
        Raises:
            ValueError: If inputs are invalid
            GPTFilterError: If filtering process fails
            
        Example:
            >>> filtered = gpt_filter.filter(
            ...     datasets, 
            ...     disease='breast cancer',
            ...     confidence_threshold=0.7
            ... )
        """
        # Input validation
        if not disease or not disease.strip():
            raise ValueError("Disease parameter cannot be empty")
        
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0.0 and 1.0")
        
        # Convert input to DataFrame
        if isinstance(data, list):
            if not data:
                return pd.DataFrame()
            data = pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            if data.empty:
                return pd.DataFrame()
        else:
            raise ValueError("data must be a pandas DataFrame or list of dictionaries")
        
        logger.info(f"Starting GPT filtering for '{disease}' on {len(data)} datasets")
        logger.info(f"Using {self.max_workers} workers, confidence threshold: {confidence_threshold}")
        
        # Process datasets in parallel
        results = []
        start_time = time.time()
        
        try:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_record = {
                    executor.submit(self._process_single, row.to_dict(), disease): idx
                    for idx, (_, row) in enumerate(data.iterrows())
                }
                
                # Collect results with progress bar
                for future in tqdm(
                    as_completed(future_to_record), 
                    total=len(future_to_record),
                    desc="Processing datasets"
                ):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        # This shouldn't happen due to error handling in _process_single
                        logger.error(f"Unexpected error in future result: {e}")
                        # Create minimal error record
                        idx = future_to_record[future]
                        error_record = data.iloc[idx].to_dict()
                        error_record.update({
                            'gpt_answer': 'Error',
                            'gpt_reason': f"Unexpected error: {e}",
                            'gpt_confidence': 0.0,
                            'gpt_error': str(e)
                        })
                        results.append(error_record)
        
        except Exception as e:
            raise GPTFilterError(f"Parallel processing failed: {e}")
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Log statistics
        processing_time = time.time() - start_time
        total_count = len(results_df)
        yes_count = len(results_df[results_df['gpt_answer'] == 'Yes'])
        error_count = len(results_df[results_df['gpt_answer'] == 'Error'])
        
        logger.info(f"GPT filtering completed in {processing_time:.1f}s")
        logger.info(f"Results: {yes_count}/{total_count} suitable datasets, {error_count} errors")
        
        # Apply filtering
        if return_all:
            return results_df.reset_index(drop=True)
        else:
            # Filter for suitable datasets above confidence threshold
            suitable_mask = (
                (results_df['gpt_answer'] == 'Yes') & 
                (results_df['gpt_confidence'] >= confidence_threshold)
            )
            filtered_df = results_df[suitable_mask].reset_index(drop=True)
            
            logger.info(f"Filtered to {len(filtered_df)} datasets above confidence threshold {confidence_threshold}")
            return filtered_df
    
    def get_stats(self) -> Dict:
        """
        Get configuration statistics.
        
        Returns:
            Dictionary with current configuration
        """
        return {
            'model': self.model,
            'max_workers': self.max_workers,
            'temperature': self.temperature,
            'timeout': self.timeout,
            'api_url': self._api_url,
            'recent_requests': len(self._request_times)
        }
    
    def __repr__(self) -> str:
        """String representation of the GPTFilter instance."""
        return (f"GPTFilter(model='{self.model}', max_workers={self.max_workers}, "
                f"temperature={self.temperature})")