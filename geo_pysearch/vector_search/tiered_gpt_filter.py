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
to assess dataset suitability for differential expression analysis using an enhanced
robust evaluation framework.
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
import re

load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# Supported GPT models
GPTModel = Literal['gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo', 'gpt-3.5-turbo']


@dataclass
class EnhancedGPTResponse:
    """Structured response from enhanced GPT API."""
    tier: Optional[int] = None
    disease_samples: Optional[str] = None
    control_samples: Optional[str] = None
    tissue_type: Optional[str] = None
    anatomical_relevance: Optional[str] = None
    study_design: Optional[str] = None
    reason: str = ""
    key_limitations: str = ""
    confidence: float = 0.0
    error: Optional[str] = None
    raw_response: str = ""


class GPTFilterError(Exception):
    """Custom exception for GPTFilter operations."""
    pass


class EnhancedGPTFilter:
    """
    Intelligent dataset filtering using GPT models with enhanced robust evaluation.
    
    This class uses OpenAI's GPT models to assess genomic datasets using a comprehensive
    tiered evaluation framework for differential expression analysis suitability.
    
    Attributes:
        model: GPT model to use for filtering
        max_workers: Maximum number of parallel API requests
        temperature: GPT temperature parameter (0.0-1.0)
        
    Example:
        >>> filter_engine = EnhancedGPTFilter(model='gpt-4o', max_workers=4)
        >>> filtered_data = filter_engine.filter(datasets, disease='cancer')
        >>> print(f"Found {len(filtered_data)} Tier 1 datasets")
    """
    
    def __init__(
        self,
        model: GPTModel = 'gpt-4o',
        max_workers: int = 4,
        temperature: float = 0.3,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        timeout: int = 45  # Increased for more complex prompt
    ):
        """
        Initialize the EnhancedGPTFilter instance.
        
        Args:
            model: GPT model to use for filtering
            max_workers: Maximum number of concurrent API requests (default: 4)
            temperature: GPT temperature parameter for response variability (default: 0.3)
            api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var
            api_url: OpenAI API URL. If None, reads from OPENAI_API_URL env var
            timeout: Request timeout in seconds (default: 45)
            
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
                "OpenAI API key not provided. Set OPENAI_API_KEY environment variable"
            )
        
        self._api_url = api_url or os.getenv("OPENAI_API_URL")
        if not self._api_url:
            raise GPTFilterError(
                "OpenAI API URL not provided. Set OPENAI_API_URL environment variable"
            )
        
        # Setup headers
        self._headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json"
        }
        
        # Rate limiting tracking
        self._request_times = []
        
        logger.info(f"Initialized EnhancedGPTFilter with model={model}, max_workers={max_workers}")
    
    def _build_enhanced_prompt(self, record: Dict, disease: str) -> str:
        """
        Build the prompt for GPT assessment.
        
        Args:
            record: Dataset metadata record
            disease: Target disease for analysis
            
        Returns:
            Enhanced formatted prompt string
        """
        return f"""# Enhanced Biomedical Data Curation Assistant

## Core Task
You are a biomedical data curation specialist. Evaluate GEO datasets for their suitability in **Differential Expression (DE) analysis** comparing **{disease}** samples against appropriate controls.

## Dataset Information
- **GSE ID**: {record.get('gse', 'N/A')}
- **Summary**: {record.get('cleaned_text', 'N/A')}
- **Target Disease**: {disease}

## Evaluation Framework

### TIER 1: Directly Suitable (High Confidence)
**ALL criteria must be met:**

**Sample Composition (REQUIRED):**
- Contains disease samples AND matched controls (healthy/normal)
- Minimum 3 samples per group (prefer ≥5)
- Clear case-control design for comparative analysis

**Tissue Relevance (REQUIRED):**
- Uses primary human tissue directly relevant to disease pathophysiology
- Appropriate anatomical site (e.g., brain for neurological diseases)
- Fresh/frozen tissue preferred over FFPE when possible

**Study Design (REQUIRED):**
- Designed for disease comparison (not primarily drug response/intervention)
- Appropriate controls (not just different disease states)
- Sufficient metadata for proper analysis

**Note:** Datasets with only miRNA or non-coding RNA **cannot be Tier 1**, regardless of sample size, tissue relevance, or study design.

### TIER 2: Conditionally Suitable (Moderate Utility)
**Must meet basic comparison criteria PLUS one or more limitations:**

**Acceptable with Limitations:**
- Disease-relevant cell lines or established organoid models
- Smaller sample sizes (2-4 per group) but adequate for exploratory analysis
- Mixed sample types but includes relevant tissue
- Induced disease models (iPSC-derived, xenografts) in appropriate context
- Subset analysis possible from larger study
- Well-characterized animal models standard for this disease field

**Study Design Considerations:**
- Primary focus on intervention but includes baseline comparisons
- Time-series with appropriate control timepoints
- Multi-condition studies where disease vs control can be extracted

**Note:** Datasets with only miRNA or non-coding RNA **cannot be Tier 2**, regardless of sample size, tissue relevance, or study design.

### TIER 3: Not Suitable (Exclude)
**Any of these conditions automatically disqualifies:**

**Fatal Flaws:**
- No control samples or inappropriate controls only
- Single condition studies (no comparative element)
- Wrong tissue/organ system for disease biology
- Only treatment response without baseline disease comparison
- Technical replicates only (no biological replicates)
- Only cell culture without disease relevance
- Purely methodological studies (platform comparisons, etc.)
- miRNA / non-coding RNA datasets (cannot directly score genes; gene-level DE analysis not possible; can be flagged for regulatory or biomarker studies)

**Insufficient Context:**
- Unclear sample composition from metadata
- No disease-control distinction possible
- Unrelated biological context

## Decision Process

### Step 0: miRNA / non-coding RNA check
- If the dataset primarily measures miRNA or non-coding RNA, set:
  Tier = 3
  Is miRNA = TRUE
  Reason = "Dataset contains primarily miRNA/non-coding RNA; gene-level DE analysis not possible."
  Confidence < 1.0
- If miRNA = FALSE, proceed to Step 1.

### Step 1: Sample Composition Analysis
Identify if dataset contains:
- Disease samples: YES/NO
- Control samples: YES/NO  
- Sample sizes: Disease=X, Control=Y

### Step 2: Tissue/Model Relevance Assessment
- Primary human tissue: YES/NO
- Disease-relevant tissue: YES/NO
- Model system type: [primary/cell_line/organoid/animal/other]

### Step 3: Study Design Evaluation
- Designed for disease comparison: YES/NO
- Appropriate for DE analysis: YES/NO
- Major confounding factors: [list or none]

### Step 4: Quality Indicators
Count positive factors:
- Adequate sample size (≥5 per group): +1
- Primary human tissue: +2  
- Disease-relevant anatomical site: +2
- Clear case-control design: +2
- Good metadata quality: +1

## Output Requirements

**MANDATORY FORMAT - Do not deviate:**

```
Tier: [1/2/3]
Disease Samples: [YES/NO] (count if available)
Control Samples: [YES/NO] (count if available)  
Tissue Type: [primary_human/cell_line/organoid/animal_model/unclear]
Anatomical Relevance: [HIGH/MODERATE/LOW/IRRELEVANT]
Study Design: [case_control/intervention/time_series/other]
Reason: [2-3 sentences explaining classification with specific evidence]
Is miRNA: [TRUE/FALSE]
Key Limitations: [list main concerns or "none"]
Confidence: [0.0-1.0]
```

## Quality Control Checks

**Before finalizing, verify:**
1. Classification aligns with tier definitions
2. Evidence supports reasoning
3. All required output fields completed
4. Confidence score reflects certainty level
5. Key limitations identified if present

## Edge Case Handling

**If metadata is unclear or insufficient:**
- Default to Tier 3 with confidence <0.6
- Note "insufficient metadata" in limitations
- Specify what information is missing

**If multiple interpretations possible:**
- Choose more conservative tier
- Lower confidence score accordingly
- Explain uncertainty in reasoning

**Cross-disease relevance:**
- Focus on specified disease context
- Consider tissue-disease biology relationship
- Account for disease heterogeneity when relevant

## Confidence Scoring Guide
- **0.9-1.0**: Clear, unambiguous classification with strong evidence
- **0.7-0.8**: Good confidence with minor uncertainties
- **0.5-0.6**: Moderate confidence, some ambiguity in metadata
- **0.3-0.4**: Low confidence, significant uncertainty
- **0.0-0.2**: Very uncertain, insufficient information"""
    
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
    def _call_gpt(self, prompt: str) -> EnhancedGPTResponse:
        """
        Make API call to GPT model with retry logic.
        
        Args:
            prompt: The enhanced prompt to send to GPT
            
        Returns:
            Structured enhanced GPT response
            
        Raises:
            GPTFilterError: If API call fails after retries
        """
        # Apply rate limiting
        self._rate_limit_check()
        
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are an expert biomedical research data curation specialist with extensive knowledge of genomics, differential expression analysis, and dataset quality assessment."},
                {"role": "user", "content": prompt}
            ],
            "temperature": self.temperature,
            "max_tokens": 500  # Increased for more detailed structured response
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
            return self._parse_enhanced_response(content)
            
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
    
    def _parse_enhanced_response(self, response: str) -> EnhancedGPTResponse:
        """
        Parse enhanced GPT response into structured format.
        
        Args:
            response: Raw GPT response text
            
        Returns:
            Structured enhanced GPT response object
        """
        lines = response.strip().splitlines()
        result = EnhancedGPTResponse(raw_response=response)
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('```'):
                continue
                
            # Parse each field with more robust regex
            if self._extract_field(line, "tier:", result, 'tier', int):
                continue
            elif self._extract_field(line, "disease samples:", result, 'disease_samples', str):
                continue
            elif self._extract_field(line, "control samples:", result, 'control_samples', str):
                continue
            elif self._extract_field(line, "tissue type:", result, 'tissue_type', str):
                continue
            elif self._extract_field(line, "anatomical relevance:", result, 'anatomical_relevance', str):
                continue
            elif self._extract_field(line, "study design:", result, 'study_design', str):
                continue
            elif self._extract_field(line, "reason:", result, 'reason', str):
                continue
            elif self._extract_field(line, "is mirna", result, 'is_mirna', str):
                continue
            elif self._extract_field(line, "key limitations:", result, 'key_limitations', str):
                continue
            elif self._extract_field(line, "confidence:", result, 'confidence', float):
                continue
        
        # Validation and defaults
        self._validate_response(result)
        
        return result
    
    def _extract_field(self, line: str, field_name: str, result: EnhancedGPTResponse, 
                      attr_name: str, field_type: type) -> bool:
        """
        Extract and set a field from a response line.
        
        Args:
            line: Line to parse
            field_name: Field identifier to look for
            result: Result object to update
            attr_name: Attribute name to set
            field_type: Type to convert to
            
        Returns:
            True if field was found and parsed, False otherwise
        """
        if line.lower().startswith(field_name.lower()):
            try:
                value_str = line.split(":", 1)[1].strip()
                
                if field_type == int:
                    # Extract tier number
                    tier_match = re.search(r'\d+', value_str)
                    if tier_match:
                        value = int(tier_match.group())
                        # Validate tier range
                        if 1 <= value <= 3:
                            setattr(result, attr_name, value)
                        else:
                            logger.warning(f"Invalid tier value: {value}, defaulting to 3")
                            setattr(result, attr_name, 3)
                    else:
                        setattr(result, attr_name, 3)  # Default to Tier 3
                        
                elif field_type == float:
                    # Extract confidence score
                    confidence_match = re.search(r'(\d+\.?\d*)', value_str)
                    if confidence_match:
                        value = float(confidence_match.group())
                        # Clamp to valid range
                        value = max(0.0, min(1.0, value))
                        setattr(result, attr_name, value)
                    else:
                        setattr(result, attr_name, 0.0)
                        
                elif field_type == str:
                    # Clean and set string value
                    clean_value = re.sub(r'[^\w\s\-/(),.]', '', value_str).strip()
                    setattr(result, attr_name, clean_value or "unclear")
                    
                return True
                
            except (ValueError, IndexError) as e:
                logger.warning(f"Could not parse {field_name} from: {line}, error: {e}")
                # Set safe defaults
                if field_type == int:
                    setattr(result, attr_name, 3)
                elif field_type == float:
                    setattr(result, attr_name, 0.0)
                else:
                    setattr(result, attr_name, "unclear")
                return True
        
        return False
    
    def _validate_response(self, result: EnhancedGPTResponse) -> None:
        """
        Validate and set defaults for missing or invalid response fields.
        
        Args:
            result: Response object to validate
        """
        # Set defaults for missing fields
        if result.tier is None:
            logger.warning("GPT response missing 'Tier' field, defaulting to 3")
            result.tier = 3
            result.confidence = max(0.0, result.confidence - 0.2)  # Lower confidence
        
        if not result.reason:
            result.reason = "Insufficient analysis provided"
            result.confidence = max(0.0, result.confidence - 0.3)
        
        # Validate tier range
        if result.tier not in [1, 2, 3]:
            logger.warning(f"Invalid tier {result.tier}, defaulting to 3")
            result.tier = 3
            result.confidence = max(0.0, result.confidence - 0.2)
        
        # Ensure confidence is in valid range
        result.confidence = max(0.0, min(1.0, result.confidence))
        
        # Set default limitations if empty
        if not result.key_limitations:
            if result.tier == 3:
                result.key_limitations = "not suitable for DE analysis"
            elif result.tier == 2:
                result.key_limitations = "has moderate limitations"
            else:
                result.key_limitations = "none"
    
    def _process_single(self, record: Dict, disease: str) -> Dict:
        """
        Process a single dataset record with enhanced evaluation.
        
        Args:
            record: Dataset metadata record
            disease: Target disease
            
        Returns:
            Record with enhanced GPT assessment added
        """
        try:
            prompt = self._build_enhanced_prompt(record, disease)
            gpt_result = self._call_gpt(prompt)
            
            # Create enhanced result record
            final_record = record.copy()
            final_record.update({
                'gpt_tier': gpt_result.tier,
                'gpt_disease_samples': gpt_result.disease_samples,
                'gpt_control_samples': gpt_result.control_samples,
                'gpt_tissue_type': gpt_result.tissue_type,
                'gpt_anatomical_relevance': gpt_result.anatomical_relevance,
                'gpt_study_design': gpt_result.study_design,
                'gpt_reason': gpt_result.reason,
                'gpt_is_mirna': gpt_result.is_mirna,
                'gpt_key_limitations': gpt_result.key_limitations,
                'gpt_confidence': gpt_result.confidence,
                'gpt_raw_response': gpt_result.raw_response,
                'gpt_error': None
            })
            
        except Exception as e:
            logger.error(f"Error processing record {record.get('gse', 'unknown')}: {e}")
            # Create error record
            final_record = record.copy()
            final_record.update({
                'gpt_tier': 3,
                'gpt_disease_samples': 'Error',
                'gpt_control_samples': 'Error',
                'gpt_tissue_type': 'Error',
                'gpt_anatomical_relevance': 'Error',
                'gpt_study_design': 'Error',
                'gpt_reason': f"Processing failed: {str(e)[:100]}...",
                'gpt_is_mirna': 'Error',
                'gpt_key_limitations': f"Processing error: {str(e)[:50]}...",
                'gpt_confidence': 0.0,
                'gpt_raw_response': '',
                'gpt_error': str(e)
            })
        
        return final_record
    
    def filter(
        self,
        data: Union[pd.DataFrame, List[Dict]],
        disease: str,
        tier_filter: Optional[List[int]] = None,
        confidence_threshold: float = 0.5,
        return_all: bool = False
    ) -> pd.DataFrame:
        """
        Filter datasets using enhanced GPT assessment for differential expression suitability.
        
        Args:
            data: Input datasets as DataFrame or list of dictionaries
            disease: Target disease for differential expression analysis
            tier_filter: List of acceptable tiers (default: [1] for Tier 1 only)
            confidence_threshold: Minimum confidence score to include dataset (0.0-1.0)
            return_all: If True, return all records with GPT assessments.
                       If False, return only datasets matching tier and confidence filters.
            
        Returns:
            Filtered DataFrame with enhanced GPT assessment columns added:
            - gpt_tier: Tier classification (1/2/3)
            - gpt_disease_samples, gpt_control_samples: Sample presence
            - gpt_tissue_type: Tissue/model type
            - gpt_anatomical_relevance: Anatomical relevance level
            - gpt_study_design: Study design type
            - gpt_reason: Detailed explanation
            - gpt_is_mirna: If the dataset is an miRNA dataset or not
            - gpt_key_limitations: Key limitations identified
            - gpt_confidence: Confidence score (0.0-1.0)
            - gpt_raw_response: Full GPT response
            - gpt_error: Error message if processing failed
            
        Raises:
            ValueError: If inputs are invalid
            GPTFilterError: If filtering process fails
            
        Example:
            >>> # Get only Tier 1 datasets with high confidence
            >>> tier1_datasets = filter_engine.filter(
            ...     datasets, 
            ...     disease='breast cancer',
            ...     tier_filter=[1],
            ...     confidence_threshold=0.8
            ... )
            >>> 
            >>> # Get Tier 1 and 2 datasets for broader analysis
            >>> suitable_datasets = filter_engine.filter(
            ...     datasets,
            ...     disease='alzheimer',
            ...     tier_filter=[1, 2],
            ...     confidence_threshold=0.6
            ... )
        """
        # Input validation
        if not disease or not disease.strip():
            raise ValueError("Disease parameter cannot be empty")
        
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0.0 and 1.0")
        
        if tier_filter is None:
            tier_filter = [1, 2, 3]  # Default to all tiers
        
        if not all(t in [1, 2, 3] for t in tier_filter):
            raise ValueError("tier_filter must contain only values 1, 2, or 3")
        
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
        
        logger.info(f"Starting enhanced GPT filtering for '{disease}' on {len(data)} datasets")
        logger.info(f"Using {self.max_workers} workers, tier filter: {tier_filter}, confidence threshold: {confidence_threshold}")
        
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
                    desc="Processing datasets with enhanced evaluation"
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
                            'gpt_tier': 3,
                            'gpt_reason': f"Unexpected error: {e}",
                            'gpt_confidence': 0.0,
                            'gpt_error': str(e)
                        })
                        results.append(error_record)
        
        except Exception as e:
            raise GPTFilterError(f"Parallel processing failed: {e}")
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Log detailed statistics
        processing_time = time.time() - start_time
        total_count = len(results_df)
        tier1_count = len(results_df[results_df['gpt_tier'] == 1])
        tier2_count = len(results_df[results_df['gpt_tier'] == 2])
        tier3_count = len(results_df[results_df['gpt_tier'] == 3])
        error_count = len(results_df[results_df['gpt_error'].notna()])
        
        logger.info(f"Enhanced GPT filtering completed in {processing_time:.1f}s")
        logger.info(f"Results: Tier 1: {tier1_count}, Tier 2: {tier2_count}, Tier 3: {tier3_count}, Errors: {error_count}")
        
        # Apply filtering
        if return_all:
            return results_df.reset_index(drop=True)
        else:
            # Filter for suitable datasets matching tier and confidence criteria
            suitable_mask = (
                (results_df['gpt_tier'].isin(tier_filter)) & 
                (results_df['gpt_confidence'] >= confidence_threshold)
            )
            filtered_df = results_df[suitable_mask].reset_index(drop=True)
            
            logger.info(f"Filtered to {len(filtered_df)} datasets matching tier {tier_filter} with confidence ≥ {confidence_threshold}")
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
    
    def analyze_results(self, results_df: pd.DataFrame) -> Dict:
        """
        Analyze filtering results and provide summary statistics.
        
        Args:
            results_df: DataFrame with GPT filtering results
            
        Returns:
            Dictionary with detailed analysis
        """
        if results_df.empty:
            return {"message": "No data to analyze"}
        
        analysis = {
            'total_datasets': len(results_df),
            'tier_distribution': {
                'tier_1': len(results_df[results_df['gpt_tier'] == 1]),
                'tier_2': len(results_df[results_df['gpt_tier'] == 2]),
                'tier_3': len(results_df[results_df['gpt_tier'] == 3])
            },
            'avg_confidence': results_df['gpt_confidence'].mean(),
            'tissue_types': results_df['gpt_tissue_type'].value_counts().to_dict(),
            'study_designs': results_df['gpt_study_design'].value_counts().to_dict(),
            'error_rate': len(results_df[results_df['gpt_error'].notna()]) / len(results_df)
        }
        
        return analysis
    
    def __repr__(self) -> str:
        """String representation of the EnhancedGPTFilter instance."""
        return (f"EnhancedGPTFilter(model='{self.model}', max_workers={self.max_workers}, "
                f"temperature={self.temperature})")