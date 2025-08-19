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
    primary_method: str = ""
    exclusion_check: str = ""
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
        return f"""# Biomedical Data Curation Assistant - STRICT CLASSIFICATION

## CRITICAL INSTRUCTIONS - READ CAREFULLY
You are evaluating datasets for BULK transcriptomic Differential Expression (DE) analysis comparing {disease} samples against controls. This requires BULK tissue/cell samples for either RNA-seq OR microarray analysis, NOT single-cell data.

## Dataset Information
- **GSE ID**: {record.get('gse', 'N/A')}
- **Summary**: {record.get('cleaned_text', 'N/A')}
- **Target Disease**: {disease}

## CONTEXTUAL ANALYSIS FRAMEWORK

**The key question is: What did the researchers ACTUALLY DO in their experiment?**

### STEP 1: IDENTIFY PRIMARY EXPERIMENTAL METHOD

**Use these analytical principles:**

1. **Look for ACTIVE experimental descriptions:**
   - What verbs describe the main methodology?
   - What was the primary sample processing approach?
   - What platform/technology was the core method?

2. **Distinguish between PRIMARY vs SECONDARY mentions:**
   - **PRIMARY**: The main experimental approach described
   - **SECONDARY**: Background, comparisons, related work, or additional context

3. **Focus on SAMPLE PROCESSING and ANALYSIS:**
   - How were the biological samples processed?
   - What was sequenced/analyzed at what resolution?
   - Individual cells/nuclei OR bulk tissue/cell populations?

4. **Identify the RESEARCH QUESTION and DESIGN:**
   - Is the study designed around single-cell resolution?
   - Or is it designed for population-level (bulk) analysis?

### EXCLUSION DECISION FRAMEWORK

**EXCLUDE (Tier 3) if PRIMARY method is:**
- Single-cell or single-nucleus RNA sequencing
- Spatial transcriptomics
- Non-coding RNA focused (miRNA/lncRNA only)
- Proteomics/metabolomics
- Technical platform comparisons
- Sequencing methods other than bulk RNA-seq (e.g., ChIP-seq, ATAC-seq, methylation)

**Key reasoning approach:**
- Ask: "If I had to reproduce this study, what would be my main experimental protocol?"
- The answer to that question determines the primary method

**CONTINUE ANALYSIS if:**
- Primary method is bulk tissue/cell population RNA sequencing OR microarray analysis
- Even if single-cell technologies are mentioned for context/comparison

## BULK RNA-SEQ CLASSIFICATION FRAMEWORK

### TIER 1: Optimal for Bulk DE Analysis
**ALL criteria must be met:**

**Sample Requirements (MANDATORY):**
- BULK tissue or cell population samples (not single cells)
- Disease samples AND appropriate controls present (healthy/normal)
- Minimum 3 biological replicates per group (prefer ≥5)
- Clear case-control design

**Biological Relevance (MANDATORY):**
- Primary human tissue directly relevant to {disease} pathophysiology
- Anatomically appropriate tissue type
- Disease-relevant context

**Study Design (MANDATORY):**
- Designed for disease comparison (not intervention/treatment primary focus)
- Controls are healthy/normal (not just different disease states)
- Adequate metadata for proper DE analysis

**Technology Requirements:**
- Bulk RNA-seq with standard library preparation OR
- Microarray with appropriate probe coverage and normalization

### TIER 2: Suitable with Limitations
**Must have bulk samples AND disease vs control comparison, PLUS:**

**Acceptable Limitations:**
- Disease-relevant cell lines or well-characterized models
- Smaller sample sizes (2-4 per group) but biologically meaningful
- Mixed sample types but includes relevant material
- Intervention studies with baseline disease vs control comparisons
- Animal models standard for the disease field

### TIER 3: Not Suitable
**Any of these automatically disqualifies:**
- Single-cell or single-nucleus data
- No control samples or inappropriate controls
- Wrong tissue/system for disease biology
- Only treatment response without baseline comparison
- Technical replicates only (no biological replicates)
- Non-coding RNA focus (eg: miRNA/lncRNA only studies)
- Spatial transcriptomics
- Methodological/technical studies
- Non-transcriptomic methods (proteomics, metabolomics, epigenomics)

## CLASSIFICATION PROCESS

### STEP 1: PRIMARY METHOD IDENTIFICATION
**Read the entire summary and ask:**
1. "What is the main experimental approach described?"
2. "How were samples processed for analysis?"
3. "What technology/platform was used as the core method?"
4. "What resolution does the analysis operate at - cellular or tissue/population level?"

**Decision logic:**
- If the core experimental design is single-cell/nucleus → EXCLUDE
- If the core experimental design is bulk tissue/population transcriptomics → CONTINUE
- If unclear or mixed methods → Default to EXCLUDE (Tier 3)

### STEP 2: BULK TRANSCRIPTOMIC CONFIRMATION  
**For studies that pass Step 1, confirm bulk transcriptomic characteristics:**
- Population-level gene expression analysis
- Tissue or cell population samples (not individual cells)
- Standard RNA-seq library preparation OR microarray hybridization workflows
- Focus on aggregate/average expression across sample groups

### STEP 3: DISEASE-CONTROL ASSESSMENT
**Check for:**
- Disease samples present: YES/NO
- Control samples present: YES/NO
- Sample sizes adequate: YES/NO

### STEP 4: BIOLOGICAL RELEVANCE
**Evaluate:**
- Tissue type appropriate for {disease}: YES/NO
- Primary human samples preferred: YES/NO
- Disease context relevant: YES/NO

### STEP 5: STUDY DESIGN
**Assess:**
- Designed for disease comparison: YES/NO
- Appropriate controls: YES/NO
- Suitable for DE analysis: YES/NO

## MANDATORY OUTPUT FORMAT

```
Primary Method: [bulk_rnaseq/microarray/single_cell/spatial/non_coding_rna/proteomics/unclear]
Exclusion Check: [PASS/FAIL] - [detailed reason if failed]
Tier: [1/2/3]
Disease Samples: [YES/NO] (count if available)
Control Samples: [YES/NO] (count if available)
Tissue Type: [primary_human/cell_line/animal_model/other]
Anatomical Relevance: [HIGH/MODERATE/LOW/IRRELEVANT]
Study Design: [case_control/intervention/time_series/other]
Reason: [Clear explanation with specific evidence from summary]
Key Limitations: [Specific limitations or "none"]
Confidence: [0.0-1.0]
```

## QUALITY ASSURANCE RULES

1. **If summary mentions single-cell/single-nucleus → Tier 3 ALWAYS**
2. **If unclear whether bulk or single-cell → Default Tier 3**
3. **If no clear controls mentioned → Tier 3**
4. **If wrong tissue type for disease → Tier 3**
5. **If studying a DIFFERENT disease than target → Tier 3 ALWAYS**
    - Even if diseases are related or share symptoms
    - Even if mentioned as secondary/related condition
    - Primary disease focus must match target disease exactly
6. **When in doubt → Choose lower tier**

## CONFIDENCE SCORING
- **0.9-1.0**: Crystal clear classification with strong evidence
- **0.7-0.8**: Good confidence, minor ambiguities
- **0.5-0.6**: Moderate confidence, some uncertainty
- **0.3-0.4**: Low confidence, significant ambiguity
- **0.0-0.2**: Very uncertain, insufficient information

## REASONING EXAMPLES (Principles, not exhaustive rules)

**Example reasoning for PRIMARY method identification:**

**Case A: Single-cell primary**
Summary: "We performed single-nucleus RNA sequencing of brain tissue from PD patients..."
Analysis: Core verb is "performed single-nucleus RNA sequencing" - this is the main experimental method
Decision: EXCLUDE - Primary method is single-nucleus

**Case B: Bulk primary with single-cell context**  
Summary: "Previous single-cell studies suggested heterogeneity. We used bulk tissue RNA-seq to measure average expression changes..."
Analysis: Core method is "bulk tissue RNA-seq" - single-cell is background context
Decision: CONTINUE - Primary method is bulk RNA-seq

**Case C: Unclear methodology**
Summary: "We analyzed gene expression in disease samples using genomic approaches..."
Analysis: No clear indication of single-cell vs bulk methodology
Decision: EXCLUDE (default when unclear)

**Case D: Wrong disease focus**
Summary: "We performed bulk RNA-seq of cerebellar tissue from Multiple System Atrophy patients and controls..."
Target Disease: Parkinson's disease
Analysis: Core method is bulk RNA-seq, but primary disease is MSA, not Parkinson's
Decision: EXCLUDE - Wrong disease focus (MSA ≠ Parkinson's)

The key is asking: "What would someone need to do to replicate this study's main findings?"

Remember: Be STRICT. When uncertain, default to Tier 3. The goal is to identify only clearly suitable bulk RNA-seq datasets for robust differential expression analysis."""
    
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
            if self._extract_field(line, "primary method:", result, 'primary_method', str):
                continue
            if self._extract_field(line, "exclusion check:", result, 'exclusion_check', str):
                continue
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
        if not result.primary_method:
            result.primary_method = "unclear"
        if not result.exclusion_check:
            result.exclusion_check = "unclear"
        
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
                'gpt_primary_method': gpt_result.primary_method,
                'gpt_exclusion_check': gpt_result.exclusion_check,
                'gpt_tier': gpt_result.tier,
                'gpt_disease_samples': gpt_result.disease_samples,
                'gpt_control_samples': gpt_result.control_samples,
                'gpt_tissue_type': gpt_result.tissue_type,
                'gpt_anatomical_relevance': gpt_result.anatomical_relevance,
                'gpt_study_design': gpt_result.study_design,
                'gpt_reason': gpt_result.reason,
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
                'gpt_primary_method':'Error',
                'gpt_exclusion_check': 'Error',
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