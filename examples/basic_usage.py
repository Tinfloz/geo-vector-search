from geo_pysearch.sdk import search_microarray, search_rnaseq, search_with_gpt, search_datasets

# Basic microarray search without GPT filtering
results_microarray = search_datasets("breast cancer", dataset_type="microarray", top_k=30)

# RNA-seq dataset search without GPT
results_rnaseq = search_datasets("Parkinson's disease", dataset_type="rnaseq", top_k=25)

# GPT filtering enabled
results_gpt_filtered = search_datasets("duchenne muscular dystrophy", dataset_type="microarray", top_k=50, use_gpt_filter=True, return_all_gpt_results=True)
results_gpt_filtered.to_csv("examples/test.csv")

# Microarray dataset search using convenience wrapper
results_microarray_wrapper = search_microarray("Alzheimer's disease", top_k=40)


