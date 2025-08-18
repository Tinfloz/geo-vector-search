import os
import questionary
import pandas as pd
from pathlib import Path
from geo_pysearch.sdk import (
    search_datasets, 
    search_with_enhanced_gpt,
    analyze_gpt_results,
    print_gpt_summary,
    get_cache_info, 
    clear_cache, 
    print_cache_info,
    preload_datasets
)


def search_command():
    """Main search functionality with enhanced GPT filtering options"""
    print("\n🧬 Welcome to GeoVectorSearch CLI 🧬\n")

    # Step 1: Get disease or query
    query = questionary.text("🔍 Enter your disease query or research topic:").ask()
    if not query:
        print("❌ Query is required!")
        return

    # Step 2: Dataset type selection
    dataset_type = questionary.select(
        "🧪 Choose dataset type:",
        choices=["microarray", "rnaseq"]
    ).ask()

    # Step 3: Number of top results
    top_k = questionary.text(
        "📊 How many top results would you like to retrieve?",
        default="50"
    ).ask()
    try:
        top_k = int(top_k.strip())
    except ValueError:
        print("❌ Invalid number format!")
        return

    # Step 4: Cache options
    cache_options = questionary.select(
        "💾 Cache options:",
        choices=[
            "Use default cache location",
            "Specify custom cache directory", 
            "Force re-download files"
        ]
    ).ask()

    cache_dir = None
    force_download = False

    if cache_options == "Specify custom cache directory":
        cache_path = questionary.path("📁 Enter cache directory path:").ask()
        if cache_path:
            cache_dir = Path(cache_path)
    elif cache_options == "Force re-download files":
        force_download = True
        print("⚠️  Files will be re-downloaded even if cached")

    # Step 5: GPT filter toggle and type selection
    gpt_filter_options = questionary.select(
        "🤖 Choose filtering option:",
        choices=[
            "No GPT filtering (semantic search only)",
            "Basic GPT filtering",
            "Enhanced GPT filtering (recommended for DE analysis)",
        ]
    ).ask()

    use_gpt_filter = "No GPT filtering" not in gpt_filter_options
    gpt_filter_type = "enhanced" if "Enhanced" in gpt_filter_options else "basic"

    # Step 6: GPT settings (if enabled)
    confidence_threshold = 0.3
    tier_filter = None
    return_all_gpt_results = False
    api_key = None
    api_url = None

    if use_gpt_filter:
        if gpt_filter_type == "enhanced":
            tier_filter = questionary.checkbox(
                "🎯 Choose tier filtering strategy (default: All tiers):",
                choices=[
                    questionary.Choice("Tier 1 only (highest quality datasets)", value=[1]),
                    questionary.Choice("Tier 1 & 2 (suitable datasets)", value=[1,2]),
                    questionary.Choice("All tiers (includes Tier-3: not fully suitable for DE datasets as well)", value=[1,2,3]),
                ],
                default=[1,2,3]
            ).ask()
            if tier_filter:
                tier_filter = [tier for sublist in tier_filter for tier in sublist]
            
            print(f"\n📋 Selected tiers: {", ".join([str(x) for x in tier_filter])}")

        # Common GPT settings
        confidence_threshold = questionary.text(
            "🎯 Minimum GPT confidence score (0.0 - 1.0)?",
            default="0.3"
        ).ask()
        try:
            confidence_threshold = float(confidence_threshold)
            if not 0.0 <= confidence_threshold <= 1.0:
                raise ValueError("Confidence must be between 0.0 and 1.0")
        except ValueError as e:
            print(f"❌ Invalid confidence score: {e}")
            return

        return_all_gpt_results = questionary.confirm(
            "📁 Return all GPT results (not just filtered)?",
            default=False
        ).ask()

        # API credentials
        print("\n🔐 API Configuration:")
        api_key = questionary.password("Enter your OpenAI API key:").ask()
        if not api_key or not api_key.strip():
            print("\n❌ Error: API key is required for GPT filtering.\n")
            return

        api_url = questionary.text(
            "Enter your OpenAI API URL (eg: https://api.openai.com/v1/chat/completions):"
        ).ask()
        if not api_url or not api_url.strip():
            print("\n❌ Error: API URL is required for GPT filtering.\n")
            return

    # Step 7: Perform search
    print(f"\n🔎 Searching datasets with {gpt_filter_type if use_gpt_filter else 'semantic'} filtering...")
    try:
        # Show progress message for first-time users
        cache_info = get_cache_info(cache_dir)
        if cache_info['total_files'] == 0:
            print("📥 First run detected - downloading and caching dataset files...")
            print("⏳ This may take a few minutes but will be faster on subsequent runs.")

        # Use enhanced search if enhanced GPT filtering is selected
        if use_gpt_filter and gpt_filter_type == "enhanced":
            results = search_with_enhanced_gpt(
                query=query,
                dataset_type=dataset_type,
                top_k=top_k,
                tier_filter=tier_filter,
                confidence_threshold=confidence_threshold,
                return_all_gpt_results=return_all_gpt_results,
                api_key=api_key,
                api_url=api_url,
                cache_dir=cache_dir,
                force_download=force_download
            )
        else:
            # Use standard search for basic or no GPT filtering
            results = search_datasets(
                query=query,
                dataset_type=dataset_type,
                top_k=top_k,
                use_gpt_filter=use_gpt_filter,
                gpt_filter_type=gpt_filter_type,
                confidence_threshold=confidence_threshold,
                return_all_gpt_results=return_all_gpt_results,
                api_key=api_key,
                api_url=api_url,
                cache_dir=cache_dir,
                force_download=force_download
            )

        # Step 8: Display results summary
        if not results.empty:
            print(f"\n✅ Found {len(results)} results!")
            
            # Enhanced GPT results analysis
            if use_gpt_filter and 'gpt_tier' in results.columns:
                print("\n📊 GPT Filtering Summary:")
                print_gpt_summary(results)
            
            # Show top results preview
            print(f"\n📋 Top {min(5, len(results))} results preview:")
            preview_cols = ['gse']
            
            available_cols = results.columns.tolist()
            preview_additions = ['similarity', 'gpt_confidence', 'gpt_tier', 'gpt_tissue_type', 
                               'gpt_study_design', 'gpt_anatomical_relevance']
            
            for col in preview_additions:
                if col in available_cols:
                    preview_cols.append(col)
            
            preview_df = results[preview_cols].head(5)
            print(preview_df.to_string(index=False, max_colwidth=30))
            
            # Show detailed info for enhanced results
            if 'gpt_tier' in results.columns:
                show_details = questionary.confirm(
                    "\n🔍 Show detailed GPT assessments for top results?",
                    default=False
                ).ask()
                
                if show_details:
                    print("\n" + "="*80)
                    print("DETAILED GPT ASSESSMENTS")
                    print("="*80)
                    
                    for idx, (_, row) in enumerate(results.head(3).iterrows(), 1):
                        print(f"\n--- Dataset #{idx}: {row['gse']} ---")
                        print(f"Tier: {row.get('gpt_tier', 'N/A')}")
                        print(f"Confidence: {row.get('gpt_confidence', 'N/A'):.3f}")
                        print(f"Disease Samples: {row.get('gpt_disease_samples', 'N/A')}")
                        print(f"Control Samples: {row.get('gpt_control_samples', 'N/A')}")
                        print(f"Tissue Type: {row.get('gpt_tissue_type', 'N/A')}")
                        print(f"Study Design: {row.get('gpt_study_design', 'N/A')}")
                        print(f"Reason: {row.get('gpt_reason', 'N/A')}")
                        if row.get('gpt_key_limitations') and row.get('gpt_key_limitations') != 'none':
                            print(f"Limitations: {row.get('gpt_key_limitations', 'N/A')}")
            
            # Step 9: Save results
            save_results = questionary.confirm(
                "\n💾 Save results to CSV file?",
                default=True
            ).ask()
            
            if save_results:
                # Generate more descriptive filename
                filter_suffix = ""
                if use_gpt_filter:
                    if gpt_filter_type == "enhanced" and tier_filter:
                        tier_str = "_".join(map(str, sorted(tier_filter)))
                        filter_suffix = f"_enhanced_tier{tier_str}"
                    else:
                        filter_suffix = f"_{gpt_filter_type}"
                
                filename = f"results_{dataset_type}_{query.replace(' ', '_').replace('/', '_').replace("'",'')}{filter_suffix}.csv"
                results.to_csv(filename, index=False)
                print(f"\n✅ Results saved to: {filename}")
                print(f"   📄 Columns saved: {len(results.columns)}")
                print(f"   📊 Rows saved: {len(results)}")
            
            # Show cache info
            print(f"\n💾 Cache info:")
            cache_info = get_cache_info(cache_dir)
            print(f"   📁 Cache location: {cache_info['cache_dir']}")
            print(f"   📦 Cached files: {cache_info['total_files']}")
            print(f"   💿 Cache size: {cache_info['total_size_mb']} MB")
            
        else:
            print("\n⚠️ No results found for your query.")
            if use_gpt_filter and tier_filter and len(tier_filter) == 1 and tier_filter[0] == 1:
                suggest_broader = questionary.confirm(
                    "💡 No Tier 1 datasets found. Would you like to search again with Tier 1 & 2?",
                    default=True
                ).ask()
                if suggest_broader:
                    print("🔄 Searching with broader tier criteria...")
                    print("💡 Please run the search again and select both Tier 1 and Tier 2.")
            
    except Exception as e:
        print(f"\n❌ Error during search: {str(e)}")
        if "API" in str(e):
            print("💡 Tip: Check your API key and URL are correct")
        elif "tier" in str(e).lower():
            print("💡 Tip: Try selecting multiple tiers or lowering confidence threshold")


def cache_management_menu():
    """Cache management submenu"""
    while True:
        choice = questionary.select(
            "💾 Cache Management:",
            choices=[
                "📊 View cache information",
                "🗑️ Clear all cache",
                "🎯 Clear cache for specific dataset type",
                "📥 Preload datasets", 
                "⬅️ Back to main menu"
            ]
        ).ask()

        if choice == "📊 View cache information":
            print("\n📊 Cache Information:")
            print_cache_info()
            
        elif choice == "🗑️ Clear all cache":
            confirm = questionary.confirm(
                "⚠️  Are you sure you want to clear all cached files?",
                default=False
            ).ask()
            if confirm:
                clear_cache()
                print("✅ All cache cleared!")
            
        elif choice == "🎯 Clear cache for specific dataset type":
            dataset_type = questionary.select(
                "Choose dataset type to clear:",
                choices=["microarray", "rnaseq"]
            ).ask()
            confirm = questionary.confirm(
                f"⚠️  Clear cache for {dataset_type} datasets?",
                default=False
            ).ask()
            if confirm:
                clear_cache(dataset_type=dataset_type)
                print(f"✅ Cache cleared for {dataset_type} datasets!")
                
        elif choice == "📥 Preload datasets":
            dataset_choices = questionary.checkbox(
                "Select datasets to preload:",
                choices=[
                    questionary.Choice("🧬 Microarray datasets", value="microarray"),
                    questionary.Choice("🧪 RNA-seq datasets", value="rnaseq")
                ]
            ).ask()
            
            if dataset_choices:
                force = questionary.confirm(
                    "🔄 Force re-download even if cached?",
                    default=False
                ).ask()
                
                print(f"\n📥 Preloading {', '.join(dataset_choices)} datasets...")
                try:
                    preload_datasets(dataset_choices, force_download=force)
                    print("✅ Datasets preloaded successfully!")
                except Exception as e:
                    print(f"❌ Error preloading datasets: {str(e)}")
            else:
                print("No datasets selected.")
                
        elif choice == "⬅️ Back to main menu":
            break


def help_menu():
    """Enhanced help menu with GPT filtering information"""
    help_choice = questionary.select(
        "❓ Help Topics:",
        choices=[
            "📚 General Overview",
            "🔍 Search Tips", 
            "🤖 GPT Filtering Guide",
            "💾 Cache Management",
            "🔧 Troubleshooting",
            "⬅️ Back to main menu"
        ]
    ).ask()
    
    if help_choice == "📚 General Overview":
        print("""
🧬 GeoDatasetFinder CLI - General Overview

This tool helps you search for genomic datasets using semantic search and optional GPT filtering.

Key Features:
• 🔍 Semantic search across 50,000+ microarray and RNA-seq datasets  
• 🤖 AI-powered filtering for differential expression analysis suitability
• 💾 Automatic file caching for faster subsequent searches
• 📊 Customizable result filtering and detailed analysis
• 📁 CSV export with comprehensive metadata

Workflow:
1. Enter your disease/research query
2. Choose dataset type (microarray/RNA-seq)
3. Optionally apply GPT filtering for quality assessment
4. Review and export results
        """)
        
    elif help_choice == "🔍 Search Tips":
        print("""
🔍 Search Tips for Better Results

Query Best Practices:
• Use specific disease names: "breast cancer" not just "cancer"
• Include relevant terms: "alzheimer disease brain tissue"
• Try synonyms if few results: "myocardial infarction" vs "heart attack"
• Be concise: avoid very long complex queries

Examples of Good Queries:
• "breast cancer"
• "type 2 diabetes"
• "alzheimer disease"
• "lung cancer adenocarcinoma"
• "inflammatory bowel disease"
• "multiple sclerosis brain"

Dataset Types:
• Microarray: Older technology, more datasets available
• RNA-seq: Newer technology, better for novel transcript discovery
        """)
        
    elif help_choice == "🤖 GPT Filtering Guide":
        print("""
🤖 GPT Filtering Guide

Filter Types:
• Basic: Simple suitable/not suitable classification
• Enhanced: 3-tier system with detailed assessment (recommended)

Enhanced GPT Tiers:
• Tier 1: Directly suitable for DE analysis
  - Primary human tissue samples
  - Clear case-control design  
  - Adequate sample sizes (≥5 per group)
  
• Tier 2: Conditionally suitable  
  - Cell lines or smaller sample sizes
  - Mixed conditions but extractable comparisons
  - Animal models appropriate for the disease
  
• Tier 3: Not suitable
  - No controls or inappropriate study design
  - Wrong tissue type for disease
  - Technical/methodological studies only

Configuration Tips:
• Start with Tier 1 only for highest quality datasets
• Add Tier 2 if you need more options
• API key required - get one from OpenAI platform
        """)
        
    elif help_choice == "💾 Cache Management":
        print("""
💾 Cache Management

How Caching Works:
• Files automatically downloaded from Hugging Face Hub on first use
• Cached locally (~100-500MB per dataset type)
• Subsequent searches are much faster

Cache Operations:
• View Info: See cache location and size
• Clear All: Remove all cached files (will re-download on next use)
• Clear Specific: Remove cache for one dataset type only  
• Preload: Download files before searching (recommended for first use)

Tips:
• Preload datasets during off-hours for faster searches later
• Clear cache if you suspect file corruption
• Default cache location is user-specific temp directory
        """)
        
    elif help_choice == "🔧 Troubleshooting":
        print("""
🔧 Troubleshooting Common Issues

No Results Found:
• Try broader or more specific query terms
• Check spelling of disease names
• Try different dataset type (microarray vs RNA-seq)
• Lower GPT confidence threshold or add more tiers

API/GPT Errors:
• Verify OpenAI API key is correct
• Check API URL format: https://api.openai.com/v1/chat/completions
• Ensure sufficient API credits/quota
• Try reducing batch size (lower top_k)

Download/Cache Issues:
• Check internet connection
• Try clearing cache and re-downloading
• Verify write permissions in cache directory
• Check available disk space (need ~1GB free)

Performance Issues:
• First run is slower due to downloads
• Preload datasets during off-peak hours
• Use SSD storage for cache if possible
• Reduce top_k for faster processing
        """)
    
    elif help_choice == "⬅️ Back to main menu":
        return


def main():
    """Main CLI entry point with enhanced interface"""
    while True:
        choice = questionary.select(
            "\n🧬 GeoDatasetFinder CLI - Main Menu:",
            choices=[
                "🔍 Search for datasets",
                "💾 Cache management", 
                "❓ Help & Guide",
                "🚪 Exit"
            ]
        ).ask()

        if choice == "🔍 Search for datasets":
            search_command()
            
        elif choice == "💾 Cache management":
            cache_management_menu()
            
        elif choice == "❓ Help & Guide":
            help_menu()
            
        elif choice == "🚪 Exit":
            print("""
👋 Thanks for using GeoDatasetFinder CLI!

🌟 Found this tool useful? 
   Star us on GitHub: https://github.com/Tinfloz/geo-vector-search
   
📚 Need more help?
   Check our documentation for advanced usage examples
   
🤝 Have feedback?
   Open an issue or submit a PR - we'd love to hear from you!
            """)
            break


if __name__ == "__main__":
    main()