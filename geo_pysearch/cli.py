import os
import questionary
import pandas as pd
from pathlib import Path
from geo_pysearch.sdk import (
    search_datasets, 
    get_cache_info, 
    clear_cache, 
    print_cache_info,
    preload_datasets
)


def search_command():
    """Main search functionality"""
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

    # Step 5: GPT filter toggle
    use_gpt_filter = questionary.confirm(
        "🤖 Apply GPT filtering for DE suitability?",
        default=False
    ).ask()

    # Step 6: GPT settings (if enabled)
    confidence_threshold = 0.6
    return_all_gpt_results = False
    api_key = None
    api_url = None

    if use_gpt_filter:
        confidence_threshold = questionary.text(
            "🎯 Minimum GPT confidence score (0.0 - 1.0)?",
            default="0.6"
        ).ask()
        try:
            confidence_threshold = float(confidence_threshold)
        except ValueError:
            print("❌ Invalid confidence score format!")
            return

        return_all_gpt_results = questionary.confirm(
            "📁 Return all GPT results (not just filtered)?",
            default=False
        ).ask()

        # API key (hidden input)
        api_key = questionary.password("🔐 Enter your OpenAI API key:").ask()
        if not api_key or not api_key.strip():
            print("\n❌ Error: API key is required for GPT filtering.\n")
            return

        # API URL (required, no default)
        api_url = questionary.text("🌐 Enter your OpenAI API URL (e.g. https://api.openai.com/v1):").ask()
        if not api_url or not api_url.strip():
            print("\n❌ Error: API URL is required for GPT filtering.\n")
            return

    # Step 7: Perform search
    print("\n🔎 Searching datasets...")
    try:
        # Show progress message for first-time users
        cache_info = get_cache_info(cache_dir)
        if cache_info['total_files'] == 0:
            print("📥 First run detected - downloading and caching dataset files...")
            print("⏳ This may take a few minutes but will be faster on subsequent runs.")

        results = search_datasets(
            query=query,
            dataset_type=dataset_type,
            top_k=top_k,
            use_gpt_filter=use_gpt_filter,
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
            
            # Show top 3 results preview
            print("\n📋 Top 3 results preview:")
            preview_cols = ['gse', 'similarity']
            if 'gpt_confidence' in results.columns:
                preview_cols.append('gpt_confidence')
            
            print(results[preview_cols].head(3).to_string(index=False))
            
            # Step 9: Save results
            save_results = questionary.confirm(
                "💾 Save results to CSV file?",
                default=True
            ).ask()
            
            if save_results:
                filename = f"results_{dataset_type}_{query.replace(' ', '_').replace('/', '_')}.csv"
                results.to_csv(filename, index=False)
                print(f"\n✅ Results saved to: {filename}")
            
            # Show cache info
            print(f"\n📊 Cache info:")
            cache_info = get_cache_info(cache_dir)
            print(f"   Cache location: {cache_info['cache_dir']}")
            print(f"   Cached files: {cache_info['total_files']}")
            print(f"   Cache size: {cache_info['total_size_mb']} MB")
            
        else:
            print("\n⚠️ No results found for your query.")
            
    except Exception as e:
        print(f"\n❌ Error during search: {str(e)}")


def cache_management_menu():
    """Cache management submenu"""
    while True:
        choice = questionary.select(
            "💾 Cache Management:",
            choices=[
                "View cache information",
                "Clear all cache",
                "Clear cache for specific dataset type",
                "Preload datasets", 
                "Back to main menu"
            ]
        ).ask()

        if choice == "View cache information":
            print("\n📊 Cache Information:")
            print_cache_info()
            
        elif choice == "Clear all cache":
            confirm = questionary.confirm(
                "⚠️  Are you sure you want to clear all cached files?",
                default=False
            ).ask()
            if confirm:
                clear_cache()
                print("✅ All cache cleared!")
            
        elif choice == "Clear cache for specific dataset type":
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
                
        elif choice == "Preload datasets":
            dataset_choices = questionary.checkbox(
                "Select datasets to preload:",
                choices=["microarray", "rnaseq"]
            ).ask()
            
            if dataset_choices:
                force = questionary.confirm(
                    "Force re-download even if cached?",
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
                
        elif choice == "Back to main menu":
            break


def main():
    """Main CLI entry point"""
    while True:
        choice = questionary.select(
            "\n🧬 GeoDatasetFinder CLI - Main Menu:",
            choices=[
                "🔍 Search for datasets",
                "💾 Cache management", 
                "❓ Help",
                "🚪 Exit"
            ]
        ).ask()

        if choice == "🔍 Search for datasets":
            search_command()
            
        elif choice == "💾 Cache management":
            cache_management_menu()
            
        elif choice == "❓ Help":
            print("""
🧬 GeoDatasetFinder CLI Help

This tool helps you search for genomic datasets using semantic search and optional GPT filtering.

Features:
• 🔍 Semantic search across microarray and RNA-seq datasets
• 🤖 Optional GPT-powered filtering for differential expression suitability
• 💾 Automatic file caching for faster subsequent searches
• 📊 Customizable result filtering and export

Cache Management:
• Files are automatically downloaded from Hugging Face Hub on first use
• Cached locally for faster access on subsequent runs
• Use cache management to view, clear, or preload datasets

Tips:
• First run will be slower due to file downloads (~100-500MB per dataset type)
• Use specific, descriptive queries for better results
• GPT filtering requires an OpenAI API key but provides more relevant results

For more information, visit: https://github.com/Tinfloz/geo-vector-search
            """)
            
        elif choice == "🚪 Exit":
            print("\n👋 Thanks for using GeoVectorSearch CLI!")
            break


if __name__ == "__main__":
    main()