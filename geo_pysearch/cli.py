import os
import questionary
import pandas as pd
from geo_pysearch.sdk import search_datasets


def main():
    print("\n🧬 Welcome to GeoDatasetFinder CLI 🧬\n")

    # Step 1: Get disease or query
    query = questionary.text("🔍 Enter your disease query or research topic:").ask()

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
    top_k = int(top_k.strip())

    # Step 4: GPT filter toggle
    use_gpt_filter = questionary.confirm(
        "🤖 Apply GPT filtering for DE suitability?",
        default=False
    ).ask()

    # Step 5: GPT settings (if enabled)
    confidence_threshold = 0.6
    return_all_gpt_results = False
    api_key = None
    api_url = None

    if use_gpt_filter:
        confidence_threshold = float(questionary.text(
            "🎯 Minimum GPT confidence score (0.0 - 1.0)?",
            default="0.6"
        ).ask())

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

    # Step 6: Perform search
    print("\n🔎 Searching datasets...")
    results = search_datasets(
        query=query,
        dataset_type=dataset_type,
        top_k=top_k,
        use_gpt_filter=use_gpt_filter,
        confidence_threshold=confidence_threshold,
        return_all_gpt_results=return_all_gpt_results,
        api_key=api_key,
        api_url=api_url
    )

    # Step 7: Save results
    if not results.empty:
        filename = f"results_{dataset_type}_{query.replace(' ', '_')}.csv"
        results.to_csv(filename, index=False)
        print(f"\n✅ Search completed! Results saved to: {filename}\n")
    else:
        print("\n⚠️ No results found.\n")


if __name__ == "__main__":
    main()
