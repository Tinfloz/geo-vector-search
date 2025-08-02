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

    # Step 5: GPT confidence threshold (if filtering)
    confidence_threshold = 0.6
    return_all_gpt_results = False
    if use_gpt_filter:
        confidence_threshold = float(questionary.text(
            "🎯 Minimum GPT confidence score (0.0 - 1.0)?",
            default="0.6"
        ).ask())

        return_all_gpt_results = questionary.confirm(
            "📁 Return all GPT results (not just filtered)?",
            default=False
        ).ask()

    # Step 6: Perform search
    print("\n🔎 Searching datasets...")
    results = search_datasets(
        query=query,
        dataset_type=dataset_type,
        top_k=top_k,
        use_gpt_filter=use_gpt_filter,
        confidence_threshold=confidence_threshold,
        return_all_gpt_results=return_all_gpt_results
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