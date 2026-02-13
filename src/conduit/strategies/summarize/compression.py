from dataclasses import dataclass


@dataclass
class CompressionTier:
    max_input: int
    ratio: float
    min_tokens: int
    max_tokens: int


# Define the formal compression mapping
# Tier A: Detailed (20% for very short, 10% for standard)
# Tier B: Narrative (5% for long technical docs)
# Tier C: Strategic (Fixed cap for massive datasets)
COMPRESSION_SCHEDULE = [
    CompressionTier(max_input=2000, ratio=0.15, min_tokens=300, max_tokens=400),
    CompressionTier(max_input=10000, ratio=0.10, min_tokens=400, max_tokens=1000),
    CompressionTier(max_input=40000, ratio=0.05, min_tokens=1000, max_tokens=2000),
]

GLOBAL_MAX_TOKENS = 2500


def get_target_summary_length(input_tokens: int) -> int:
    """
    Calculates the target summary length based on input token volume.
    Applies a tiered ratio with a hard saturation cap.
    """
    target = 0

    # Find the appropriate tier
    selected_tier = None
    for tier in COMPRESSION_SCHEDULE:
        if input_tokens <= tier.max_input:
            selected_tier = tier
            break

    if selected_tier:
        raw_target = int(input_tokens * selected_tier.ratio)
        # Clamp between tier min/max
        target = max(
            selected_tier.min_tokens, min(raw_target, selected_tier.max_tokens)
        )
    else:
        # Default for anything above 40k
        target = GLOBAL_MAX_TOKENS

    return target


if __name__ == "__main__":
    # Standalone test to visualize the map
    test_values = [500, 1500, 5000, 15000, 50000, 100000]
    print(f"{'Input Tokens':<15} | {'Target Tokens':<15} | {'Effective Ratio':<15}")
    print("-" * 50)
    for val in test_values:
        t = get_target_summary_length(val)
        ratio = (t / val) * 100
        print(f"{val:<15} | {t:<15} | {ratio:>5.1f}%")
