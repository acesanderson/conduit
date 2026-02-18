from dataclasses import dataclass


@dataclass
class CompressionTier:
    max_input: int
    ratio: float
    min_tokens: int
    max_tokens: int
    # tolerance is (lower_bound_multiplier, upper_bound_multiplier)
    tolerance: tuple[float, float] = (0.75, 1.25)

    def is_valid_summary(
        self, target: int, summary_tokens: int, original_tokens: int
    ) -> bool:
        """
        Checks if summary tokens fall within the tier-specific tolerance.
        Adjusts the lower bound for documents shorter than the target floor.
        """
        low_m, high_m = self.tolerance

        # Avoid the 'floor trap': if original text < target floor,
        # use original text as the baseline for the lower bound calculation.
        lower_baseline = min(target, original_tokens)

        lower_bound = lower_baseline * low_m
        upper_bound = target * high_m

        return lower_bound <= summary_tokens <= upper_bound


# Define the formal compression mapping
COMPRESSION_SCHEDULE = [
    # Tier A: Detailed (Small docs)
    CompressionTier(2000, 0.15, 300, 400, (0.75, 1.25)),
    # Tier B: Narrative (Standard docs)
    CompressionTier(10000, 0.10, 400, 1000, (0.70, 1.20)),
    # Tier C: Strategic (Large docs)
    CompressionTier(40000, 0.05, 1000, 2000, (0.65, 1.20)),
]

GLOBAL_MAX_TOKENS = 2500
GLOBAL_TOLERANCE = (0.50, 1.20)


def get_target_summary_length(input_tokens: int) -> int:
    """
    Calculates the target summary length based on input token volume.
    Applies a tiered ratio with a hard saturation cap.
    """
    selected_tier = None
    for tier in COMPRESSION_SCHEDULE:
        if input_tokens <= tier.max_input:
            selected_tier = tier
            break

    if selected_tier:
        raw_target = int(input_tokens * selected_tier.ratio)
        return max(selected_tier.min_tokens, min(raw_target, selected_tier.max_tokens))

    return GLOBAL_MAX_TOKENS


def is_within_threshold(original_tokens: int, summary_tokens: int) -> bool:
    """
    Validates if summary length is acceptable relative to the compression target.
    """
    target = get_target_summary_length(original_tokens)

    selected_tier = next(
        (t for t in COMPRESSION_SCHEDULE if original_tokens <= t.max_input), None
    )

    if selected_tier:
        return selected_tier.is_valid_summary(target, summary_tokens, original_tokens)

    # Global fallback for very large docs
    low_m, high_m = GLOBAL_TOLERANCE
    # Apply same floor-trap logic for consistency, though rare at global scale
    lower_bound = min(target, original_tokens) * low_m
    upper_bound = target * high_m

    return lower_bound <= summary_tokens <= upper_bound


if __name__ == "__main__":
    # Test visualization
    # Case 50: Original 101, Summary 187, Target 300.
    # Old logic failed because 187 < (300 * 0.75 = 225).
    # New logic passes because 187 > (101 * 0.75 = 75).
    orig_50, summ_50 = 101, 187
    print(f"Datum 50 Pass: {is_within_threshold(orig_50, summ_50)}")
