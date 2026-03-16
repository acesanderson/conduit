- [ ] finalize data models
- [ ] recreate the more simplified gold standard dataset

### Strategies (5) for run matrix
one_shot.py
recursive.py
rolling_refine.py
extractive_pre_filter.py
hierarchical_tree.py

### Our config params for RecursiveSummarizer
- model
- effective_context_window_percentage
- chunk length
- chunk overlap
- temperature
- max tokens (i.e. num_ctx)

### How to handle token cutoffs?
- Make sure data set has a large continuous set of documents across multiple token lengths. Optimize for shorter documents first, identify where this noticeably breaks, then do the same for the next range of token lengths. Do this over and over again.

### Strategy
- generate new golden standard
- initial run as dry run
- create run matrix
- run on run matrix
