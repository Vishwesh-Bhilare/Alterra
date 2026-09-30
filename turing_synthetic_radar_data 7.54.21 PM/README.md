---
license: apache-2.0
tags:
- radar
- deinterleaving
- EW
pretty_name: T
size_categories:
- 1B<n<10B
---

# The Turing Synthetic Radar Dataset (TSRD)

## Dataset Summary

The Turing Synthetic Radar Dataset is the first publicly available, comprehensively simulated pulse train dataset designed for radar pulse deinterleaving research. It provides a large-scale benchmark for developing and evaluating electronic warfare (EW) and signal intelligence (SIGINT) applications, enabling researchers to address the critical challenge of separating interleaved radar pulses from multiple unknown emitters.

NOTE: We provide a small summary below but substantially more detailed documentation as well as scripts for workflow examples, downloading, and evaluation etc. are available in the [Turing Deinterleaving Challenge GitHub repository](https://github.com/alan-turing-institute/turing-deinterleaving-challenge). Please open a discussion if you have any queries or clarifications.

## Dataset Details

### Size and Composition
- **Total pulse trains**: 6,000 (2,500 training, 250 validation, 250 test per receiver mode)
- **Total pulses**: ~4 billion
- **Emitters**: Up to 90 per pulse train
- **Data format**: Pulse Descriptor Words (PDWs): sequences of 5-dimensional feature vectors

### Features
Each PDW contains:
- **Time of Arrival (ToA)**: Microseconds
- **Centre Frequency**: MHz
- **Pulse Width**: Microseconds
- **Angle of Arrival (AoA)**: Degrees
- **Amplitude**: dB

### Receiver Modes

**Stare Mode**: Oracle receiver detecting all signals across the entire frequency spectrum (0-18 GHz) simultaneously over 10 seconds
- ~3.8 billion total pulses
- Average 1.29 million pulses per training pulse train
- Captures up to 85 emitters

**Scan Mode**: Realistic receiver sweeping through frequency bands
- ~282 million total pulses
- Average 94 thousand pulses per training pulse train
- Captures up to 90 emitters

## Realism and Complexity

### Challenges
- Significant parameter space overlap between emitters
- Label imbalance (up to 99.7% dominated by single emitter)
- Unknown number of emitters at test time
- Varying sequence lengths
- Complex emitter behaviors (frequency hopping, staggered PRI, agile systems)

## Evaluation Framework

### Metrics
- **Clustering metrics**: V-measure, Adjusted Rand Index (ARI), Adjusted Mutual Information (AMI), homogeneity, completeness
- **Pairwise-binary metrics**: MCC, F1 score

### Challenge
The Turing Deinterleaving Challenge provides:
- Standardised evaluation procedures
- Public and private leaderboards
- Python library for data loading and preprocessing
- GitHub repository with evaluation utilities

## Data Access

- **Repository**: GitHub (evaluation utilities and documentation)
- **Citation**: Gunn et al., "The Turing Synthetic Radar Dataset: A dataset for pulse deinterleaving"

## Authors

Edward Gunn, Adam Hosford, Robert Jones, Leo Zeitler, Ian Groves, Victoria Nockles

## Acknowledgments

This work was supported by the Turing's Defence and Security programme through a partnership with the UK government.