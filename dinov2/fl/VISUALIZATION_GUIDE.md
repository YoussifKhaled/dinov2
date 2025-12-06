# Visualization Guide: Embedding-Based FL Data Heterogeneity

This document explains each visualization in the analysis notebook and how to interpret the results.

---

## Overview

The pipeline generates **10 visualizations** organized into 4 sections:
1. **Embedding Space Analysis** - Understanding DINOv2 representations
2. **Clustering Analysis** - Understanding discovered scene types
3. **Client Partition Analysis** - Understanding data distribution
4. **Alpha Comparison** - Understanding heterogeneity control

---

## Section 3.2: Embedding Space Analysis

### Visualization 1: t-SNE by City (`viz1_tsne_by_city.png`)

**What it shows:** 2D projection of 1024-dimensional DINOv2 embeddings, colored by source city (e.g., Frankfurt, Munster, etc.)

**How to read it:**
- Each dot = one Cityscapes image
- Color = which city the image is from
- Proximity = visual similarity according to DINOv2

**What to look for:**
| Pattern | Meaning |
|---------|---------|
| Cities form separate clusters | Location-specific visual patterns dominate (architecture, road style) |
| Cities are mixed together | DINOv2 captures scene semantics beyond geography |
| Partial mixing | Both location and semantic factors matter |

**Why it matters:** If cities naturally cluster together, then a city-based FL partition would already be non-IID. Our embedding-based approach captures deeper semantic similarities.

---

## Section 3.3: Clustering Analysis

### Visualization 2: t-SNE by Cluster (`viz2_tsne_by_cluster.png`)

**What it shows:** Same t-SNE projection, but colored by K-Means cluster assignment.

**How to read it:**
- Each color = one cluster (discovered scene type)
- Compact clusters = clear semantic categories
- Overlapping clusters = gradual transitions

**What to look for:**
| Pattern | Meaning |
|---------|---------|
| Well-separated clusters | K-Means found distinct scene types |
| Overlapping clusters | Some scenes are transitional (e.g., suburban ↔ urban) |
| Fragmented clusters | K may be too high, or scenes don't have clean boundaries |

---

### Visualization 3: Cluster Size Distribution (`viz3_cluster_sizes.png`)

**What it shows:** Bar chart of how many images belong to each cluster.

**How to read it:**
- X-axis = Cluster ID (0 to K-1)
- Y-axis = Number of images
- Red line = Mean across clusters

**What to look for:**
| Pattern | Meaning |
|---------|---------|
| Uniform bars | Scene types are equally represented |
| Very uneven bars | Some scene types are rare (e.g., highways) or common (e.g., urban) |
| One dominant cluster | K might be too low, or dataset is homogeneous |

**Why it matters:** Cluster sizes affect how Dirichlet sampling works. Very uneven clusters can lead to some clients having almost no data from rare clusters.

---

### Visualization 4: Cluster Samples (`viz4_cluster_samples.png`)

**What it shows:** Grid of actual images from each cluster.

**How to read it:**
- Each row = one cluster
- Images in same row should look visually similar

**What to look for:**
| Pattern | Meaning |
|---------|---------|
| Clear visual themes per row | Clustering captured meaningful scene types |
| Mixed/random images per row | Clustering may not be semantically meaningful |
| Gradients across rows | Clusters represent a continuum (e.g., sparse → dense) |

**Common cluster themes in Cityscapes:**
- Highway/freeway scenes
- Dense urban intersections
- Residential streets
- Tunnel/underpass
- Straight roads with perspective
- Parking areas

---

## Section 3.4: Client Partition Analysis

### Visualization 5: Client Overview (`viz5_client_overview.png`)

**What it shows:** Two bar charts showing (1) samples per client and (2) cluster diversity per client.

**Left chart - Sample Count:**
- X-axis = Client ID
- Y-axis = Number of images assigned
- Red line = Mean

**Right chart - Cluster Diversity:**
- X-axis = Client ID
- Y-axis = Number of unique clusters in client's data
- Green line = Maximum possible (K clusters)

**What to look for:**
| Metric | Low α (non-IID) | High α (IID) |
|--------|-----------------|--------------|
| Sample count range | Wide variation | Narrow variation |
| Clusters per client | Few (1-5) | Many (close to K) |

---

### Visualization 6: Client-Cluster Heatmap (`viz6_client_cluster_heatmap.png`) ⭐ KEY VISUALIZATION

**What it shows:** Proportion of each client's data that comes from each cluster.

**How to read it:**
- Rows = Clients (0 to N-1)
- Columns = Clusters (0 to K-1)
- Cell color = Proportion (dark = high, light = low)
- Each row sums to 1.0

**What to look for:**
| Pattern | Meaning |
|---------|---------|
| Striped rows (few dark cells) | Client specializes in few scene types → **Non-IID** |
| Uniform rows (all similar colors) | Client has diverse data → **IID-like** |
| Block patterns | Groups of clients share similar distributions |

**Example interpretations:**
```
Non-IID (α=0.1):
Client 0: [0.8, 0.2, 0.0, 0.0, ...] → 80% from cluster 0, specialized
Client 1: [0.0, 0.0, 0.9, 0.1, ...] → 90% from cluster 2, specialized

IID (α=100):
Client 0: [0.06, 0.07, 0.05, 0.06, ...] → ~6% from each, diverse
Client 1: [0.07, 0.06, 0.06, 0.05, ...] → ~6% from each, diverse
```

---

### Visualization 7: Client Samples (`viz7_client_samples.png`)

**What it shows:** Actual images assigned to each client.

**How to read it:**
- Each row = one client's data
- With low α: images in same row should look similar
- With high α: images in same row should look diverse

---

## Section 4: Alpha Comparison

### Visualization 8: Alpha Comparison Heatmaps (`viz8_alpha_comparison.png`) ⭐ KEY VISUALIZATION

**What it shows:** Side-by-side client-cluster heatmaps for different α values.

**How to read it:**
- Left to right: α = 0.1, 0.5, 1.0, 10.0, 100.0
- Same color scale across all
- Compare row patterns across α values

**What to look for:**
| α = 0.1 (leftmost) | α = 100 (rightmost) |
|-------------------|---------------------|
| Sparse rows (few dark cells) | Dense rows (many colored cells) |
| High contrast | Low contrast |
| Clients are specialized | Clients are similar |

---

### Visualization 9: Heterogeneity Metrics (`viz9_heterogeneity_metrics.png`)

**What it shows:** Quantitative metrics as a function of α.

**Left plot - Coefficient of Variation (CV):**
- CV = std(samples) / mean(samples)
- Higher CV = more unequal sample sizes
- Should decrease as α increases

**Right plot - Avg Clusters per Client:**
- Average number of unique clusters each client has
- Should increase toward K as α increases

**Expected curves:**
```
CV:                    Avg Clusters:
  │                      │
  │\                     │      ___
  │ \                    │    /
  │  \__                 │   /
  │     \_____           │  /
  └─────────── α         └─────────── α
   0.1    100             0.1    100
```

---

### Visualization 10: Embedding Space by Client (`viz10_embedding_by_client.png`)

**What it shows:** t-SNE colored by which client owns each sample.

**How to read it:**
- Three panels: α = 0.1, 1.0, 100.0
- Each color = one client
- Look at color mixing patterns

**What to look for:**
| α = 0.1 | α = 100 |
|---------|---------|
| Colors form spatial clusters | Colors are mixed everywhere |
| Each client "owns" a region | No spatial ownership |
| Clear boundaries | No boundaries |

---

## Summary: Choosing α for Your Experiments

| α Value | Heterogeneity Level | Use Case |
|---------|---------------------|----------|
| 0.1 | Extreme non-IID | Stress-test FL algorithms |
| 0.5 | High non-IID | Realistic FL simulation |
| 1.0 | Moderate non-IID | Balanced heterogeneity |
| 10.0 | Low non-IID | Mild distribution shift |
| 100.0 | Near IID | Baseline/upper bound |

**Recommendation:** Start with α = 0.5 for realistic FL experiments, then compare with α = 0.1 (extreme) and α = 10.0 (mild) to study robustness.

---

## Output Files Reference

| File | Contents | Size |
|------|----------|------|
| `embeddings.pth` | DINOv2 CLS embeddings [N, 1024] | ~12 MB |
| `clusters.pth` | K-Means labels and centroids | ~100 KB |
| `client_splits.pth` | Default partition (α from command) | ~50 KB |
| `client_splits_alpha_X.pth` | Partition with specific α | ~50 KB each |
| `viz*.png` | All visualizations | ~500 KB each |
