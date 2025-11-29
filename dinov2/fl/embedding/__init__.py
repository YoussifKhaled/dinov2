# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""Embedding extraction module for FL pipeline."""

from .extractor import DINOv2Extractor, extract_embeddings, load_embeddings

__all__ = ["DINOv2Extractor", "extract_embeddings", "load_embeddings"]
