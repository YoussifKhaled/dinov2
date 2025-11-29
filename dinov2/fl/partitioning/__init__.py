# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""Partitioning module for FL pipeline."""

from .dirichlet import partition_data, load_client_splits, DirichletPartitioner

__all__ = ["partition_data", "load_client_splits", "DirichletPartitioner"]
