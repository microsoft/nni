// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

export interface DLTSClusterConfig {
  dashboard: string;

  cluster: string;
  team: string;

  email: string;
  password: string;

  gpuType?: string;
}
