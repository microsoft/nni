// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import { DLTSClusterConfig } from "./dltsClusterConfig";

export class DLTSJobConfig {
  public readonly team: string;
  public readonly userName: string;
  public readonly vcName: string;
  public readonly gpuType: string;
  public readonly jobType = "training";
  public readonly jobtrainingtype = "RegularJob";
  public readonly ssh = false;
  public readonly ipython = false;
  public readonly tensorboard = false;
  public readonly workPath = '';
  public readonly enableworkpath = true;
  public readonly dataPath = '';
  public readonly enabledatapath = false;
  public readonly jobPath = '';
  public readonly enablejobpath = true;
  public readonly mountpoints = [];
  public readonly env = [{ name: 'TMPDIR', value: '$HOME/tmp' }]
  public readonly hostNetwork = false;
  public readonly useGPUTopology = false;
  public readonly isPrivileged = false;
  public readonly hostIPC = false;
  public readonly preemptionAllowed = "False"

  public constructor(
    clusterConfig: DLTSClusterConfig,
    public readonly jobName: string,
    public readonly resourcegpu: number,
    public readonly image: string,
    public readonly cmd: string,
    public readonly interactivePorts: number[],
  ) {
    if (clusterConfig.gpuType === undefined) {
      throw Error('GPU type not fetched')
    }
    this.vcName = this.team = clusterConfig.team
    this.gpuType = clusterConfig.gpuType
    this.userName = clusterConfig.email
  }
}
