// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import { TrialConfig } from "training_service/common/trialConfig";

export class DLTSTrialConfig extends TrialConfig {
  public constructor(
    command: string,
    codeDir: string,
    gpuNum: number,
    public readonly image: string
  ) {
    super(command, codeDir, gpuNum);
  }
}
