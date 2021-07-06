// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import {
  TrialJobDetail,
  TrialJobStatus,
  TrialJobApplicationForm
} from "../../common/trainingService";

export class DLTSTrialJobDetail implements TrialJobDetail {
  public startTime?: number;
  public endTime?: number;
  public tags?: string[];
  public url?: string;
  public isEarlyStopped?: boolean;

  // DLTS staff
  public dltsJobId?: string;
  public dltsPaused: boolean = false;

  public constructor (
    public id: string,
    public status: TrialJobStatus,
    public submitTime: number,
    public workingDirectory: string,
    public form: TrialJobApplicationForm,

    // DLTS staff
    public dltsJobName: string,
  ) {}
}
