// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

/**
 *  TODO
 *  This type is planned to be an interface.
 *  To minimize the change, it is currently identical to TrialJobMetric.data
 **/
export type Metric = string;

/**
 *  TODO
 *  This type is planned to be an interface.
 *  To minimize the change, it is currently identical to HyperParameters.value
 **/
export type Parameter = string;

export interface EnvironmentInfo {
    id: string;
}

export interface TrainingServiceV3 {
    init(): Promise<void>;
    start(): Promise<EnvironmentInfo[]>;
    stop(): Promise<void>;

    uploadDirectory(directoryName: string, path: string): Promise<void>;
    createTrial(environmentId: string, trialCommand: string, directoryName: string): Promise<string | null>;
    stopTrial(trialId: string): Promise<void>;
    sendParameter(trialId: string, parameter: Parameter): Promise<void>;

    onRequestParameter(callback: (trialId: string) => Promise<void>): void;
    onMetric(callback: (trialId: string, metric: Metric) => Promise<void>): void;
    onTrialEnd(callback: (trialId: string, timestamp: number, exitCode: number | null) => Promise<void>): void;
    onTrialStart(callback: (trialId: string, timestamp: number) => Promise<void>): void;
    //onTrialUpdate(callback: (trialId: string, info: any) => Promise<void>): void;
    onEnvironmentUpdate(callback: (environments: EnvironmentInfo[]) => Promise<void>): void;
}
