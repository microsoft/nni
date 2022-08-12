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
    // TODO
}

export interface TrainingServiceV3 {
    /**
     *  Invoked during experiment initialization.
     *
     *  It should verify the config and raise error if the specified training resource is not available.
     *
     *  It should not start daemon on worker machine.
     *  If another component of the experiment failed to initialize,
     *  the process will abort without invoking any clean up function.
     **/
    init(): Promise<void>;

    /**
     *  Invoked after init().
     *
     *  It is suggested to resolve the promise after all daemons initialized.
     **/
    start(): Promise<EnvironmentInfo[]>;

    /**
     *  Invoked after start().
     *
     *  Currently the max clean up time is hard coded to 60s.
     *  If the returned promise does not resolve in time, the process will abort.
     **/
    stop(): Promise<void>;


    /** Following methods are guaranteed to be invoked after `start()` and before `stop()`. **/

    /**
     *  "Upload" a directory and assign a name.
     *  The name is guaranteed not to contain special characters.
     **/
    uploadDirectory(directoryName: string, path: string): Promise<void>;

    /**
     *  Create a trial process in specified environment, using the uploaded directory as PWD.
     **/
    createTrial(environmentId: string, trialCommand: string, directoryName: string): Promise<string | null>;

    /**
     *  Kill a trial.
     *  The trial ID is guaranteed to exist, but the trial is not guaranteed to be running.
     **/
    stopTrial(trialId: string): Promise<void>;

    sendParameter(trialId: string, parameter: Parameter): Promise<void>;

    /** Following methods are guaranteed to be invoked after `init()` and before `start()`. **/

    onRequestParameter(callback: (trialId: string) => Promise<void>): void;
    onMetric(callback: (trialId: string, metric: Metric) => Promise<void>): void;
    onTrialStart(callback: (trialId: string, timestamp: number) => Promise<void>): void;
    onTrialEnd(callback: (trialId: string, timestamp: number, exitCode: number | null) => Promise<void>): void;
    onEnvironmentUpdate(callback: (environments: EnvironmentInfo[]) => Promise<void>): void;
}
