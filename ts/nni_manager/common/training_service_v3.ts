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

    [key: string]: any;
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


    /* Following methods are guaranteed to be invoked after start() and before stop(). */

    /**
     *  "Upload" a directory and assign a name.
     *  The name is guaranteed not to contain special characters.
     **/
    uploadDirectory(directoryName: string, path: string): Promise<void>;

    /**
     *  Create a trial process in specified environment, using the uploaded directory as PWD.
     *
     *  Return trial ID on success.
     *  Return null if the environment is not available.
     **/
    createTrial(
        environmentId: string,
        trialCommand: string,
        directoryName: string,
        sequenceId: number,  // TODO: move to global counter
        trialId?: string,  // FIXME: temporary solution for resuming trial
    ): Promise<string | null>;

    /**
     *  Kill a trial.
     *  The trial ID is guaranteed to exist, but the trial is not guaranteed to be running.
     **/
    stopTrial(trialId: string): Promise<void>;

    // TODO: resume trial

    /**
     *  Send a hyperparameter configuration to a trial.
     *  Will only be invoked after onRequestParameter().
     **/
    sendParameter(trialId: string, parameter: Parameter): Promise<void>;

    /* Following methods are guaranteed to be invoked after init() and before start(). */

    /**
     *  Invoke the callback when a trial invokes nni.get_next_parameter().
     **/
    onRequestParameter(callback: (trialId: string) => Promise<void>): void;

    /**
     *  Invoke the callback when a trial invokes nni.report_final_result() and nni.report_intermediate_result().
     **/
    onMetric(callback: (trialId: string, metric: Metric) => Promise<void>): void;

    /**
     *  Invoke the callback when a trial process is launched.
     *
     *  If there are multiple listeners, `timestamp` should be consistent.
     *
     *  If the training platform automatically retries failed jobs, the callback should only be invoked on first start.
     **/
    onTrialStart(callback: (trialId: string, timestamp: number) => Promise<void>): void;

    /**
     *  Invoke the callback when a trial stops.
     *
     *  If the trial stops on its own, provide the exit code.
     *  If the trial is killed for any reason, set `exitCode` to null.
     *
     *  If there are multiple listeners, `timestamp` should be consistent.
     *
     *  If the training platform automatically retries failed jobs, the callback should only be invoked on last end.
     **/
    onTrialEnd(callback: (trialId: string, timestamp: number, exitCode: number | null) => Promise<void>): void;

    /**
     *  Invoke the callback when any environment's status changes.
     *
     *  Note that `environments` object should be immutable.
     **/
    onEnvironmentUpdate(callback: (environments: EnvironmentInfo[]) => Promise<void>): void;

    // TODO: temporary api
    downloadTrialDirectory(trialId: string): Promise<string>;
}
