// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import { PlacementConstraint } from "common/trainingService";
import { assert } from "console";
import { IMPORT_DATA, INITIALIZE, INITIALIZED, KILL_TRIAL_JOB, PING, 
    REPORT_METRIC_DATA, REQUEST_TRIAL_JOBS, 
    TERMINATE, TRIAL_END, UPDATE_SEARCH_SPACE } from "./commands";

export abstract class BaseCommand {
    public abstract toLegacyCommand(): string;
    public abstract validate(): void;
}

export class Initialize implements BaseCommand {
    searchSpace?: string;

    constructor(searchSpace?: string) {
        this.searchSpace = searchSpace;
    }

    public toLegacyCommand(): string {
        return INITIALIZE + this.searchSpace;
    }

    public validate(): void {
        throw new Error("Method not implemented.");
    }
}

export class RequestTrialJobs implements BaseCommand {
    jobNum: string;

    constructor(jobNum: number) {
        this.jobNum = String(jobNum);
    }

    public toLegacyCommand(): string {
        return REQUEST_TRIAL_JOBS + this.jobNum;
    }

    public validate(): void {
        assert(typeof this.jobNum === 'string');
    }
}

export class ReportMetricData implements BaseCommand {
    metricData: string;

    constructor(metricData: string) {
        this.metricData = metricData;
    }

    public toLegacyCommand(): string {
        return REPORT_METRIC_DATA + this.metricData;
    }

    public validate(): void {
        assert(typeof this.metricData === 'string');
    }
}

export class UpdateSearchSpace implements BaseCommand {
    searchSpace: string;

    constructor(searchSpace: string) {
        this.searchSpace = searchSpace;
    }

    public toLegacyCommand(): string {
        return UPDATE_SEARCH_SPACE + this.searchSpace;
    }

    public validate(): void {
        assert(typeof this.searchSpace === 'string');
    }
}

export class ImportData implements BaseCommand {
    data: string;

    constructor(data: string) {
        this.data = data;
    }

    public toLegacyCommand(): string {
        return IMPORT_DATA + this.data;
    }

    public validate(): void {
        assert(typeof this.data === 'string');
    }
}

export class TrialEnd implements BaseCommand {
    trialJobId: string;
    event: string;
    hyperParams: string;

    constructor(trialJobId: string, event: string, hyperParams: string) {
        this.trialJobId = trialJobId;
        this.event = event;
        this.hyperParams = hyperParams;
    }

    public toLegacyCommand(): string {
        return TRIAL_END + JSON.stringify({
            trial_job_id: this.trialJobId, // eslint-disable-line @typescript-eslint/camelcase
            event: this.event, 
            hyper_params: this.hyperParams});// eslint-disable-line @typescript-eslint/camelcase
    }

    public validate(): void {
        assert(typeof this.trialJobId === 'string');
        assert(typeof this.event === 'string');
        assert(typeof this.hyperParams === 'string');
    }
}

export class Terminate implements BaseCommand {
    public toLegacyCommand(): string {
        return TERMINATE;
    }

    public validate(): void {
        throw new Error("Method not implemented.");
    }
}

export class Ping implements BaseCommand {
    public toLegacyCommand(): string {
        return PING;
    }

    public validate(): void {
        throw new Error("Method not implemented.");
    }
}

export class Initialized implements BaseCommand {
    public toLegacyCommand(): string {
        return INITIALIZED;
    }

    public validate(): void {
        throw new Error("Method not implemented.");
    }
}

export class NewTrialJob implements BaseCommand {
    parameterId: number;
    parameters: object;
    parameterSource: string;
    parameterIndex?: number;
    placementConstraint?: PlacementConstraint;
    versionInfo?: string;

    constructor(content: NewTrialJobContent) {
        this.parameterId = content.parameter_id;
        this.parameters = content.parameters;
        this.parameterSource = content.parameter_source;
        this.parameterIndex = content.parameter_index;
        this.placementConstraint = content.placement_constraint;
        this.versionInfo = content.version_info;  
    }

    public toLegacyCommand(): string {
        throw new Error("Method not implemented.");
    }

    public validate(): void {
        throw new Error("Method not implemented.");
    }
}

export class SendTrialJobParameter implements BaseCommand {
    parameterSource: string;
    parameters: object;
    parameterIndex: number;
    trialJobId: string;
    parameterId?: number;

    constructor(content: SendTrialJobParameterContent) {
        this.trialJobId = content.trial_job_id;
        this.parameterId = content.parameter_id;
        this.parameterSource = content.parameter_source;
        this.parameters = content.parameters;
        this.parameterIndex = content.parameter_index;
    }

    public toLegacyCommand(): string {
        throw new Error("Method not implemented.");
    }

    public validate(): void {
        assert(this.parameterIndex >= 0);
        assert(this.trialJobId !== undefined);
    }
}

export class NoMoreTrialJobs implements BaseCommand {
    parameterSource?: string;
    parameters?: object;
    parameterIndex?: number;
    parameterId?: number;

    constructor(content: NoMoreTrialJobsContent) {
        this.parameterSource = content.parameter_source;
        this.parameters = content.parameters;
        this.parameterIndex = content.parameter_index;
        this.parameterId = content.parameter_id;
    }

    public toLegacyCommand(): string {
        throw new Error("Method not implemented.");
    }

    public validate(): void {
        throw new Error("Method not implemented.");
    }
}

export class KillTrialJob implements BaseCommand {
    trialJobId?: string;

    constructor(trialJobId?: string) {
        this.trialJobId = trialJobId;
    }

    public toLegacyCommand(): string {
        return KILL_TRIAL_JOB + this.trialJobId;
    }

    public validate(): void {
        throw new Error("Method not implemented.");
    }
}

export interface NoMoreTrialJobsContent {
    readonly parameter_id?: number;
    readonly parameter_source?: string;
    readonly parameters?: object;
    readonly parameter_index?: number;
}

export interface SendTrialJobParameterContent {
    readonly parameter_id: number;
    readonly parameter_source: string;
    readonly parameters: object;
    readonly parameter_index: number;
    readonly trial_job_id: string;
}

export interface NewTrialJobContent {
    readonly parameter_id: number;
    readonly parameters: object;
    readonly parameter_source: string;
    readonly parameter_index?: number;
    readonly placement_constraint?: PlacementConstraint;
    readonly version_info?: string;
}