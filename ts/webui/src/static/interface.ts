// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import { ExperimentConfig } from './experimentConfig';

/**
 * Definition of single dimension in search space.
 */
interface SingleAxis {
    baseName: string;
    fullName: string;
    type: string;
    scale: 'log' | 'linear' | 'ordinal';
    domain: any;
    nested: boolean;
}

/**
 * Definition of combination of multiple dimensions.
 * The decision in multiple dimensions will be combined together.
 * Typically, it is a search space or a sub search space.
 */
interface MultipleAxes {
    baseName: string;
    fullName: string;
    axes: Map<string, SingleAxis>;
}

interface TableRecord {
    _key: string;
    sequenceId: number;
    startTime: number;
    endTime?: number;
    id: string;
    duration: number;
    status: string;
    message: string;
    intermediateCount: number;
    latestAccuracy: number | undefined;
    _formattedLatestAccuracy: string; // format (LATEST/FINAL),
}

interface SearchSpace {
    _value: Array<number | string>;
    _type: string;
}

interface FinalType {
    default: string;
}

// trial accuracy
interface AccurPoint {
    acc: number;
    index: number;
}

interface DetailAccurPoint {
    acc: number;
    index: number;
    searchSpace: object;
}

interface TooltipForIntermediate {
    data: string;
    seriesName: string;
    dataIndex: number;
}

interface TooltipForAccuracy {
    data: Array<number | object>;
}

interface Dimobj {
    dim: number;
    name: string;
    max?: number;
    min?: number;
    type?: string;
    data?: string[];
    boundaryGap?: boolean;
    axisTick?: object;
    axisLabel?: object;
    axisLine?: object;
    nameTextStyle?: object;
    scale?: boolean;
}

interface ParaObj {
    data: number[][];
    parallelAxis: Array<Dimobj>;
}

interface MetricDataRecord {
    timestamp: number;
    trialJobId: string;
    type: string;
    sequence: number;
    data: string;
}

interface TrialJobInfo {
    trialJobId: string;
    sequenceId: number;
    status: string;
    message: string;
    startTime?: number;
    endTime?: number;
    hyperParameters?: string[];
    logPath?: string;
    finalMetricData?: MetricDataRecord[];
}

interface ExperimentProfile {
    params: ExperimentConfig;
    id: string;
    execDuration: number;
    logDir: string;
    startTime: number;
    endTime?: number;
    maxSequenceId: number;
    revision: number;
}

interface ExperimentMetadata {
    id: string;
    port: number;
    startTime: number | string;
    endTime: number | string;
    status: string;
    platform: string;
    experimentName: string;
    tag: any[];
    pid: number;
    webuiUrl: any[];
    logDir: string;
    prefixUrl: string | null;
}

interface NNIManagerStatus {
    status: string;
    errors: string[];
}

interface EventMap {
    [key: string]: () => void;
}

// table column sort
interface SortInfo {
    field: string;
    isDescend?: boolean;
}

interface AllExperimentList {
    id: string;
    experimentName: string;
    port: number;
    status: string;
    platform: string;
    startTime: number;
    endTime: number;
    tag: string[];
    pid: number;
    webuiUrl: string[];
    logDir: string[];
    prefixUrl: string;
}

interface KillJobIsError {
    isError: boolean;
    message: string;
}

type TensorboardTaskStatus = 'RUNNING' | 'DOWNLOADING_DATA' | 'STOPPING' | 'STOPPED' | 'ERROR' | 'FAIL_DOWNLOAD_DATA';

interface TensorboardTaskInfo {
    id: string;
    status: TensorboardTaskStatus;
    trialJobIdList: string[];
    trialLogDirectoryList: string[];
    pid?: number;
    port?: string;
}

// for TableList search
interface SearchItems {
    name: string;
    operator: string;
    value1: string; // first input value
    value2: string; // second input value
    choice: string[]; // use select multiy value list
    isChoice: boolean; // for parameters: type = choice and status also as choice type
}

interface AllTrialsIntermediateChart {
    name: string;
    // id: string;
    sequenceId: number;
    data: number[];
    parameter: object;
    type: string;
}

export {
    TableRecord,
    SearchSpace,
    FinalType,
    AccurPoint,
    DetailAccurPoint,
    TooltipForIntermediate,
    TooltipForAccuracy,
    Dimobj,
    ParaObj,
    MetricDataRecord,
    TrialJobInfo,
    ExperimentProfile,
    ExperimentMetadata,
    NNIManagerStatus,
    EventMap,
    SingleAxis,
    MultipleAxes,
    SortInfo,
    AllExperimentList,
    TensorboardTaskInfo,
    SearchItems,
    KillJobIsError,
    AllTrialsIntermediateChart
};
