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

// draw accuracy graph data export interface
interface TableObj {
    key: number;
    sequenceId: number;
    id: string;
    duration: number;
    status: string;
    acc?: FinalType; // draw accuracy graph
    description: Parameters;
    color?: string;
    startTime?: number;
    endTime?: number;
    intermediates: (MetricDataRecord | undefined)[];
    parameters(axes: MultipleAxes): Map<SingleAxis, any>;
    metrics(axes: MultipleAxes): Map<SingleAxis, any>;
}

interface TableRecord {
    key: string;
    sequenceId: number;
    startTime: number;
    endTime?: number;
    id: string;
    duration: number;
    status: string;
    message: string;
    intermediateCount: number;
    accuracy?: number | any;
    latestAccuracy: number | undefined;
    formattedLatestAccuracy: string; // format (LATEST/FINAL),
    accDictionary: FinalType | undefined;
}

interface SearchSpace {
    _value: Array<number | string>;
    _type: string;
}

interface FinalType {
    default: string;
}

interface ErrorParameter {
    error?: string;
}

interface Parameters {
    parameters: ErrorParameter;
    logPath?: string;
    intermediate: number[];
    multiProgress?: number;
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

interface Intermedia {
    name: string; // id
    type: string;
    data: Array<number | object>; // intermediate data
    hyperPara: object; // each trial hyperpara value
    trialNum: number;
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
    stderrPath?: string;
}

interface ClusterItem {
    command?: string;
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
}

interface Tensorboard {
    id: string;
    status: string;
    trialJobIdList: string[];
    trialLogDirectoryList: string[];
    pid: number;
    port: string;
}

export {
    TableObj,
    TableRecord,
    SearchSpace,
    FinalType,
    ErrorParameter,
    Parameters,
    AccurPoint,
    DetailAccurPoint,
    TooltipForIntermediate,
    TooltipForAccuracy,
    Dimobj,
    ParaObj,
    Intermedia,
    MetricDataRecord,
    TrialJobInfo,
    ExperimentProfile,
    NNIManagerStatus,
    EventMap,
    SingleAxis,
    MultipleAxes,
    SortInfo,
    AllExperimentList,
    Tensorboard
};
