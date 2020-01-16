// draw accuracy graph data export interface
export interface TableObj {
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
}

export interface TableRecord {
    key: string;
    sequenceId: number;
    startTime: number;
    endTime?: number;
    id: string;
    duration: number;
    status: string;
    intermediateCount: number;
    accuracy?: number;
    latestAccuracy: number | undefined;
    formattedLatestAccuracy: string; // format (LATEST/FINAL)
}

export interface SearchSpace {
    _value: Array<number | string>;
    _type: string;
}

export interface FinalType {
    default: string;
}

export interface ErrorParameter {
    error?: string;
}

export interface Parameters {
    parameters: ErrorParameter;
    logPath?: string;
    intermediate: Array<number>;
    multiProgress?: number;
}

// trial accuracy
export interface AccurPoint {
    acc: number;
    index: number;
}

export interface DetailAccurPoint {
    acc: number;
    index: number;
    searchSpace: object;
}

export interface TooltipForIntermediate {
    data: string;
    seriesName: string;
    dataIndex: number;
}

export interface TooltipForAccuracy {
    data: Array<number | object>;
}

export interface Dimobj {
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
}

export interface ParaObj {
    data: number[][];
    parallelAxis: Array<Dimobj>;
}

export interface Intermedia {
    name: string; // id
    type: string;
    data: Array<number | object>; // intermediate data
    hyperPara: object; // each trial hyperpara value
}

export interface MetricDataRecord {
    timestamp: number;
    trialJobId: string;
    parameterId: string;
    type: string;
    sequence: number;
    data: string;
}

export interface TrialJobInfo {
    id: string;
    sequenceId: number;
    status: string;
    startTime?: number;
    endTime?: number;
    hyperParameters?: string[];
    logPath?: string;
    finalMetricData?: MetricDataRecord[];
    stderrPath?: string;
}

export interface ExperimentParams {
    authorName: string;
    experimentName: string;
    description?: string;
    trialConcurrency: number;
    maxExecDuration: number; // seconds
    maxTrialNum: number;
    searchSpace: string;
    trainingServicePlatform: string;
    multiPhase?: boolean;
    multiThread?: boolean;
    versionCheck?: boolean;
    logCollection?: string;
    tuner?: {
        className: string;
        builtinTunerName?: string;
        codeDir?: string;
        classArgs?: any;
        classFileName?: string;
        checkpointDir: string;
        gpuNum?: number;
        includeIntermediateResults?: boolean;
    };
    assessor?: {
        className: string;
        builtinAssessorName?: string;
        codeDir?: string;
        classArgs?: any;
        classFileName?: string;
        checkpointDir: string;
        gpuNum?: number;
    };
    advisor?: {
        className: string;
        builtinAdvisorName?: string;
        codeDir?: string;
        classArgs?: any;
        classFileName?: string;
        checkpointDir: string;
        gpuNum?: number;
    };
    clusterMetaData?: {
        key: string;
        value: string;
    }[];
}

export interface ExperimentProfile {
    params: ExperimentParams;
    id: string;
    execDuration: number;
    logDir?: string;
    startTime?: number;
    endTime?: number;
    maxSequenceId: number;
    revision: number;
}

export interface NNIManagerStatus {
    status: string;
    errors: string[];
}