// draw accuracy graph data interface
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
}

interface TableRecord {
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
    intermediate: Array<number>;
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
}

interface MetricDataRecord {
    timestamp: number;
    trialJobId: string;
    parameterId: string;
    type: string;
    sequence: number;
    data: string;
}

interface TrialJobInfo {
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

interface ExperimentParams {
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

interface ExperimentProfile {
    params: ExperimentParams;
    id: string;
    execDuration: number;
    logDir?: string;
    startTime?: number;
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

export {
    TableObj, TableRecord, Parameters, ExperimentProfile, AccurPoint,
    DetailAccurPoint, TooltipForAccuracy, ParaObj, Dimobj, FinalType,
    TooltipForIntermediate, SearchSpace, Intermedia, MetricDataRecord, TrialJobInfo,
    NNIManagerStatus, EventMap
};
