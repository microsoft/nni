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
}

interface Experiment {
    id: string;
    author: string;
    revision?: number;
    experName: string;
    logDir?: string;
    runConcurren: number;
    maxDuration: number;
    execDuration: number;
    MaxTrialNum: number;
    startTime: number;
    endTime?: number;
    trainingServicePlatform: string;
    tuner: object;
    assessor?: object;
    advisor?: object;
    clusterMetaData?: object;
    logCollection?: string;
}

// trial accuracy
interface AccurPoint {
    acc: number;
    index: number;
}

interface DetailAccurPoint {
    acc: number;
    index: number;
    searchSpace: string;
}

interface TooltipForIntermediate {
    data: string;
    seriesName: string;
    dataIndex: number;
}

interface TooltipForAccuracy {
    data: Array<number | object>;
}

interface TrialNumber {
    succTrial: number;
    failTrial: number;
    stopTrial: number;
    waitTrial: number;
    runTrial: number;
    unknowTrial: number;
    totalCurrentTrial: number;
}

interface TrialJob {
    text: string;
    value: string;
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
}

interface ParaObj {
    data: number[][];
    parallelAxis: Array<Dimobj>;
}

interface FinalResult {
    data: string;
}

export {
    TableObj, Parameters, Experiment, 
    AccurPoint, TrialNumber, TrialJob,
    DetailAccurPoint, TooltipForAccuracy,
    ParaObj, Dimobj, FinalResult, FinalType,
    TooltipForIntermediate, SearchSpace
};
