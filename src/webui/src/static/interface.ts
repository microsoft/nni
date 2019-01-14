// draw accuracy graph data interface
interface TableObj {
    key: number;
    sequenceId: number;
    id: string;
    duration: number;
    status: string;
    acc?: number; // draw accuracy graph
    description: Parameters;
    color?: string;
}

interface TableObjFianl {
    key: number;
    sequenceId: number;
    id: string;
    duration: number;
    status: string;
    acc?: FinalType;
    description: Parameters;
    color?: string;
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
}

interface ParaObj {
    data: number[][];
    parallelAxis: Array<Dimobj>;
}

interface VisualMapValue {
    maxAccuracy: number;
    minAccuracy: number;
}

interface FinalResult {
    data: string;
}

export {
    TableObj, Parameters, Experiment, 
    AccurPoint, TrialNumber, TrialJob,
    DetailAccurPoint, TooltipForAccuracy,
    ParaObj, VisualMapValue, Dimobj, FinalResult,
    TableObjFianl, FinalType
};
