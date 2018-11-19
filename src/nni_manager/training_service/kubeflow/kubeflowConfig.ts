import { TrialConfig } from "training_service/common/trialConfig";

/**
 * Copyright (c) Microsoft Corporation
 * All rights reserved.
 *
 * MIT License
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
 * documentation files (the "Software"), to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
 * to permit persons to whom the Software is furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

'use strict';


/** operator types that kubeflow supported */
export type KubeflowOperator = 'tf-operator' | 'pytorch-operator' | 'mxnet-operator' | 'caffe2-operator' | 'chainer-operator' | 'mpi-operator';
export type KubeflowOperatorPlural = 'tfjobs' | 'pytorchjobs' | 'mxjobs' | 'caffe2jobs' | 'chainerjobs' | 'mpijobs';

/**
 * map from Kubeflow operator name to its plural name in K8S
 */
export const kubeflowOperatorMap : Map<KubeflowOperator, KubeflowOperatorPlural> =  new Map<KubeflowOperator, KubeflowOperatorPlural>([
    ['tf-operator' , 'tfjobs'],
    ['pytorch-operator', 'pytorchjobs'],
    ['mxnet-operator', 'mxjobs'],
    ['caffe2-operator', 'caffe2jobs'],
    ['chainer-operator', 'chainerjobs'],
    ['mpi-operator', 'mpijobs']    
]);

/**
 * Kuberflow cluster configuration
 * 
 */
export class KubeflowClusterConfig {
    /** Name of Kubeflow operator, like tf-operator */
    public readonly operator: KubeflowOperator;
    public readonly nfs: NFSConfig;
    public readonly kubernetesServer: string;

    /**
     * Constructor
     * @param userName User name of PAI Cluster
     * @param passWord password of PAI Cluster
     * @param host Host IP of PAI Cluster
     */
    constructor(operator: KubeflowOperator, nfs : NFSConfig, kubernetesServer : string) {
        this.operator = operator;
        this.nfs = nfs;
        this.kubernetesServer = kubernetesServer;
    }
}

/**
 * NFS configuration to store Kubeflow job related files
 */
export class NFSConfig {
    /** IP Adress of NFS server */
    public readonly server : string;
    /** exported NFS path on NFS server */
    public readonly path : string;

    constructor(server : string, path : string) {
        this.server = server;
        this.path = path;
    }
}

/**
 * Trial job configuration for Kubeflow
 */
export class KubeflowTrialConfig extends TrialConfig {
    public readonly cpuNum: number;
    public readonly memoryMB: number;
    public readonly image: string;
    
    constructor(command : string, codeDir : string, gpuNum : number, cpuNum: number, memoryMB: number, image: string) {
        super(command, codeDir, gpuNum);
        this.cpuNum = cpuNum;
        this.memoryMB = memoryMB;
        this.image = image;
    }
}