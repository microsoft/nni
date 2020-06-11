// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

// eslint-disable-next-line @typescript-eslint/camelcase
import { Client1_10, config } from 'kubernetes-client';
import { getLogger, Logger } from '../../common/log';

/**
 * Generict Kubernetes client, target version >= 1.9
 */
class GeneralK8sClient {
    protected readonly client: any;
    protected readonly log: Logger = getLogger();

    constructor() {
        this.client = new Client1_10({ config: config.fromKubeconfig(), version: '1.9'});
        this.client.loadSpec();
    }

    public async createSecret(secretManifest: any): Promise<boolean> {
        let result: Promise<boolean>;
        const response: any = await this.client.api.v1.namespaces('default').secrets
          .post({body: secretManifest});
        if (response.statusCode && (response.statusCode >= 200 && response.statusCode <= 299)) {
            result = Promise.resolve(true);
        } else {
            result = Promise.reject(`Create secrets failed, statusCode is ${response.statusCode}`);
        }

        return result;
    }
}

/**
 * Kubernetes CRD client
 */
abstract class KubernetesCRDClient {
    protected readonly client: any;
    protected readonly log: Logger = getLogger();
    protected crdSchema: any;

    constructor() {
        this.client = new Client1_10({ config: config.fromKubeconfig() });
        this.client.loadSpec();
    }

    protected abstract get operator(): any;

    public abstract get containerName(): string;

    public get jobKind(): string {
        if (this.crdSchema
            && this.crdSchema.spec
            && this.crdSchema.spec.names
            && this.crdSchema.spec.names.kind) {
            return this.crdSchema.spec.names.kind;
        } else {
            throw new Error('KubeflowOperatorClient: getJobKind failed, kind is undefined in crd schema!');
        }
    }

    public get apiVersion(): string {
        if (this.crdSchema
            && this.crdSchema.spec
            && this.crdSchema.spec.version) {
            return this.crdSchema.spec.version;
        } else {
            throw new Error('KubeflowOperatorClient: get apiVersion failed, version is undefined in crd schema!');
        }
    }

    public async createKubernetesJob(jobManifest: any): Promise<boolean> {
        let result: Promise<boolean>;
        const response: any = await this.operator.post({body: jobManifest});
        if (response.statusCode && (response.statusCode >= 200 && response.statusCode <= 299)) {
            result = Promise.resolve(true);
        } else {
            result = Promise.reject(`Create kubernetes job failed, statusCode is ${response.statusCode}`);
        }

        return result;
    }

    //TODO : replace any
    public async getKubernetesJob(kubeflowJobName: string): Promise<any> {
        let result: Promise<any>;
        const response: any = await this.operator(kubeflowJobName)
          .get();
        if (response.statusCode && (response.statusCode >= 200 && response.statusCode <= 299)) {
            result = Promise.resolve(response.body);
        } else {
            result = Promise.reject(`KubeflowOperatorClient get tfjobs failed, statusCode is ${response.statusCode}`);
        }

        return result;
    }

    public async deleteKubernetesJob(labels: Map<string, string>): Promise<boolean> {
        let result: Promise<boolean>;
        // construct match query from labels for deleting tfjob
        const matchQuery: string = Array.from(labels.keys())
                                     .map((labelKey: string) => `${labelKey}=${labels.get(labelKey)}`)
                                     .join(',');
        try {
            const deleteResult: any = await this.operator()
              .delete({
                 qs: {
                      labelSelector: matchQuery,
                      propagationPolicy: 'Background'
                     }
            });
            if (deleteResult.statusCode && deleteResult.statusCode >= 200 && deleteResult.statusCode <= 299) {
                result = Promise.resolve(true);
            } else {
                result = Promise.reject(
                    `KubeflowOperatorClient, delete labels ${matchQuery} get wrong statusCode ${deleteResult.statusCode}`);
            }
        } catch (err) {
            result = Promise.reject(err);
        }

        return result;
    }
}

export { KubernetesCRDClient, GeneralK8sClient };
