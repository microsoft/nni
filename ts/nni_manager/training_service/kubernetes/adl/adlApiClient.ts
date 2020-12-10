// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import * as fs from 'fs';
import { GeneralK8sClient, KubernetesCRDClient } from '../kubernetesApiClient';

/**
 * Adl ClientV1
 */
class AdlClientV1 extends KubernetesCRDClient {
    /**
     * constructor, to initialize adl CRD definition
     */
    protected readonly namespace: string;

    public constructor(namespace: string) {
        super();
        this.namespace = namespace;
        this.crdSchema = JSON.parse(fs.readFileSync('./config/adl/adaptdl-crd-v1.json', 'utf8'));
        this.client.addCustomResourceDefinition(this.crdSchema);
    }

    protected get operator(): any {
        return this.client.apis['adaptdl.petuum.com'].v1.namespaces(this.namespace).adaptdljobs;
    }

    public get containerName(): string {
        return 'main';
    }

    public async getKubernetesPods(jobName: string): Promise<any> {
        let result: Promise<any>;
        const response = await this.client.api.v1.namespaces(this.namespace).pods
            .get({ qs: { labelSelector: `adaptdl/job=${jobName}` } });
        if (response.statusCode && (response.statusCode >= 200 && response.statusCode <= 299)) {
            result = Promise.resolve(response.body);
        } else {
            result = Promise.reject(`AdlClient getKubernetesPods failed, statusCode is ${response.statusCode}`);
        }
        return result;
    }
}

/**
 * Adl Client
 */
class AdlClientFactory {
    /**
     * Factory method to generate operator client
     */
    public static createClient(namespace: string): KubernetesCRDClient {
        return new AdlClientV1(namespace);
    }
}

export { AdlClientFactory, GeneralK8sClient };
export { AdlClientV1 }
