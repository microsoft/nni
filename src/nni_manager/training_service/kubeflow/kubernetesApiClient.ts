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

import * as fs from 'fs';
import * as os from 'os'
import * as path from 'path';
import { getLogger, Logger } from '../../common/log';

var K8SClient = require('kubernetes-client').Client;
var K8SConfig = require('kubernetes-client').config;

abstract class KubeflowOperatorClient {
    protected readonly client: any;
    protected readonly log: Logger = getLogger();
    protected crdSchema: any;

    constructor() {
        this.client = new K8SClient({ config: K8SConfig.fromKubeconfig(path.join(os.homedir(), '.kube', 'config'))});
        this.client.loadSpec();
    }

    protected abstract get operator(): any;

    public abstract get containerName(): string;

    public get jobKind(): string {
        if(this.crdSchema 
            && this.crdSchema.spec 
            && this.crdSchema.spec.names
            && this.crdSchema.spec.names.kind) {
            return this.crdSchema.spec.names.kind;
        } else {
            throw new Error('KubeflowOperatorClient: getJobKind failed, kind is undefined in tfjob crd schema!');
        }
    }
    
    public async createKubeflowJob(jobManifest: any): Promise<boolean> {
        let result: Promise<boolean>;
        const response : any = await this.operator.post({body: jobManifest});
        if(response.statusCode && (response.statusCode >= 200 && response.statusCode <= 299)) {
            result = Promise.resolve(true);
        } else {
            result = Promise.reject(`TFOperatorClient create tfjobs failed, statusCode is ${response.statusCode}`);
        }
        return result;
    }

    //TODO : replace any
    public async getKubeflowJob(kubeflowJobName: string): Promise<any> {
        let result: Promise<any>;
        const response : any = await this.operator(kubeflowJobName).get();
        if(response.statusCode && (response.statusCode >= 200 && response.statusCode <= 299)) {
            console.log(`TFOperatorClient: tfjobs conditions is ${JSON.stringify(response.body.status.conditions)}`);
            result = Promise.resolve(response.body);
        } else {
            result = Promise.reject(`TFOperatorClient get tfjobs failed, statusCode is ${response.statusCode}`);
        }
        return result;
    }

    public async deleteKubeflowJob(labels: Map<string, string>): Promise<boolean> {
        let result: Promise<boolean>;
        // construct match query from labels for deleting tfjob
        const matchQuery: string = Array.from(labels.keys()).map(labelKey => `${labelKey}=${labels.get(labelKey)}`).join(',');
        try {
            const deleteResult : any = await this.operator().delete({ qs: { labelSelector: matchQuery } });
            if(deleteResult.statusCode && deleteResult.statusCode >= 200 && deleteResult.statusCode <= 299) {
                result = Promise.resolve(true);
            } else {
                result = Promise.reject(`TFOperatorClient, delete labels ${matchQuery} get wrong statusCode ${deleteResult.statusCode}`);
            }
        } catch(err) {
            result = Promise.reject(err);
        }

        return result;
    }
}

class TFOperatorClient extends KubeflowOperatorClient {
    /**
     * constructor, to initialize tfjob CRD definition
     */
    public constructor() {
        super();
        this.crdSchema = JSON.parse(fs.readFileSync('./config/tfjob-crd-v1alpha2.json', 'utf8'));
        this.client.addCustomResourceDefinition(this.crdSchema);
    }

    protected get operator(): any {
        return this.client.apis["kubeflow.org"].v1alpha2.namespaces('default').tfjobs;
    }

    public get containerName(): string {
        return 'tensorflow';
    }    
}

class PytorchOperatorClient extends KubeflowOperatorClient {
    /**
     * constructor, to initialize tfjob CRD definition
     */
    public constructor() {
        super();
        this.crdSchema = JSON.parse(fs.readFileSync('./config/pytorchjob-crd-v1alpha2.json', 'utf8'));
        this.client.addCustomResourceDefinition(this.crdSchema);
    }

    protected get operator(): any {
        return this.client.apis["kubeflow.org"].v1alpha2.namespaces('default').pytorchjobs;
    }

    public get containerName(): string {
        return 'pytorch';
    }
}

export { KubeflowOperatorClient, PytorchOperatorClient, TFOperatorClient }
