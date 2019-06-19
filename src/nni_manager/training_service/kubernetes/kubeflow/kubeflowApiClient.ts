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
import { GeneralK8sClient, KubernetesCRDClient } from '../kubernetesApiClient';
import { KubeflowOperator } from './kubeflowConfig';

abstract class KubeflowOperatorClient extends KubernetesCRDClient {
    /**
     * Factory method to generate operator client
     */
    // tslint:disable-next-line:function-name
    public static generateOperatorClient(kubeflowOperator: KubeflowOperator,
                                         operatorApiVersion: string): KubernetesCRDClient {
        switch (kubeflowOperator) {
            case 'tf-operator': {
                switch (operatorApiVersion) {
                    case 'v1alpha2': {
                        return new TFOperatorClientV1Alpha2();
                    }
                    case 'v1beta1': {
                        return new TFOperatorClientV1Beta1();
                    }
                    case 'v1beta2': {
                        return new TFOperatorClientV1Beta2();
                    }
                    default:
                        throw new Error(`Invalid tf-operator apiVersion ${operatorApiVersion}`);
                }
            }
            case 'pytorch-operator': {
                switch (operatorApiVersion) {
                    case 'v1alpha2': {
                        return new PyTorchOperatorClientV1Alpha2();
                    }
                    case 'v1beta1': {
                        return new PyTorchOperatorClientV1Beta1();
                    }
                    case 'v1beta2': {
                        return new PyTorchOperatorClientV1Beta2();
                    }
                    default:
                        throw new Error(`Invalid pytorch-operator apiVersion ${operatorApiVersion}`);
                }
            }
            default:
                throw new Error(`Invalid operator ${kubeflowOperator}`);
        }
    }
}

// tslint:disable: no-unsafe-any no-any
class TFOperatorClientV1Alpha2 extends KubeflowOperatorClient {
    /**
     * constructor, to initialize tfjob CRD definition
     */
    public constructor() {
        super();
        this.crdSchema = JSON.parse(fs.readFileSync('./config/kubeflow/tfjob-crd-v1alpha2.json', 'utf8'));
        this.client.addCustomResourceDefinition(this.crdSchema);
    }

    protected get operator(): any {
        return this.client.apis['kubeflow.org'].v1alpha2.namespaces('default').tfjobs;
    }

    public get containerName(): string {
        return 'tensorflow';
    }
}

class TFOperatorClientV1Beta1 extends KubernetesCRDClient {
    /**
     * constructor, to initialize tfjob CRD definition
     */
    public constructor() {
        super();
        this.crdSchema = JSON.parse(fs.readFileSync('./config/kubeflow/tfjob-crd-v1beta1.json', 'utf8'));
        this.client.addCustomResourceDefinition(this.crdSchema);
    }

    protected get operator(): any {
        return this.client.apis['kubeflow.org'].v1beta1.namespaces('default').tfjobs;
    }

    public get containerName(): string {
        return 'tensorflow';
    }
}

class TFOperatorClientV1Beta2 extends KubernetesCRDClient {
    /**
     * constructor, to initialize tfjob CRD definition
     */
    public constructor() {
        super();
        this.crdSchema = JSON.parse(fs.readFileSync('./config/kubeflow/tfjob-crd-v1beta2.json', 'utf8'));
        this.client.addCustomResourceDefinition(this.crdSchema);
    }

    protected get operator(): any {
        return this.client.apis['kubeflow.org'].v1beta2.namespaces('default').tfjobs;
    }

    public get containerName(): string {
        return 'tensorflow';
    }
}

class PyTorchOperatorClientV1Alpha2 extends KubeflowOperatorClient {
    /**
     * constructor, to initialize tfjob CRD definition
     */
    public constructor() {
        super();
        this.crdSchema = JSON.parse(fs.readFileSync('./config/kubeflow/pytorchjob-crd-v1alpha2.json', 'utf8'));
        this.client.addCustomResourceDefinition(this.crdSchema);
    }

    protected get operator(): any {
        return this.client.apis['kubeflow.org'].v1alpha2.namespaces('default').pytorchjobs;
    }

    public get containerName(): string {
        return 'pytorch';
    }
}

class PyTorchOperatorClientV1Beta1 extends KubernetesCRDClient {
    /**
     * constructor, to initialize tfjob CRD definition
     */
    public constructor() {
        super();
        this.crdSchema = JSON.parse(fs.readFileSync('./config/kubeflow/pytorchjob-crd-v1beta1.json', 'utf8'));
        this.client.addCustomResourceDefinition(this.crdSchema);
    }

    protected get operator(): any {
        return this.client.apis['kubeflow.org'].v1beta1.namespaces('default').pytorchjobs;
    }

    public get containerName(): string {
        return 'pytorch';
    }
}

class PyTorchOperatorClientV1Beta2 extends KubernetesCRDClient {
    /**
     * constructor, to initialize tfjob CRD definition
     */
    public constructor() {
        super();
        this.crdSchema = JSON.parse(fs.readFileSync('./config/kubeflow/pytorchjob-crd-v1beta2.json', 'utf8'));
        this.client.addCustomResourceDefinition(this.crdSchema);
    }

    protected get operator(): any {
        return this.client.apis['kubeflow.org'].v1beta2.namespaces('default').pytorchjobs;
    }

    public get containerName(): string {
        return 'pytorch';
    }
}

// tslint:enable: no-unsafe-any
export { KubeflowOperatorClient, GeneralK8sClient };
