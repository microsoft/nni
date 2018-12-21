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
import { KubernetesCRDClient, GeneralK8sClient } from '../kubernetesApiClient';

abstract class FrameworkControllerClient extends KubernetesCRDClient{
    /**
     * Factory method to generate operator cliet
     */
    public static generateFrameworkControllerClient(): KubernetesCRDClient {
        return new FrameworkControllerClientV1();
    }
}

class FrameworkControllerClientV1 extends FrameworkControllerClient {
    /**
     * constructor, to initialize frameworkcontroller CRD definition
     */
    public constructor() {
        super();
        this.crdSchema = JSON.parse(fs.readFileSync('./config/frameworkcontroller/frameworkcontrollerjob-crd-v1.json', 'utf8'));
        this.client.addCustomResourceDefinition(this.crdSchema);
    }

    protected get operator(): any {
        return this.client.apis["frameworkcontroller.microsoft.com"].v1.namespaces('default').frameworks;
    }

    public get containerName(): string {
        return 'framework';
    }    
}

export { FrameworkControllerClient, GeneralK8sClient };

