// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import * as fs from 'fs';
import { GeneralK8sClient, KubernetesCRDClient } from '../kubernetesApiClient';

/**
 * FrameworkController Client
 */
abstract class FrameworkControllerClient extends KubernetesCRDClient {
    /**
     * Factory method to generate operator client
     */
    // tslint:disable-next-line:function-name
    public static generateFrameworkControllerClient(): KubernetesCRDClient {
        return new FrameworkControllerClientV1();
    }
}

/**
 * FrameworkController ClientV1
 */
class FrameworkControllerClientV1 extends FrameworkControllerClient {
    /**
     * constructor, to initialize frameworkcontroller CRD definition
     */
    // tslint:disable: no-unsafe-any no-any
    public constructor() {
        super();
        this.crdSchema = JSON.parse(fs.readFileSync('./config/frameworkcontroller/frameworkcontrollerjob-crd-v1.json', 'utf8'));
        this.client.addCustomResourceDefinition(this.crdSchema);
    }

    protected get operator(): any {
        return this.client.apis['frameworkcontroller.microsoft.com'].v1.namespaces('default').frameworks;
    }
    // tslint:enable: no-unsafe-any no-any

    public get containerName(): string {
        return 'framework';
    }
}

export { FrameworkControllerClient, GeneralK8sClient };
