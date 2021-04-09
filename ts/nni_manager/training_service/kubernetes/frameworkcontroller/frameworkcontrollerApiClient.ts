// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import * as fs from 'fs';
import {GeneralK8sClient, KubernetesCRDClient} from '../kubernetesApiClient';

/**
 * FrameworkController ClientV1
 */
class FrameworkControllerClientV1 extends KubernetesCRDClient {
    /**
     * constructor, to initialize frameworkcontroller CRD definition
     */
    public namespace: string;
    public constructor(namespace?: string) {
        super();
        this.namespace = namespace ? namespace : "default"
        this.crdSchema = JSON.parse(fs.readFileSync('./config/frameworkcontroller/frameworkcontrollerjob-crd-v1.json', 'utf8'));
        this.client.addCustomResourceDefinition(this.crdSchema);
    }

    protected get operator(): any {
        return this.client.apis['frameworkcontroller.microsoft.com'].v1.namespaces(this.namespace).frameworks;
    }

    public get containerName(): string {
        return 'framework';
    }
}

/**
 * FrameworkController Client
 */
class FrameworkControllerClientFactory {
    /**
     * Factory method to generate operator client
     */
    public static createClient(namespace?: string): KubernetesCRDClient {
        return new FrameworkControllerClientV1(namespace);
    }
}

export {FrameworkControllerClientFactory, GeneralK8sClient};
