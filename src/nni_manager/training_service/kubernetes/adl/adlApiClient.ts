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
    public constructor() {
        super();
        this.crdSchema = JSON.parse(fs.readFileSync('./config/adl/adaptdl-crd-v1.json', 'utf8'));
        this.client.addCustomResourceDefinition(this.crdSchema);
    }

    protected get operator(): any {
        return this.client.apis['adaptdl.petuum.com'].v1.namespaces('default').adaptdljobs;
    }

    public get containerName(): string {
        return 'main';
    }
}

/**
 * Adl Client
 */
class AdlClientFactory {
    /**
     * Factory method to generate operator client
     */
    public static createClient(): KubernetesCRDClient {
        return new AdlClientV1();
    }
}

export { AdlClientFactory, GeneralK8sClient };
