// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

'use strict';

import fs from 'fs';
import path from 'path';
import * as component from 'common/component';
import { Logger, getLogger } from 'common/log';
import { uniqueString } from 'common/utils';
import { FrameworkControllerConfig, FrameworkControllerTaskRoleConfig, toMegaBytes } from 'common/experimentConfig';
import { ExperimentStartupInfo } from 'common/experimentStartupInfo';
import { GeneralK8sClient, KubernetesCRDClient } from 'training_service/kubernetes/kubernetesApiClient';
import { EnvironmentInformation, EnvironmentService } from './environment';
import { FrameworkControllerClientFactory } from '../kubernetes/frameworkcontroller/frameworkcontrollerApiClient';
import type { FrameworkControllerTrialConfigTemplate } from '../kubernetes/frameworkcontroller/frameworkcontrollerConfig';


export class FrameworkControllerEnvironmentService extends EnvironmentService {
    private containerMountPath: string;
    private NNI_KUBERNETES_TRIAL_LABEL: string;
    private log: Logger;
    private genericK8sClient: GeneralK8sClient;
    private experimentId: string;
    private environmentWorkingFolder: string;
    private kubernetesCRDClient?: KubernetesCRDClient;

    private config: FrameworkControllerConfig;
    private createStoragePromise?: Promise<void>;
    private readonly fcContainerPortMap: Map<string, number> = new Map<string, number>(); // store frameworkcontroller container port
    

    constructor(config: FrameworkControllerConfig, info: ExperimentStartupInfo) {
        super();
        this.log = getLogger('FrameworkControllerEnvironmentService');
        this.containerMountPath = '/tmp/mount';
        this.NNI_KUBERNETES_TRIAL_LABEL = 'nni-kubernetes-trial';
        this.genericK8sClient = new GeneralK8sClient();
        this.experimentId = info.experimentId;
        this.environmentWorkingFolder = path.join(this.containerMountPath, 'nni', this.experimentId);

        this.config = config;
        // Create kubernetesCRDClient
        this.kubernetesCRDClient = FrameworkControllerClientFactory.createClient(this.config.namespace);
        this.genericK8sClient.setNamespace = this.config.namespace ?? "default"
        // Storage is mounted by user
    }

    public get environmentMaintenceLoopInterval(): number {
        return 5000;
    }

    public get hasStorageService(): boolean {
        return false;
    }

    public get getName(): string {
        return 'frameworkcontroller';
    }

    public get getContainerMountPath(): string {
        return this.containerMountPath;
    }

    public generatePortAndCommand(command: string): string {
        this.generateContainerPort(this.config.taskRoles as any);
        const patchedCommand: string = this.generateCommandScript(this.config.taskRoles, command);
        return patchedCommand;
    }

    public async startEnvironment(environment: EnvironmentInformation): Promise<any> {
        if (this.kubernetesCRDClient === undefined) {
            throw new Error("kubernetesCRDClient not initialized!");
        }
        if (this.createStoragePromise) {
            await this.createStoragePromise;
        }

        const frameworkcontrollerJobName: string = `nniexp${this.experimentId}env${environment.id}`.toLowerCase();
        environment.maxTrialNumberPerGpu = this.config.maxTrialNumberPerGpu;
        // FIXME: create trial log and show it on webui
        environment.trackingUrl = `${this.config.storage.localMountPath}/nni/${this.experimentId}/envs/${environment.id}/`;
        // Generate kubeflow job resource config object
        const frameworkcontrollerJobConfig: any = await this.prepareFrameworkControllerConfig(
            environment.id,
            path.join(this.environmentWorkingFolder, 'envs', environment.id),
            frameworkcontrollerJobName
        );
        // Create frameworkcontroller job based on generated kubeflow job resource config
        await this.kubernetesCRDClient.createKubernetesJob(frameworkcontrollerJobConfig);

        return Promise.resolve(frameworkcontrollerJobConfig);
    }

    public async stopEnvironment(environment: EnvironmentInformation): Promise<void> {
        if (this.kubernetesCRDClient === undefined) {
            throw new Error('kubernetesCRDClient not initialized!');
        }
        try {
            await this.kubernetesCRDClient.deleteKubernetesJob(new Map(
                [
                    ['app', this.NNI_KUBERNETES_TRIAL_LABEL],
                    ['expId', this.experimentId],
                    ['envId', environment.id]
                ]
            ));
        } catch (err) {
            const errorMessage: string = `Delete env ${environment.id} failed: ${err}`;
            this.log.error(errorMessage);

            return Promise.reject(errorMessage);
        }
    }

    public generatePodResource(memory: number, cpuNum: number, gpuNum: number): any {
        const resources: any = {
            memory: `${memory}Mi`,
            cpu: `${cpuNum}`
        };

        if (gpuNum !== 0) {
            resources['nvidia.com/gpu'] = `${gpuNum}`;
        }

        return resources;
    }

    /**
     * generate trial's command for frameworkcontroller
     * expose port and execute injector.sh before executing user's command
     * @param command
     */
    private generateCommandScript(taskRoles: FrameworkControllerTaskRoleConfig[], command: string): string {
        let portScript: string = '';
        for (const taskRole of taskRoles) {
            portScript += `FB_${taskRole.name.toUpperCase()}_PORT=${this.fcContainerPortMap.get(
                taskRole.name
            )} `;
        }
        return `${portScript} . /mnt/frameworkbarrier/injector.sh && ${command}`;
    }
    
    private async prepareFrameworkControllerConfig(envId: string, trialWorkingFolder: string, frameworkcontrollerJobName: string):
            Promise<any> {
        const podResources: any = [];
        for (const taskRole of this.config.taskRoles) {
            const resource: any = {};
            resource.requests = this.generatePodResource(toMegaBytes(taskRole.memorySize), taskRole.cpuNumber, taskRole.gpuNumber);
            resource.limits = {...resource.requests};
            podResources.push(resource);
        }
        // Generate frameworkcontroller job resource config object
        const frameworkcontrollerJobConfig: any =
            await this.generateFrameworkControllerJobConfig(envId, trialWorkingFolder, frameworkcontrollerJobName, podResources);

        return Promise.resolve(frameworkcontrollerJobConfig);
    }
    
    private generateContainerPort(taskRoles: FrameworkControllerTrialConfigTemplate[]): void {
        if (taskRoles === undefined) {
            throw new Error('frameworkcontroller trial config is not initialized');
        }

        let port: number = 4000; //The default port used in container
        for (const index of taskRoles.keys()) {
            this.fcContainerPortMap.set(taskRoles[index].name, port);
            port += 1;
        }
    }

    private async createRegistrySecret(filePath: string | undefined): Promise<string | undefined> {
        if (filePath === undefined || filePath === '') {
            return undefined;
        }
        const body = fs.readFileSync(filePath).toString('base64');
        const registrySecretName = `nni-secret-${uniqueString(8).toLowerCase()}`;
        const namespace = this.genericK8sClient.getNamespace ?? "default";
        await this.genericK8sClient.createSecret(
            {
                apiVersion: 'v1',
                kind: 'Secret',
                metadata: {
                    name: registrySecretName,
                    namespace: namespace,
                    labels: {
                        app: this.NNI_KUBERNETES_TRIAL_LABEL,
                        expId: this.experimentId
                    }
                },
                type: 'kubernetes.io/dockerconfigjson',
                data: {
                    '.dockerconfigjson': body
                }
            }
        );
        return registrySecretName;
    }

    /**
     * Generate frameworkcontroller resource config file
     * @param trialJobId trial job id
     * @param trialWorkingFolder working folder
     * @param frameworkcontrollerJobName job name
     * @param podResources  pod template
     */
    private async generateFrameworkControllerJobConfig(envId: string, trialWorkingFolder: string,
        frameworkcontrollerJobName: string, podResources: any): Promise<any> {

        const taskRoles: any = [];
        for (const index of this.config.taskRoles.keys()) {
            const containerPort: number | undefined = this.fcContainerPortMap.get(this.config.taskRoles[index].name);
            if (containerPort === undefined) {
                throw new Error('Container port is not initialized');
            }

            const taskRole: any = this.generateTaskRoleConfig(
                trialWorkingFolder,
                this.config.taskRoles[index].dockerImage,
                `${envId}_run.sh`,
                podResources[index],
                containerPort,
                await this.createRegistrySecret(this.config.taskRoles[index].privateRegistryAuthPath)
            );
            taskRoles.push({
                name: this.config.taskRoles[index].name,
                taskNumber: this.config.taskRoles[index].taskNumber,
                frameworkAttemptCompletionPolicy: {
                    minFailedTaskCount: this.config.taskRoles[index].frameworkAttemptCompletionPolicy.minFailedTaskCount,
                    minSucceededTaskCount: this.config.taskRoles[index].frameworkAttemptCompletionPolicy.minSucceedTaskCount
                },
                task: taskRole
            });
        }

        return Promise.resolve({
            apiVersion: `frameworkcontroller.microsoft.com/v1`,
            kind: 'Framework',
            metadata: {
                name: frameworkcontrollerJobName,
                namespace: this.config.namespace ?? "default",
                labels: {
                    app: this.NNI_KUBERNETES_TRIAL_LABEL,
                    expId: this.experimentId,
                    envId: envId
                }
            },
            spec: {
                executionType: 'Start',
                taskRoles: taskRoles
            }
        });
    }

    private generateTaskRoleConfig(trialWorkingFolder: string, replicaImage: string, runScriptFile: string,
        podResources: any, containerPort: number, privateRegistrySecretName: string | undefined): any {

        const volumeSpecMap: Map<string, object> = new Map<string, object>();
        // Only support nfs for now
        volumeSpecMap.set('nniVolumes', [
            {
                name: 'nni-vol',
                nfs: {
                    server: `${this.config.storage.server}`,
                    path: `${this.config.storage.path}`
                }
            }, {
                name: 'frameworkbarrier-volume',
                emptyDir: {}
            }]);

        // const securityContext: any = {
        //     fsGroup: xxxx,
        //     runAsUser: xxxx,
        //     runAsGroup: xxxx
        // };

        const containers: any = [
            {
                name: 'framework',
                image: replicaImage,
                // securityContext: securityContext,
                command: ['sh', `${path.join(trialWorkingFolder, runScriptFile)}`],
                volumeMounts: [
                    {
                        name: 'nni-vol',
                        mountPath: this.containerMountPath
                    }, {
                        name: 'frameworkbarrier-volume',
                        mountPath: '/mnt/frameworkbarrier'
                    }],
                resources: podResources,
                ports: [{
                    containerPort: containerPort
                }]
            }];

        const initContainers: any = [
            {
                name: 'frameworkbarrier',
                image: 'frameworkcontroller/frameworkbarrier',
                volumeMounts: [
                    {
                        name: 'frameworkbarrier-volume',
                        mountPath: '/mnt/frameworkbarrier'
                    }]
            }];

        const spec: any = {
            containers: containers,
            initContainers: initContainers,
            restartPolicy: 'OnFailure',
            volumes: volumeSpecMap.get('nniVolumes'),
            hostNetwork: false
        };
        if (privateRegistrySecretName) {
            spec.imagePullSecrets = [
                {
                    name: privateRegistrySecretName
                }
            ]
        }

        if (this.config.serviceAccountName !== undefined) {
            spec.serviceAccountName = this.config.serviceAccountName;
        }

        return {
            pod: {
                spec: spec
            }
        };
    }

    public async getK8sJobInfo(environment: EnvironmentInformation): Promise<any> {
        if (this.kubernetesCRDClient === undefined) {
            throw new Error("kubernetesCRDClient undefined");
        }
        const kubeflowJobName: string = `nniexp${this.experimentId}env${environment.id}`.toLowerCase();
        const kubernetesJobInfo = await this.kubernetesCRDClient.getKubernetesJob(kubeflowJobName);
        return Promise.resolve(kubernetesJobInfo);
    }

}
