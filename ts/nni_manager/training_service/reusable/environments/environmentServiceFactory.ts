import { AMLEnvironmentService } from './amlEnvironmentService';
import { LocalEnvironmentService } from './localEnvironmentService';
import { RemoteEnvironmentService } from './remoteEnvironmentService';
import { KubeflowEnvironmentService } from './kubernetes/kubeflowEnvironmentService';
import { FrameworkControllerEnvironmentService } from './kubernetes/frameworkcontrollerEnvironmentService';
import { EnvironmentService } from '../environment';
import { TrainingServiceConfig } from 'common/experimentConfig';
import { ExperimentStartupInfo } from 'common/experimentStartupInfo';
import { getCustomEnvironmentServiceConfig } from 'common/nniConfig';
import { importModule } from 'common/utils';
import { DlcEnvironmentService } from './dlcEnvironmentService';

export async function createEnvironmentService(config: TrainingServiceConfig): Promise<EnvironmentService> {
    const info = ExperimentStartupInfo.getInstance();
    const configAsAny: any = config;  // environment services have different config types, skip type check

    switch (config.platform) {
        case 'local':
            return new LocalEnvironmentService(configAsAny, info);
        case 'remote':
            return new RemoteEnvironmentService(configAsAny, info);
        case 'aml':
            return new AMLEnvironmentService(configAsAny, info);
        case 'kubeflow':
            return new KubeflowEnvironmentService(configAsAny, info);
        case 'frameworkcontroller':
            return new FrameworkControllerEnvironmentService(configAsAny, info);
        case 'dlc':
            return new DlcEnvironmentService(configAsAny, info);
    }

    const esConfig = await getCustomEnvironmentServiceConfig(config.platform);
    if (esConfig === null) {
        throw new Error(`${config.platform} is not a supported training service!`);
    }
    const esModule = importModule(esConfig.nodeModulePath);
    const esClass = esModule[esConfig.nodeClassName] as any;
    return new esClass(configAsAny, info);
}
