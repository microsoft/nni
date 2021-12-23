import { AMLEnvironmentService } from './amlEnvironmentService';
import { OpenPaiEnvironmentService } from './openPaiEnvironmentService';
import { LocalEnvironmentService } from './localEnvironmentService';
import { RemoteEnvironmentService } from './remoteEnvironmentService';
import { KubeflowEnvironmentService } from './kubernetes/kubeflowEnvironmentService';
import { FrameworkControllerEnvironmentService } from './kubernetes/frameworkcontrollerEnvironmentService';
import { EnvironmentService } from '../environment';
import { ExperimentConfig } from 'common/experimentConfig';
import { ExperimentStartupInfo } from 'common/experimentStartupInfo';
import { getCustomEnvironmentServiceConfig } from 'common/nniConfig';
import { importModule } from 'common/utils';
import { DlcEnvironmentService } from './dlcEnvironmentService';

export async function createEnvironmentService(name: string, config: ExperimentConfig): Promise<EnvironmentService> {
    const info = ExperimentStartupInfo.getInstance();
    const tsConfig: any = config.trainingService;

    switch (name) {
        case 'local':
            return new LocalEnvironmentService(tsConfig, info);
        case 'remote':
            return new RemoteEnvironmentService(tsConfig, info);
        case 'aml':
            return new AMLEnvironmentService(tsConfig, info);
        case 'openpai':
            return new OpenPaiEnvironmentService(tsConfig, info);
        case 'kubeflow':
            return new KubeflowEnvironmentService(tsConfig, info);
        case 'frameworkcontroller':
            return new FrameworkControllerEnvironmentService(tsConfig, info);
        case 'dlc':
            return new DlcEnvironmentService(tsConfig, info);
    }

    const esConfig = await getCustomEnvironmentServiceConfig(name);
    if (esConfig === null) {
        throw new Error(`${name} is not a supported training service!`);
    }
    const esModule = importModule(esConfig.nodeModulePath);
    const esClass = esModule[esConfig.nodeClassName] as any;
    return new esClass(tsConfig, info);
}
