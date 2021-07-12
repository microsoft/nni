import { AMLEnvironmentService } from './amlEnvironmentService';
import { OpenPaiEnvironmentService } from './openPaiEnvironmentService';
import { LocalEnvironmentService } from './localEnvironmentService';
import { RemoteEnvironmentService } from './remoteEnvironmentService';
import { EnvironmentService } from '../environment';
import { ExperimentConfig } from '../../../common/experimentConfig';
import { ExperimentStartupInfo } from '../../../common/experimentStartupInfo';
import { getCustomEnvironmentServiceConfig } from '../../../common/nniConfig';
import { importModule } from '../../../common/utils';

export async function createEnvironmentService(name: string, config: ExperimentConfig): Promise<EnvironmentService> {
    const info = ExperimentStartupInfo.getInstance();

    switch(name) {
        case 'local':
            return new LocalEnvironmentService(config, info);
        case 'remote':
            return new RemoteEnvironmentService(config, info);
        case 'aml':
            return new AMLEnvironmentService(config, info);
        case 'openpai':
            return new OpenPaiEnvironmentService(config, info);
    }

    const esConfig = await getCustomEnvironmentServiceConfig(name);
    if (esConfig === null) {
        throw new Error(`${name} is not a supported training service!`);
    }
    const esModule = importModule(esConfig.nodeModulePath);
    const esClass = esModule[esConfig.nodeClassName] as any;
    return new esClass(config, info);
}
