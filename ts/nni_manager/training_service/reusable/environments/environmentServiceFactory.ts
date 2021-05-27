import { AMLEnvironmentService } from './amlEnvironmentService';
import { OpenPaiEnvironmentService } from './openPaiEnvironmentService';
import { LocalEnvironmentService } from './localEnvironmentService';
import { RemoteEnvironmentService } from './remoteEnvironmentService';
import { EnvironmentService } from '../environment';
import { ExperimentConfig } from '../../../common/experimentConfig';
import { getExperimentId } from '../../../common/experimentStartupInfo';
import { getCustomEnvironmentServiceConfig } from '../../../common/nniConfig';
import { getExperimentRootDir, importModule } from '../../../common/utils';


export async function createEnvironmentService(name: string, config: ExperimentConfig): Promise<EnvironmentService> {
    const expId = getExperimentId();
    const rootDir = getExperimentRootDir();

    switch(name) {
        case 'local':
            return new LocalEnvironmentService(rootDir, expId, config);
        case 'remote':
            return new RemoteEnvironmentService(rootDir, expId, config);
        case 'aml':
            return new AMLEnvironmentService(rootDir, expId, config);
        case 'openpai':
            return new OpenPaiEnvironmentService(rootDir, expId, config);
    }

    const esConfig = await getCustomEnvironmentServiceConfig(name);
    if (esConfig === null) {
        throw new Error(`${name} is not a supported training service!`);
    }
    const esModule = importModule(esConfig.nodeModulePath);
    const esClass = esModule[esConfig.nodeClassName] as any;
    return new esClass(rootDir, expId, config);
}
