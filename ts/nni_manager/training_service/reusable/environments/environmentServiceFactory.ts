import { AMLEnvironmentService } from './amlEnvironmentService';
import { OpenPaiEnvironmentService } from './openPaiEnvironmentService';
import { LocalEnvironmentService } from './localEnvironmentService';
import { RemoteEnvironmentService } from './remoteEnvironmentService';
import { EnvironmentService } from '../environment';
import { ExperimentConfig } from '../../../common/experimentConfig';
import { getCustomEnvironmentServiceConfig } from '../../../common/nniConfig';
import { importModule } from '../../../common/utils';

export class EnvironmentServiceFactory {
    public static createEnvironmentService(name: string, config: ExperimentConfig, expId: string, rootDir: string): EnvironmentService {
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

        const esConfig = getCustomEnvironmentServiceConfig(name);
        if (esConfig === null) {
            throw new Error(`${name} is not a supported training service!`);
        }
        const esModule = importModule(esConfig.nodeModulePath);
        const esClass = esModule[esConfig.nodeClassName] as any;
        return new esClass(rootDir, expId, config);
    }
}
