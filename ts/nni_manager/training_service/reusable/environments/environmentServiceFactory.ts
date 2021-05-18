import { AMLEnvironmentService } from './amlEnvironmentService';
import { OpenPaiEnvironmentService } from './openPaiEnvironmentService';
import { LocalEnvironmentService } from './localEnvironmentService';
import { RemoteEnvironmentService } from './remoteEnvironmentService';
import { EnvironmentService } from '../environment';
import { ExperimentConfig } from '../../../common/experimentConfig';
import { getCustomEnvironmentServiceConfig } from '../../../common/nniConfig';
import { importModule } from '../../../common/utils';

export class EnvironmentServiceFactory {
    public static createEnvironmentService(name: string, config: ExperimentConfig): EnvironmentService {
        switch(name) {
            case 'local':
                return new LocalEnvironmentService(config);
            case 'remote':
                return new RemoteEnvironmentService(config);
            case 'aml':
                return new AMLEnvironmentService(config);
            case 'openpai':
                return new OpenPaiEnvironmentService(config);
        }

        const customEs = getCustomEnvironmentServiceConfig(name);
        if (customEs === null) {
            throw new Error(`${name} is not a supported training service!`);
        }
        const module_ = importModule(customEs.nodeModulePath);
        const class_ = module_[customEs.nodeClassName] as any;
        return new class_(config);
    }
}
