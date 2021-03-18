import { AMLEnvironmentService } from './amlEnvironmentService';
import { OpenPaiEnvironmentService } from './openPaiEnvironmentService';
import { LocalEnvironmentService } from './localEnvironmentService';
import { RemoteEnvironmentService } from './remoteEnvironmentService';
import { EnvironmentService } from '../environment';
import { ExperimentConfig } from '../../../common/experimentConfig';

export class EnvironmentServiceFactory {
    public static createEnvironmentService(name: string, config: ExperimentConfig): EnvironmentService {
        switch(name) {
            case 'local':
                return new LocalEnvironmentService(config);
            case 'remote':
                return new RemoteEnvironmentService(config);
            //case 'aml':
            //    return new AMLEnvironmentService();
            //case 'pai':
            //    return new OpenPaiEnvironmentService();
            default:
                throw new Error(`${name} not supported!`);
        }
    }
}
