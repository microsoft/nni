import { AMLEnvironmentService } from './amlEnvironmentService';
import { OpenPaiEnvironmentService } from './openPaiEnvironmentService';
import { LocalEnvironmentService } from './localEnvironmentService';
import { RemoteEnvironmentService } from './remoteEnvironmentService';
import { EnvironmentService } from '../environment';

export class EnvironmentServiceFactory {
    public static createEnvironmentService(name: string): EnvironmentService {
        switch(name) {
            case 'local':
                return new LocalEnvironmentService();
            case 'remote':
                return new RemoteEnvironmentService();
            case 'aml':
                return new AMLEnvironmentService();
            case 'pai':
                return new OpenPaiEnvironmentService();
            default:
                throw new Error(`${name} not supported!`);
        }
    }
}
