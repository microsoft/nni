// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

/**
 *  Shutdown manager.
 *
 *  Each standalone module should register its clean up method with:
 *
 *      globals.shutdown.register('MyModule', async () => { this.cleanUp(); });
 *
 *  If a module is a children of another module (for example NNIDataStore is a children of NNIManager),
 *  it should not register shutdown callback on its own,
 *  instead the parent module should take care of its destruction.
 *
 *  Upon shutdown, all callbacks will be invoked *concurrently*. No guarantee on order.
 *
 *  A module can request for shutdown when unrecoverable error occurs:
 *
 *      try {
 *          this.doSomethingMustSuccess();
 *      } catch (error) {
 *          globals.shutdown.criticalError('MyModule', error);
 *      }
 *
 *  Note that when a module invokes `criticalError()`, its own registered callback will not get called.
 *
 *  When editting this module, keep robustness in mind.
 *  Bugs in this module can easily swallow logs and make it difficult to reproduce users' issue.
 **/

import { Logger, getRobustLogger } from 'common/log';

const logger: Logger = getRobustLogger('ShutdownManager');

export class ShutdownManager {
    private processStatus: 'initializing' | 'running' | 'stopping' = 'initializing';
    private modules: Map<string, () => Promise<void>> = new Map();
    private hasError: boolean = false;

    public register(moduleName: string, shutdownCallback: () => Promise<void>): void {
        if (this.modules.has(moduleName)) {
            logger.error(`Module ${moduleName} has registered twice.`, new Error().stack);
        }
        this.modules.set(moduleName, shutdownCallback);
    }

    public initiate(reason: string): void {
        if (this.processStatus === 'stopping') {
            logger.warning('initiate() invoked but already stopping:', reason);
        } else {
            logger.info('Initiate shutdown:', reason);
            this.shutdown();
        }
    }

    public criticalError(moduleName: string, error: Error): void {
        logger.critical(`Critical error ocurred in module ${moduleName}:`, error);
        this.hasError = true;
        if (this.processStatus === 'initializing') {
            logger.error('Starting failed.');
            process.exit(1);
        } else if (this.processStatus !== 'stopping') {
            this.modules.delete(moduleName);
            this.shutdown();
        }
    }

    public notifyInitializeComplete(): void {
        if (this.processStatus === 'initializing') {
            this.processStatus = 'running';
        } else {
            logger.error('notifyInitializeComplete() invoked in status', this.processStatus);
        }
    }

    private shutdown(): void {
        this.processStatus = 'stopping';

        const promises = Array.from(this.modules).map(async ([moduleName, callback]) => {
            try {
                await callback();
            } catch (error) {
                logger.error(`Error during shutting down ${moduleName}:`, error);
                this.hasError = true;
            }
            this.modules.delete(moduleName);
        });

        const timeoutTimer = setTimeout(async () => {
            try {
                logger.error('Following modules failed to shut down in time:', Array.from(this.modules.keys()));
                await global.nni.logStream.close();
            } finally {
                process.exit(1);
            }
        }, shutdownTimeout);

        Promise.all(promises).then(async () => {
            try {
                clearTimeout(timeoutTimer);
                logger.info('Shutdown complete.');
                await global.nni.logStream.close();
            } finally {
                process.exit(this.hasError ? 1 : 0);
            }
        });
    }
}

let shutdownTimeout: number = 60_000;

export namespace UnitTestHelpers {
    export function setShutdownTimeout(ms: number): void {
        shutdownTimeout = ms;
    }
}
