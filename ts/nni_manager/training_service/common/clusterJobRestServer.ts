// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import assert from 'assert';
import bodyParser from 'body-parser';
import { Request, Response, Router } from 'express';
import fs from 'fs';
import path from 'path';
import { Writable } from 'stream';
import { String } from 'typescript-string-operations';
import { getBasePort, getExperimentId } from 'common/experimentStartupInfo';
import { LegacyRestServer } from 'common/restServer';
import { getExperimentRootDir, mkDirPSync } from 'common/utils';

/**
 * Cluster Job Training service Rest server, provides rest API to support Cluster job metrics update
 *
 * FIXME: This should be a router, not a separate REST server.
 */
export abstract class ClusterJobRestServer extends LegacyRestServer {
    private readonly API_ROOT_URL: string = '/api/v1/nni-pai';
    private readonly NNI_METRICS_PATTERN: string = `NNISDK_MEb'(?<metrics>.*?)'`;

    private readonly expId: string = getExperimentId();

    private enableVersionCheck: boolean = true; //switch to enable version check
    private versionCheckSuccess: boolean | undefined;
    private errorMessage?: string;

    /**
     * constructor to provide NNIRestServer's own rest property, e.g. port
     */
    constructor() {
        super();
        const basePort: number = getBasePort();
        assert(basePort !== undefined && basePort > 1024);

        this.port = basePort + 1;
    }

    get apiRootUrl(): string {
        return this.API_ROOT_URL;
    }

    public get clusterRestServerPort(): number {
        if (this.port === undefined) {
            throw new Error('PAI Rest server port is undefined');
        }

        return this.port;
    }

    public get getErrorMessage(): string | undefined {
        return this.errorMessage;
    }

    public set setEnableVersionCheck(versionCheck: boolean) {
        this.enableVersionCheck = versionCheck;
    }

    /**
     * NNIRestServer's own router registration
     */
    protected registerRestHandler(): void {
        this.app.use(bodyParser.json());
        this.app.use(this.API_ROOT_URL, this.createRestHandler());
    }

    // Abstract method to handle trial metrics data
    protected abstract handleTrialMetrics(jobId: string, trialMetrics: any[]): void;

    protected createRestHandler(): Router {
        const router: Router = Router();

        router.use((req: Request, res: Response, next: any) => {
            this.log.info(`${req.method}: ${req.url}: body:`, req.body);
            res.setHeader('Content-Type', 'application/json');
            next();
        });

        router.post(`/version/${this.expId}/:trialId`, (req: Request, res: Response) => {
            if (this.enableVersionCheck) {
                try {
                    const checkResultSuccess: boolean = req.body.tag === 'VCSuccess' ? true : false;
                    if (this.versionCheckSuccess !== undefined && this.versionCheckSuccess !== checkResultSuccess) {
                        this.errorMessage = 'Version check error, version check result is inconsistent!';
                        this.log.error(this.errorMessage);
                    } else if (checkResultSuccess) {
                        this.log.info(`Version check in trialKeeper success!`);
                        this.versionCheckSuccess = true;
                    } else {
                        this.versionCheckSuccess = false;
                        this.errorMessage = req.body.msg;
                    }
                } catch (err) {
                    this.log.error(`json parse metrics error: ${err}`);
                    res.status(500);
                    res.send((err as any).message);
                }
            } else {
                this.log.info(`Skipping version check!`);
            }
            res.send();
        });

        router.post(`/update-metrics/${this.expId}/:trialId`, (req: Request, res: Response) => {
            try {
                this.log.info(`Get update-metrics request, trial job id is ${req.params['trialId']}`);
                this.log.info('update-metrics body is', req.body);

                this.handleTrialMetrics(req.body.jobId, req.body.metrics);

                res.send();
            } catch (err) {
                this.log.error(`json parse metrics error: ${err}`);
                res.status(500);
                res.send((err as any).message);
            }
        });

        router.post(`/stdout/${this.expId}/:trialId`, (req: Request, res: Response) => {
            if (this.enableVersionCheck && (this.versionCheckSuccess === undefined || !this.versionCheckSuccess)
            && this.errorMessage === undefined) {
                this.errorMessage = `Version check failed, didn't get version check response from trialKeeper,`
                 + ` please check your NNI version in NNIManager and TrialKeeper!`;
            }
            const trialLogDir: string = path.join(getExperimentRootDir(), 'trials', req.params['trialId']);
            mkDirPSync(trialLogDir);
            const trialLogPath: string = path.join(trialLogDir, 'stdout_log_collection.log');
            try {
                let skipLogging: boolean = false;
                if (req.body.tag === 'trial' && req.body.msg !== undefined) {
                    const metricsContent: any = req.body.msg.match(this.NNI_METRICS_PATTERN);
                    if (metricsContent && metricsContent.groups) {
                        const key: string = 'metrics';
                        this.handleTrialMetrics(req.params['trialId'], [metricsContent.groups[key]]);
                        skipLogging = true;
                    }
                }

                if (!skipLogging) {
                    // Construct write stream to write remote trial's log into local file
                    const writeStream: Writable = fs.createWriteStream(trialLogPath, {
                        flags: 'a+',
                        encoding: 'utf8',
                        autoClose: true
                    });

                    writeStream.write(String.Format('{0}\n', req.body.msg));
                    writeStream.end();
                }
                res.send();
            } catch (err) {
                this.log.error(`json parse stdout data error: ${err}`);
                res.status(500);
                res.send((err as any).message);
            }
        });

        return router;
    }
}
