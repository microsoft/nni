// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

import events from 'node:events';
import util from 'node:util';

import { Client, ConnectConfig, SFTPWrapper } from 'ssh2';

import { Deferred } from 'common/deferred';
import { Logger, getLogger } from 'common/log';

export interface ExecResult {
    code?: number;
    signal?: string;
    stdout: string;
    stderr: string;
}

export class Ssh {
    private config: ConnectConfig;
    private client: Client | null = null;
    private sftpSession: SFTPWrapper | null = null;

    constructor(config: ConnectConfig) {
        this.config = config;
    }

    public async connect(): Promise<void> {
        this.client = new Client();
        this.client.connect(this.config);
        await events.once(this.client, 'ready');
    }

    public disconnect(): void {
        if (this.client) {
            this.client.end();
        }
        this.client = null;
        this.sftpSession = null;
    }

    public async exec(command: string): Promise<ExecResult> {
        const deferred = new Deferred<void>();
        const result: ExecResult = { stdout: '', stderr: '' };

        this.client!.exec(command, (error, stream) => {
            if (error) {
                deferred.reject(error);
            } else {
                stream.on('data', (data: any) => { result.stdout += String(data); });
                stream.stderr.on('data', (data: any) => { result.stderr += String(data); });
                stream.on('close', (code: any, signal: any) => {
                    if (code || code === 0) {
                        result.code = Number(code);
                    }
                    if (signal) {
                        result.signal = String(signal);
                    }
                    deferred.resolve();
                });
            }
        });

        await deferred.promise;
        return result;
    }

    public async download(remotePath: string, localPath: string): Promise<void> {
        const sftp = await this.sftp();
        const f = util.promisify(sftp.fastGet.bind(sftp));
        await f(remotePath.replaceAll('\\', '/'), localPath);
    }

    public async upload(localPath: string, remotePath: string): Promise<void> {
        const sftp = await this.sftp();
        const f = util.promisify(sftp.fastPut.bind(sftp));
        await f(localPath, remotePath.replaceAll('\\', '/'));
    }

    private async sftp(): Promise<SFTPWrapper> {
        if (!this.sftpSession) {
            const f = util.promisify(this.client!.sftp.bind(this.client!));
            this.sftpSession = await f();
        }
        return this.sftpSession;
    }
}
