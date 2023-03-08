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
    private env: Record<string, string> | null = null;
    private log: Logger;

    constructor(config: ConnectConfig) {
        this.config = config;
        this.log = getLogger('Ssh.TODO');
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

    public setEnv(env: Record<string, string>): void {
        this.env = structuredClone(env);
    }

    public async exec(command: string): Promise<ExecResult> {
        this.log.debug('Execute command:', command);
        const deferred = new Deferred<void>();
        const result: ExecResult = { stdout: '', stderr: '' };

        const opts = this.env ? { env: this.env } : {};
        this.client!.exec(command, opts, (error, stream) => {
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
        this.log.debug('Command result:', result);
        return result;
    }

    public async run(command: string): Promise<string> {
        const result = await this.exec(command);
        if (result.code !== 0) {
            this.log.error('run failed:', command, result);
            throw new Error(`SSH run command failed: ${command}`);
        }
        return result.stdout.trim();
    }

    public async download(remotePath: string, localPath: string): Promise<void> {
        const sftp = await this.sftp();
        const fastGet = util.promisify(sftp.fastGet.bind(sftp));
        await fastGet(remotePath.replaceAll('\\', '/'), localPath);
    }

    public async upload(localPath: string, remotePath: string): Promise<void> {
        const sftp = await this.sftp();
        const fastPut = util.promisify(sftp.fastPut.bind(sftp));
        await fastPut(localPath, remotePath.replaceAll('\\', '/'));
    }

    public async writeFile(remotePath: string, data: string): Promise<void> {
        const sftp = await this.sftp();
        const stream = sftp.createWriteStream(remotePath.replaceAll('\\', '/'));
        const deferred = new Deferred<void>();
        stream.end(data, () => { deferred.resolve(); });
        return deferred.promise;

        // Following code does not work:  github.com/mscdex/ssh2/issues/1184
        //stream.end(data);
        //await events.once(stream, 'finish');
    }

    private async sftp(): Promise<SFTPWrapper> {
        if (!this.sftpSession) {
            const sftp = util.promisify(this.client!.sftp.bind(this.client!));
            this.sftpSession = await sftp();
        }
        return this.sftpSession;
    }
}
