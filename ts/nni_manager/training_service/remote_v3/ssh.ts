// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

/**
 *  Mainly a wrapper around ssh2 library to provide promise APIs.
 **/

import events from 'node:events';
import fs from 'node:fs/promises';
import util from 'node:util';

import { Client, ConnectConfig, SFTPWrapper } from 'ssh2';

import { Deferred } from 'common/deferred';
import type { RemoteMachineConfig } from 'common/experimentConfig';
import { Logger, getLogger } from 'common/log';

export interface ExecResult {
    code?: number;
    signal?: string;
    stdout: string;
    stderr: string;
}

export class Ssh {
    private config: RemoteMachineConfig;
    private client: Client | null = null;
    private sftpSession: SFTPWrapper | null = null;
    //private env: Record<string, string> | null = null;
    private path: string | null = null;
    private log: Logger;

    constructor(name: string, config: RemoteMachineConfig) {
        this.log = getLogger(`RemoteV3.Ssh.${name}`);
        this.config = config;
    }

    public async connect(): Promise<void> {
        this.log.debug('Connecting', this.config);

        const sshConfig: ConnectConfig = {
            host: this.config.host,
            port: this.config.port,
            username: this.config.user,
            password: this.config.password,
        };
        if (this.config.sshKeyFile) {
            sshConfig.privateKey = await fs.readFile(this.config.sshKeyFile, { encoding: 'utf8' });
            sshConfig.passphrase = this.config.sshPassphrase;
        }

        this.client = new Client();
        this.client.connect(sshConfig);
        await events.once(this.client, 'ready');

        this.log.debug('Connected');
    }

    public disconnect(): void {
        this.log.debug('Disconnect');
        if (this.client) {
            this.client.end();
        }
        this.client = null;
        this.sftpSession = null;
    }

    /**
     *  Set env for all future exec() and run() calls.
     *  FIXME: may not work (ssh2 bug)
     **/
    //public setEnv(env: Record<string, string>): void {
    //    this.log.trace('Update env:', env);
    //    this.env = structuredClone(env);
    //}

    public setPath(path: string): void {
        this.path = path;
    }

    /**
     *  Run a command and wait it to finish.
     *  Return exit code, stdout, stderr, etc.
     **/
    public async exec(command: string): Promise<ExecResult> {
        this.log.debug('Execute command:', command);
        const deferred = new Deferred<void>();
        const result: ExecResult = { stdout: '', stderr: '' };

        //const opts = this.env ? { env: this.env } : {};
        if (this.path !== null) {  // FIXME: workaround
            command = `PATH="${this.path}" ${command}`;
        }

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

        if (result.stdout.length > 100) {
            this.log.debug('Command result:', {
                code: result.code,
                stdout: result.stdout.slice(0, 80) + ' ...',
                stderr: result.stderr,
            });
            this.log.trace('Full result:', result);
        } else {
            this.log.debug('Command result:', result);
        }

        return result;
    }

    /**
     *  Run a command and wait it to finish.
     *  Return `stdout.trim()`.
     *
     *  If the command reports a non-zero exit code, throw an error.
     **/
    public async run(command: string): Promise<string> {
        const result = await this.exec(command);
        if (result.code !== 0) {
            this.log.error('Command failed:', command, result);
            throw new Error(`SSH command failed: ${command}`);
        }
        return result.stdout.trim();
    }

    public async download(remotePath: string, localPath: string): Promise<void> {
        this.log.debug(`Downloading ${localPath} <- ${remotePath}`);
        const sftp = await this.sftp();
        const fastGet = util.promisify(sftp.fastGet.bind(sftp));
        await fastGet(remotePath.replaceAll('\\', '/'), localPath);
        this.log.debug('Download success');
    }

    public async upload(localPath: string, remotePath: string): Promise<void> {
        this.log.debug(`Uploading ${localPath} -> ${remotePath}`);
        const sftp = await this.sftp();
        const fastPut = util.promisify(sftp.fastPut.bind(sftp));
        await fastPut(localPath, remotePath.replaceAll('\\', '/'));
        this.log.debug('Upload success');
    }

    public async writeFile(remotePath: string, data: string): Promise<void> {
        this.log.debug('Writing remote file', remotePath);
        const sftp = await this.sftp();
        const stream = sftp.createWriteStream(remotePath.replaceAll('\\', '/'));
        const deferred = new Deferred<void>();
        stream.end(data, () => { deferred.resolve(); });
        return deferred.promise;

        // Following code does not work (https://github.com/mscdex/ssh2/issues/1184)
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
