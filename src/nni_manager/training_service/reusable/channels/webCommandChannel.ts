/**
 * Copyright (c) Microsoft Corporation
 * All rights reserved.
 *
 * MIT License
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
 * documentation files (the "Software"), to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
 * to permit persons to whom the Software is furnished to do so, subject to the following conditions:
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

'use strict';

import { Server as HttpServer } from 'http';
import { Server as SocketServer } from "ws";
import { getExperimentId } from "../../../common/experimentStartupInfo";
import { INITIALIZED } from '../../../core/commands';
import { CommandChannel, RunnerConnection } from "../commandChannel";
import { Channel, EnvironmentInformation } from "../environment";

class WebRunnerConnection extends RunnerConnection {
    public readonly clients: WebSocket[] = [];

    public async close(): Promise<void> {
        await super.close();
        while (this.clients.length > 0) {
            const client = this.clients.shift();
            if (client !== undefined) {
                client.close();
            }
        }
    }

    public AddClient(client: WebSocket): void {
        this.clients.push(client);
    }
}

export class WebCommandChannel extends CommandChannel {
    private readonly expId: string = getExperimentId();

    private httpServer: HttpServer | undefined;
    private webSocketServer: SocketServer | undefined;
    private clients: Map<WebSocket, WebRunnerConnection | undefined> = new Map<WebSocket, WebRunnerConnection | undefined>();

    public get channelName(): Channel {
        return "web";
    }

    public async config(key: string, value: any): Promise<void> {
        switch (key) {
            case "RestServer":
                this.httpServer = value as HttpServer;
                break;
        }
    }

    public start(): void {
        if (this.httpServer === undefined) {
            throw new Error(`http server is not initialized!`);
        }

        const server = this.httpServer;
        this.webSocketServer = new SocketServer({ server });

        this.webSocketServer.on('connection', (client: WebSocket) => {
            this.log.debug(`WebCommandChannel: received connection`);

            this.clients.set(client, undefined);
            client.onmessage = (message): void => {
                this.receivedWebSocketMessage(client, message);
            };
        });
    }

    public stop(): void {
        if (this.webSocketServer !== undefined) {
            this.webSocketServer.close();
        }
    }

    protected async sendCommandInternal(environment: EnvironmentInformation, message: string): Promise<void> {
        if (this.webSocketServer === undefined) {
            throw new Error(`WebCommandChannel: uninitialized!`)
        }
        const runnerConnection = this.runnerConnections.get(environment.id) as WebRunnerConnection;
        if (runnerConnection !== undefined) {
            for (const client of runnerConnection.clients) {
                client.send(message);
            }
        } else {
            this.log.warning(`WebCommandChannel: cannot find client for env ${environment.id}, message is ignored.`);
        }
    }

    protected createRunnerConnection(environment: EnvironmentInformation): RunnerConnection {
        return new WebRunnerConnection(environment);
    }

    private receivedWebSocketMessage(client: WebSocket, message: MessageEvent): void {
        let connection = this.clients.get(client) as WebRunnerConnection | undefined;
        const rawCommands = message.data.toString();

        if (connection === undefined) {
            // undefined means it's expecting initializing message.
            const commands = this.parseCommands(rawCommands);
            let isValid = false;
            this.log.debug(`WebCommandChannel: received initialize message: ${JSON.stringify(rawCommands)}`);

            if (commands.length > 0) {
                const commandType = commands[0][0];
                const result = commands[0][1];
                if (commandType === INITIALIZED &&
                    result.expId === this.expId &&
                    this.runnerConnections.has(result.runnerId)
                ) {
                    const runnerConnection = this.runnerConnections.get(result.runnerId) as WebRunnerConnection;
                    this.clients.set(client, runnerConnection);
                    runnerConnection.AddClient(client);
                    connection = runnerConnection;
                    isValid = true;
                }
            }

            if (!isValid) {
                this.log.warning(`WebCommandChannel: rejected client with invalid init message ${rawCommands}`);
                client.close();
                this.clients.delete(client);
            }
        }

        if (connection !== undefined) {
            this.handleCommand(connection.environment, rawCommands);
        }
    }
}
