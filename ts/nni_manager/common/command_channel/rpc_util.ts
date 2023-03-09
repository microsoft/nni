// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

/**
 *  Add RPC capability to WebSocket command channel.
 *
 *
 **/

import util from 'node:util';

import type { Command } from 'common/command_channel/interface';
import { DefaultMap } from 'common/default_map';
import { Deferred } from 'common/deferred';
import { Logger, getLogger } from 'common/log';
import type { TrialKeeper } from 'common/trial_keeper/keeper';
import { WsChannel } from './websocket/channel';

interface RpcResponseCommand {
    type: 'rpc_response';
    id: number;
    result?: any;
    error?: string;
}

const rpcHelpers: Map<WsChannel, RpcHelper> = new Map();

export function getRpcHelper(channel: WsChannel): RpcHelper {
    if (!rpcHelpers.has(channel)) {
        rpcHelpers.set(channel, new RpcHelper(channel));
    }
    return rpcHelpers.get(channel)!;
}

type Class = { new(...args: any[]): any; };

export class RpcHelper {
    private channel: WsChannel;
    private lastId: number = 0;
    private localCtors: Map<string, any> = new Map();
    private localObjs: Map<number, any> = new Map();
    private localCbs: Map<number, any> = new Map();
    private responses: DefaultMap<number, Deferred<RpcResponseCommand>> = new DefaultMap(() => new Deferred());
    private log: Logger = getLogger('RpcHelper.TODO');

    constructor(channel: WsChannel) {
        this.channel = channel;
        this.channel.onCommand('rpc_constructor', command => {
            this.log.debug('invoke constructor', command);
            this.invokeLocalConstructor(command.id, command.className, command.parameters);
        });
        this.channel.onCommand('rpc_method', command => {
            this.invokeLocalMethod(command.id, command.objectId, command.methodName, command.parameters, command.callbackIds);
        });
        this.channel.onCommand('rpc_callback', command => {
            this.invokeLocalCallback(command.id, command.callbackId, command.parameters);
        });
        this.channel.onCommand('rpc_response', command => {
            this.responses.get(command.id).resolve(command);
        });
    }

    public registerClass(className: string, constructor: Class): void {
        this.localCtors.set(className, constructor);
    }

    public construct(className: string, parameters?: any[]): Promise<number> {
        return this.invokeRemoteConstructor(className, parameters ?? []);
    }

    public call(objectId: number, methodName: string, parameters?: any[], callbacks?: any[]): Promise<any> {
        return this.invokeRemoteMethod(objectId, methodName, parameters ?? [], callbacks ?? []);
    }

    private async invokeRemoteConstructor(className: string, parameters: any[]): Promise<number> {
        const id = this.generateId();
        this.channel.send({ type: 'rpc_constructor', id, className, parameters });
        await this.waitResponse(id);
        return id;
    }

    private invokeLocalConstructor(id: number, className: string, parameters: any[]): void {
        const ctor = this.localCtors.get(className);
        if (!ctor) {
            this.sendRpcError(id, `Unknown class name ${className}`);
            return;
        }

        let obj;
        try {
            obj = new ctor(...parameters);
        } catch (error) {
            this.log.debug('ctor error', error);
            this.sendError(id, error);
            return;
        }

        this.localObjs.set(id, obj);
        this.log.debug('ctor success');
        this.sendResult(id, undefined);
    }

    private async invokeRemoteMethod(
            objectId: number, methodName: string, parameters: any[], callbacks: any[]): Promise<any> {

        const id = this.generateId();
        const callbackIds = this.generateCallbackIds(callbacks);
        this.channel.send({ type: 'rpc_method', id, objectId, methodName, parameters, callbackIds });
        return await this.waitResponse(id);
    }

    private async invokeLocalMethod(
            id: number, objectId: number, methodName: string,
            parameters: any[], callbackIds: number[]): Promise<void> {

        const obj = this.localObjs.get(objectId);
        if (!obj) {
            this.sendRpcError(id, `Non-exist object ${objectId}`);
            return;
        }
        const callbacks = this.createCallbacks(callbackIds);

        let result;
        try {
            result = obj[methodName](...parameters, ...callbacks);
            if (typeof result === 'object' && result.then) {
                result = await result;
            }
        } catch (error) {
            this.sendError(id, error);
            return;
        }

        this.sendResult(id, result);
    }

    private invokeRemoteCallback(callbackId: number, parameters: any[]): void {
        const id = this.generateId();  // for debug purpose
        this.channel.send({ type: 'rpc_callback', id, callbackId, parameters });
    }

    private invokeLocalCallback(_id: number, callbackId: number, parameters: any[]): void {
        const cb = this.localCbs.get(callbackId);
        if (cb) {
            cb(...parameters);
        } else {
            this.log.error('Non-exist callback ID', callbackId);
        }
    }

    private generateId(): number {
        this.lastId += 1;
        return this.lastId;
    }

    private generateCallbackIds(callbacks: any[]): number[] {
        const ids = [];
        for (const cb of callbacks) {
            const id = this.generateId();
            ids.push(id);
            this.localCbs.set(id, cb);
        }
        return ids;
    }

    private createCallbacks(callbackIds: number[]): any[] {
        return callbackIds.map(id => (
            (...args: any[]) => { this.invokeRemoteCallback(id, args); }
        ));
    }

    private sendResult(id: number, result: any): void {
        this.channel.send({ type: 'rpc_response', id, result });
    }

    private sendError(id: number, error: any): void {
        const msg = error.stack ? String(error.stack) : util.inspect(error);
        this.channel.send({ type: 'rpc_response', id, error: msg });
    }

    private sendRpcError(id: number, message: string): void {
        this.channel.send({ type: 'rpc_response', id, error: `RPC framework error: ${message}` });
    }

    private async waitResponse(id: number): Promise<any> {
        const deferred = this.responses.get(id);
        const res = await deferred.promise;
        if (res.error) {
            throw new Error(`RPC remote error:\n${res.error}`);
        }
        return res.result;
    }
}
