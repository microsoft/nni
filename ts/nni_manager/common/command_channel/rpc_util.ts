// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

/**
 *  Add RPC capability to WebSocket command channel.
 *
 *  Suppose you want to run `new Foo(1, 2).on('event', localListener)` on a remote worker,
 *  use the following steps:
 *
 *   0. Create RPC helper on each side of the channel:
 *
 *          const rpcLocal = getRpcHelper(localChannel);  // at local side
 *          const rpcRemote = getRpcHelper(remoteChannel);  // at remote side
 *
 *   1. Register the class at remote:
 *
 *          rpcRemote.registerClass('Foo', Foo);
 *
 *   2. Construct an object at local:
 *
 *          const obj = await rpcLocal.construct('Foo', [ 1, 2 ]);
 *
 *   3. Call the method at local:
 *
 *          const result = await rpcLocal.call(obj, 'on', [ 'event' ], [ listener ]);
 *
 *  RPC methods can only have parameters of JSON and function types,
 *  and all callbacks must appear after JSON parameters.
 *
 *  This utility has limited support for exceptions.
 *  If the remote method throws an Error, `rpc.call()` will throw an Error as well.
 *  The local error message contains `inspect(remoteError)` and should be sufficient for debugging purpose.
 *  However, those two errors are totally different objects so don't try to write concrete error handlers.
 *
 *  Underlying, it uses a protocal similar to JSON-RPC:
 *
 *      ->  { type: 'rpc_constructor', id: 1, className: 'Foo', parameters: [ 1, 2 ] }
 *      <-  { type: 'rpc_response', id: 1 }
 *
 *      ->  { type: 'rpc_method', id: 2, objectId: 1, methodName: 'bar', parameters: [ 'event' ], callbackIds: [ 3 ] }
 *      <-  { type: 'rpc_response', id: 2, result: 'result' }
 *
 *      <-  { type: 'rpc_callback', id: 4, callbackId: 3, parameters: [ 'baz' ] }
 **/

import util from 'node:util';

import { DefaultMap } from 'common/default_map';
import { Deferred } from 'common/deferred';
import { Logger, getLogger } from 'common/log';
import type { CommandChannel } from './interface';

interface RpcResponseCommand {
    type: 'rpc_response';
    id: number;
    result?: any;
    error?: string;
}

type Class = { new(...args: any[]): any; };

const rpcHelpers: Map<CommandChannel, RpcHelper> = new Map();

/**
 *  Enable RPC on a channel.
 *
 *  The channel does not need to be connected for calling this function.
 **/
export function getRpcHelper(channel: CommandChannel): RpcHelper {
    if (!rpcHelpers.has(channel)) {
        rpcHelpers.set(channel, new RpcHelper(channel));
    }
    return rpcHelpers.get(channel)!;
}

export class RpcHelper {
    private channel: CommandChannel;
    private lastId: number = 0;
    private localCtors: Map<string, Class> = new Map();
    private localObjs: Map<number, any> = new Map();
    private localCbs: Map<number, Function> = new Map();  // eslint-disable-line
    private log: Logger;
    private responses: DefaultMap<number, Deferred<RpcResponseCommand>> = new DefaultMap(() => new Deferred());

    /**
     *  NOTE: Don't use this constructor directly. Use `getRpcHelper()`.
     **/
    constructor(channel: CommandChannel) {
        this.log = getLogger(`RpcHelper.${channel.name}`);
        this.channel = channel;
        this.channel.onCommand('rpc_constructor', command => {
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

    /**
     *  Register a class for RPC use.
     *
     *  This method must be called at remote side before calling `construct()` at local side.
     *  To ensure this, the client can call `getRpcHelper().registerClass()` before calling `connect()`.
     **/
    public registerClass(className: string, constructor: Class): void {
        this.log.debug('Register class', className);
        this.localCtors.set(className, constructor);
    }

    /**
     *  Construct a class object remotely.
     *
     *  Must be called after `registerClass()` at remote side, or an error will be raised.
     **/
    public construct(className: string, parameters?: any[]): Promise<number> {
        return this.invokeRemoteConstructor(className, parameters ?? []);
    }

    /**
     *  Call a method on a remote object.
     *
     *  The `objectId` is the return value of `construct()`.
     *
     *  If the method returns a promise, `call()` will wait for it to resolve.
     **/
    public call(objectId: number, methodName: string, parameters?: any[], callbacks?: any[]): Promise<any> {
        return this.invokeRemoteMethod(objectId, methodName, parameters ?? [], callbacks ?? []);
    }

    private async invokeRemoteConstructor(className: string, parameters: any[]): Promise<number> {
        this.log.debug('Send constructor command', className, parameters);
        const id = this.generateId();
        this.channel.send({ type: 'rpc_constructor', id, className, parameters });
        await this.waitResponse(id);
        return id;
    }

    private invokeLocalConstructor(id: number, className: string, parameters: any[]): void {
        this.log.debug('Receive constructor command', className, parameters);
        const ctor = this.localCtors.get(className);
        if (!ctor) {
            this.sendRpcError(id, `Unknown class name ${className}`);
            return;
        }

        let obj;
        try {
            obj = new ctor(...parameters);
        } catch (error) {
            this.log.debug('Constructor throws error', className, error);
            this.sendError(id, error);
            return;
        }

        this.localObjs.set(id, obj);
        this.sendResult(id, undefined);
    }

    private async invokeRemoteMethod(
            objectId: number, methodName: string, parameters: any[], callbacks: any[]): Promise<any> {
        this.log.debug('Send method command', methodName, parameters);
        const id = this.generateId();
        const callbackIds = this.generateCallbackIds(callbacks);
        this.channel.send({ type: 'rpc_method', id, objectId, methodName, parameters, callbackIds });
        return await this.waitResponse(id);
    }

    private async invokeLocalMethod(
            id: number, objectId: number, methodName: string,
            parameters: any[], callbackIds: number[]): Promise<void> {
        this.log.debug('Receive method command', methodName, parameters);

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
            this.log.debug('Command throws error', methodName, error);
            this.sendError(id, error);
            return;
        }

        this.log.debug('Command returns', result);
        this.sendResult(id, result);
    }

    private invokeRemoteCallback(callbackId: number, parameters: any[]): void {
        this.log.debug('Send callback command', parameters);
        const id = this.generateId();  // for debug purpose
        this.channel.send({ type: 'rpc_callback', id, callbackId, parameters });
    }

    private invokeLocalCallback(_id: number, callbackId: number, parameters: any[]): void {
        this.log.debug('Receive callback command', parameters);
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
        try {
            JSON.stringify(result);
        } catch {
            this.sendRpcError(id, 'method returns non-JSON value ' + util.inspect(result));
            return;
        }

        this.channel.send({ type: 'rpc_response', id, result });
    }

    private sendError(id: number, error: any): void {
        const msg = error?.stack ? String(error.stack) : util.inspect(error);
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
