// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

/**
 *  WIP REST API manager.
 *  Functions will be added when used.
 **/

import express, { Express, Request, Response, Router } from 'express';
import expressWs, { Router as WsRouter } from 'express-ws';
import type { WebSocket } from 'ws';

type HttpMethod = 'GET' | 'PUT';

type ExpressCallback = (req: Request, res: Response) => void;
type WebSocketCallback = (ws: WebSocket, req: Request) => void;

export class RestManager {
    private app: Express;
    private router: Router;

    constructor() {
        // we don't actually need the app here,
        // but expressWs() must be called before router.ws(), and it requires an app instance
        this.app = express();
        expressWs(this.app, undefined, { wsOptions: { maxPayload: 4 * 1024 * 1024 * 1024 }});

        this.router = Router();
        this.router.use(express.json({ limit: '50mb' }));
    }

    public getExpressApp(): Express {
        return this.app;
    }

    public getExpressRouter(): Router {
        return this.router;
    }

    public registerSyncHandler(method: HttpMethod, path: string, callback: ExpressCallback): void {
        const p = '/' + trimSlash(path);
        if (method === 'GET') {
            this.router.get(p, callback);
        } else if (method === 'PUT') {
            this.router.put(p, callback);
        } else {
            throw new Error(`RestManager: Bad method ${method}`);
        }
    }

    public registerWebSocketHandler(path: string, callback: WebSocketCallback): void {
        const p = '/' + trimSlash(path);
        (this.router as WsRouter).ws(p, callback);
    }

    public registerExpressRouter(path: string, router: Router): void {
        this.router.use(path, router);
    }

    public urlJoin(...parts: string[]): string {
        return parts.map(trimSlash).filter(part => part).join('/');
    }

    public getFullUrl(protocol: string, ip: string, ...parts: string[]): string {
        const root = `${protocol}://${ip}:${global.nni.args.port}/`;
        return root + this.urlJoin(global.nni.args.urlPrefix, ...parts);
    }
}

function trimSlash(s: string): string {
    return s.replace(/^\/+|\/+$/g, '');
}
