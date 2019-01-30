declare module 'tail-stream' {
    export interface Stream {
        on(type: 'data', callback: (data: Buffer) => void): void;
    }
    export function createReadStream(path: string): Stream;
}