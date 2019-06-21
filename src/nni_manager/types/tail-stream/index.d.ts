declare module 'tail-stream' {
    export interface Stream {
        on(type: 'data', callback: (data: Buffer) => void): void;
        end(data: number): void;
        emit(data: string): void;
    }
    export function createReadStream(path: string): Stream;
}