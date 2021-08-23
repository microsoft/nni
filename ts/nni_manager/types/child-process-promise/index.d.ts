declare module 'child-process-promise' {
    export function exec(command: string): Promise<childProcessPromise.Result>;

    export namespace childProcessPromise {
        interface Result {
            stdout: string;
            stderr: string,
            message: string
        }
    }
}