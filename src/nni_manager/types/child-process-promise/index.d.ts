declare module 'child-process-promise' {
    export function exec(command: string): Promise<void>;
}