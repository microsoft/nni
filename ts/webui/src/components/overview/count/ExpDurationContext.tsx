import React from 'react';
export const ExpDurationContext = React.createContext({
    maxExecDuration: 0,
    execDuration: 0,
    // eslint-disable-next-line @typescript-eslint/no-empty-function
    updateOverviewPage: (): void => {},
    maxDurationUnit: 'm',
    // eslint-disable-next-line @typescript-eslint/no-empty-function
    changeMaxDurationUnit: (_val: 'd' | 'h' | 'm'): void => {}
});
