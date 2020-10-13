import React from 'react';
export const ExpDurationContext = React.createContext({
    maxExecDuration: 0,
    execDuration: 0,
    // eslint-disable-next-line @typescript-eslint/no-empty-function
    updateOverviewPage: (): void => {}
});
