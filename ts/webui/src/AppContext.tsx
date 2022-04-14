import React from 'react';
import { COLUMN } from '@static/const';

const AppContext = React.createContext({
    interval: 10, // sendons
    columnList: COLUMN,
    experimentUpdateBroadcast: 0,
    trialsUpdateBroadcast: 0,
    metricGraphMode: 'max',
    bestTrialEntries: '10',
    maxDurationUnit: 'm',
    expandRowIDs: new Set(['']),
    // eslint-disable-next-line @typescript-eslint/no-empty-function
    changeColumn: (_val: string[]): void => {},
    // eslint-disable-next-line @typescript-eslint/no-empty-function
    changeMetricGraphMode: (_val: 'max' | 'min'): void => {},
    // eslint-disable-next-line @typescript-eslint/no-empty-function
    changeMaxDurationUnit: (_val: string): void => {},
    // eslint-disable-next-line @typescript-eslint/no-empty-function
    changeEntries: (_val: string): void => {},
    // eslint-disable-next-line @typescript-eslint/no-empty-function
    updateOverviewPage: () => {},
    // eslint-disable-next-line @typescript-eslint/no-empty-function
    updateDetailPage: () => {},
    // eslint-disable-next-line @typescript-eslint/no-empty-function
    changeExpandRowIDs: (_val: string, _type?: string): void => {},
    // eslint-disable-next-line @typescript-eslint/no-empty-function
    startTimer: () => {},
    // eslint-disable-next-line @typescript-eslint/no-empty-function
    closeTimer: (): void => {},
    // eslint-disable-next-line @typescript-eslint/no-empty-function
    lastRefresh: (): void => {}
});

export default AppContext;